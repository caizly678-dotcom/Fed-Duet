from collections import OrderedDict
from typing import Tuple, Union, List

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .adapter import Adapter
from torch.distributions.normal import Normal
from collections import Counter

# 新增: AnalystAttention 模块
from .analyst_attention import AnalystAttention

global_taskid = 0  # 初始化为0

# 添加全局门控信息收集器
global_gate_collector = {
    'image': {},  # 将存储 {layer_idx: {'gates': tensor, 'logits': tensor}}
    'text': {},  # 将存储 {layer_idx: {'gates': tensor, 'logits': tensor}}
    'current_task_id': None
}

# 这里不需要添加全局正交损失收集器

# 添加全局历史门控信息存储
global_historical_gates = {}


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` of shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out: List[torch.Tensor], multiply_by_gates: bool = True) -> torch.Tensor:
        """
        将各专家的输出按 gate 权重加和，恢复到原始 batch 维度。

        Args:
          expert_out: 长度为 num_experts 的列表，每项为 Tensor，形状 [n_i, d1, d2, …, dk]
          multiply_by_gates: 是否用 gate 值加权

        Returns:
          Tensor of shape [batch_size, d1, d2, …, dk]
        """
        # 1) 拼接所有专家输出，得到 stitched: [sum_i n_i, d1, d2, …, dk]
        stitched = torch.cat(expert_out, dim=0)

        # 2) 如果需要，用 gate 权重加权
        if multiply_by_gates:
            # self._nonzero_gates: [sum_i n_i]
            # reshape 为 [sum_i n_i, 1, 1, …, 1] 与 stitched 逐元素相乘
            shape = [self._nonzero_gates.size(0)] + [1] * (stitched.dim() - 1)
            gates = self._nonzero_gates.view(*shape)
            stitched = stitched * gates

        # 3) 初始化一个全零张量，形状为 [batch_size, d1, d2, …, dk]
        batch_size = self._gates.size(0)
        out_shape = [batch_size] + list(stitched.shape[1:])
        combined = torch.zeros(out_shape, device=stitched.device, dtype=stitched.dtype)

        # 4) 把 stitched 按照原来的 batch 索引累加回去
        #    self._batch_index: 长度 sum_i n_i，记录每行 stitched 属于哪个 batch
        combined.index_add_(0, self._batch_index, stitched)

        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if isinstance(x, tuple):
            x = x[0]
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=32, alpha=32.0, bias=True):
        """
        LoRA 低秩增量：
          y = W0 x + (alpha / r) * (B A^T) x
        其中 W0 是原始全连接层，A∈R^{in×r}, B∈R^{out×r} 是低秩矩阵。

        参数:
            r    : 秩 (默认8)
            alpha: 缩放系数 (默认16.0)
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = r/alpha  # 关键缩放因子

        # 冻结原始权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # LoRA参数 (修正维度顺序)
        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 原始权重初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # LoRA初始化: A用Kaiming，B初始化为零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始输出
        orig_out = F.linear(x, self.weight, self.bias)
        # 低秩增量
        lora_part = (x @ self.lora_A)  # [..., r]
        lora_part = lora_part @ self.lora_B.T  # [..., out_features]
        return orig_out + self.scaling * lora_part


class IncrementalLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=32, alpha=32.0, bias=True, orth_penalty=0.1):
        """
        增量式LoRA实现，为每个任务维护独立的LoRA参数，并支持正交约束。

        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            r: 秩 (默认16)
            alpha: 缩放系数 (默认8.0)
            bias: 是否使用偏置
            orth_penalty: 正交约束权重 (默认0.1)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = r/alpha  # 关键缩放因子
        self.orth_penalty = orth_penalty

        # 冻结原始权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # 初始化原始权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 任务级LoRA参数存储
        self.lora_As = nn.ParameterDict()  # 存储每个任务的A矩阵
        self.lora_Bs = nn.ParameterDict()  # 存储每个任务的B矩阵

        # 当前活跃任务ID
        self.current_task_id = None

        # 合并模式: 'current', 'all', 或自定义任务ID列表
        self.merge_mode = 'all'
        self.merge_task_ids = []

        # 初始化第一个任务的参数
        self.add_task_parameters('0')
        self.current_task_id = '0'

    def add_task_parameters(self, task_id):
        """为新任务添加LoRA参数"""
        task_id = str(task_id)  # 确保task_id是字符串

        if task_id in self.lora_As:
            return  # 如果已存在，不重复添加

        # 创建新任务的LoRA参数
        self.lora_As[task_id] = nn.Parameter(torch.empty(self.in_features, self.r))
        self.lora_Bs[task_id] = nn.Parameter(torch.empty(self.out_features, self.r))

        # 初始化新任务的参数
        nn.init.kaiming_uniform_(self.lora_As[task_id], a=math.sqrt(5))
        nn.init.zeros_(self.lora_Bs[task_id])

    def set_task(self, task_id):
        """设置当前活跃任务"""
        task_id = str(task_id)

        # 如果是新任务，添加参数
        if task_id not in self.lora_As:
            self.add_task_parameters(task_id)

        self.current_task_id = task_id

    def set_merge_mode(self, mode, task_ids=None):
        """
        设置参数合并模式

        参数:
            mode: 'current' - 仅使用当前任务参数
                  'all' - 使用所有任务参数
                  'custom' - 使用指定任务ID列表的参数
            task_ids: 当mode='custom'时使用的任务ID列表
        """
        self.merge_mode = mode
        if mode == 'custom' and task_ids is not None:
            self.merge_task_ids = [str(tid) for tid in task_ids]

    def compute_orthogonal_loss(self):
        """
        计算正交约束损失: L_ort = ∑_{i=1}^{t-1} ||A_i^T A_t||_F^2

        返回:
            orth_loss: 正交约束损失
        """
        if self.current_task_id is None or self.current_task_id == '0':
            return torch.tensor(0.0, device=self.weight.device)

        current_A = self.lora_As[self.current_task_id]
        orth_loss = torch.tensor(0.0, device=current_A.device)

        # 计算当前任务A与所有先前任务A的正交损失
        for task_id, A in self.lora_As.items():
            if task_id != self.current_task_id:
                # 确保A与current_A在同一设备上
                A = A.to(current_A.device)

                # 计算 ||A_i^T A_t||_F^2
                orth_term = torch.matmul(A.t(), current_A)
                orth_loss += torch.norm(orth_term, p='fro')**2
                orth_loss.to('cuda')

        return orth_loss * self.orth_penalty

    def forward(self, x):
        """
        前向传播，根据合并模式使用不同的LoRA参数

        参数:
            x: 输入张量

        返回:
            output: 模型输出
            orth_loss: 正交约束损失 (如果在训练模式下)
        """
        # 处理输入可能是元组的情况
        if isinstance(x, tuple):
            x = x[0]

        # 原始输出
        orig_out = F.linear(x, self.weight, self.bias)

        # 确定要使用的任务ID列表
        if self.merge_mode == 'current' and self.current_task_id is not None:
            task_ids = [self.current_task_id]
        elif self.merge_mode == 'all':
            task_ids = list(self.lora_As.keys())
        elif self.merge_mode == 'custom':
            task_ids = [tid for tid in self.merge_task_ids if tid in self.lora_As]
        else:
            # 默认使用当前任务
            task_ids = [self.current_task_id] if self.current_task_id is not None else []

        # 累加所有选定任务的LoRA输出
        lora_out = torch.zeros_like(orig_out)
        for task_id in task_ids:
            lora_A = self.lora_As[task_id]
            lora_B = self.lora_Bs[task_id]

            # 确保 lora_A 和 lora_B 与输入 x 在同一设备上
            lora_A = lora_A.to(x.device)
            lora_B = lora_B.to(x.device)

            # 计算该任务的LoRA部分
            lora_part = (x @ lora_A) @ lora_B.t()
            lora_out += lora_part

        # 应用缩放因子
        lora_out = lora_out * self.scaling

        # 计算正交损失（仅在训练模式下）
        #orth_loss = self.compute_orthogonal_loss() if self.training else torch.tensor(0.0, device=x.device)

        return orig_out + lora_out#, orth_loss

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, text_or_image=None, layer_idx=None,
                 cfg=None):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.use_mlp = cfg.use_mlp if cfg is not None and hasattr(cfg, 'use_mlp') else True
        if self.use_mlp:
            # print("拥有平行的mlp层")
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # self.is_train 已被移除, 将使用 self.training
        self.step = cfg.task_num if cfg is not None and hasattr(cfg, 'task_num') else 1
        #print("task_num:",self.step)

        self.cfg = cfg
        self.use_auxiliary_loss = cfg.use_auxiliary_loss if cfg is not None and hasattr(cfg,'use_auxiliary_loss') else False
        self.top_k = cfg.top_k if cfg is not None and hasattr(cfg, 'top_k') else 1

        #print("d_model size:", d_model)
        self.experts_num = cfg.num_experts if cfg is not None and hasattr(cfg, 'num_experts') else 22
        # print("专家数量为：", self.experts_num)

        # self.ffn_num = cfg.ffn_num if cfg is not None and hasattr(cfg, 'ffn_num') else 64
        self.ffn_num = getattr(cfg, 'ffn_num', 64)  # 默认值为64

        self.shared_expert_num = cfg.n_shared_experts if cfg is not None and hasattr(cfg, 'n_shared_experts') else 1
        self.drop_out = cfg.drop_out if cfg is not None and hasattr(cfg, 'drop_out') else 0.1
        #缩放因子
        # if cfg.use_adaptive_scalar:
        #     #设置为可更新参数
        #     self.moe_scalar = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        #     self.min_moe_scalar = 0.1
        #     self.max_moe_scalar = 3.0
        #     #print("moe_scalar:",self.moe_scalar)
        # else:
        self.moe_scalar = cfg.moe_scalar if cfg is not None and hasattr(cfg, 'moe_scalar') else 1
        #print(self.moe_scalar)

        self.r = cfg.lora_rank if hasattr(cfg, 'lora_rank') else 32
        self.alpha = cfg.lora_alpha if hasattr(cfg, 'lora_alpha') else 32
        # 如果cfg没有adapter_scalar,默认值为0.1
        self.adapter_scalar = cfg.adapter_scalar if hasattr(cfg, 'adapter_scalar') else 0.3

        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.noisy_gating = True

        # self.adaptmlp_list = nn.ModuleList([
        #     nn.Sequential(OrderedDict([
        #         ("c_fc", LoRALinear(d_model, self.ffn_num, r=self.r, alpha=self.alpha)),
        #         ("gelu", QuickGELU()),
        #         ("dropout", nn.Dropout(self.drop_out)),
        #         ("c_proj", LoRALinear(self.ffn_num, d_model, r=self.r, alpha=self.alpha))
        #     ]))
        #     for _ in range(self.experts_num)
        # ])
        self.adaptmlp_list = nn.ModuleList()
        for i in range(self.experts_num):
            self.adaptmlp = Adapter(d_model=d_model, dropout=0.4, bottleneck=self.ffn_num,
                                    init_option='lora',
                                    adapter_scalar=self.adapter_scalar,
                                    adapter_layernorm_option='none',
                                    )
            self.adaptmlp_list.append(self.adaptmlp)

        self.shared_expert = nn.ModuleList()
        for i in range(self.shared_expert_num):
            self.shared_expert_unit = Adapter(d_model=d_model, dropout=0.4, bottleneck=self.ffn_num*self.experts_num,
                                                init_option='lora',
                                                adapter_scalar=self.adapter_scalar,
                                                adapter_layernorm_option='none',
                                                )
            self.shared_expert.append(self.shared_expert_unit)

        # self.shared_expert = nn.ModuleList([
        #     nn.Sequential(OrderedDict([
        #         ("c_fc", LoRALinear(d_model, self.ffn_num * self.experts_num, r=self.r, alpha=self.alpha)),
        #         ("gelu", QuickGELU()),
        #         ("dropout", nn.Dropout(self.drop_out)),
        #         ("c_proj", LoRALinear( self.ffn_num * self.experts_num,d_model, r=self.r, alpha=self.alpha))
        #     ])) for _ in range(self.shared_expert_num)
        # ])
        self.text_or_image = text_or_image
        self.layer_idx = layer_idx  # 添加层索引
        self.model = None  # 将在初始化后设置对主模型的引用

        # 动态专家和top-k配置
        self.initial_experts = self.experts_num
        self.max_experts = 16  # 最大专家数量
        self.experts_per_task = 1  # 每个新任务增加的专家数量
        self.initial_k = self.top_k
        self.max_k = 4  # 最大top-k值
        self.k_strategy = 'linear'  # top-k增长策略: 'linear', 'step', 'exp', 'log'

        if text_or_image == 'text':
            self.choose_map_text = torch.zeros([self.experts_num])
        else:
            self.choose_map_image = torch.zeros([self.experts_num])
        # ------------ Router 参数尺寸根据输入特征维度动态确定 ------------
        # image 模态 + AnalystAttention 时，rep_feat = concat(x, x_att) ⇒ 2*d_model
        self.router_in_dim = d_model * 2 if (text_or_image == 'image') else d_model

        self.router_list = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()
        for i in range(self.step):
            self.router_list.append(
                nn.Parameter(torch.zeros(self.router_in_dim, self.experts_num), requires_grad=True)
            )
            self.w_noise_list.append(
                nn.Parameter(torch.zeros(self.router_in_dim, self.experts_num), requires_grad=True)
            )

        # -------- Prompt 条件化 Cross-Attention Analyst (仅 image 模态) --------
        analyst_fusion = getattr(cfg, "analyst_fusion", "add") if cfg is not None else "add"
        use_ca = getattr(cfg, "use_cross_attention", False) if cfg is not None else True

        if use_ca and text_or_image == 'image' and analyst_fusion != 'none' and use_ca is True:
            # 使用 cfg.ctx_dim (由 CLIP 初始化时写入) 与 d_model 可能不同，例如 512 vs 768
            d_ctx = getattr(cfg, "ctx_dim", d_model)
            self.analyst = AnalystAttention(d_feat=d_model, d_ctx=d_ctx, fusion_mode=analyst_fusion)
        else:
            self.analyst = None

        # ------------ Router 参数尺寸根据输入特征维度动态确定 ------------
        if self.analyst is not None and analyst_fusion == 'concat':
            self.router_in_dim = d_model * 2
        else:
            self.router_in_dim = d_model

        self.router_list = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()
        for i in range(self.step):
            self.router_list.append(
                nn.Parameter(torch.zeros(self.router_in_dim, self.experts_num), requires_grad=True)
            )
            self.w_noise_list.append(
                nn.Parameter(torch.zeros(self.router_in_dim, self.experts_num), requires_grad=True)
            )

    def expand_experts_for_new_task(self, task_id):
        """
        为新任务扩展专家数量

        参数:
            task_id: 新任务的ID

        返回:
            bool: 是否成功添加了专家
        """
        # 如果已达到最大专家数量，则不再增加
        if self.experts_num >= self.max_experts:
            return False

        # 计算要添加的专家数量
        experts_to_add = min(self.experts_per_task, self.max_experts - self.experts_num)

        if experts_to_add <= 0:
            return False

        # 保存旧的专家数量
        old_expert_count = self.experts_num

        # 创建新专家
        for _ in range(experts_to_add):
            new_expert = nn.Sequential(
                nn.Linear(self.adaptmlp_list[0][0].in_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.adaptmlp_list[0][-1].out_features)
            ).to(self.adaptmlp_list[0][0].weight.device)

            # 初始化新专家参数（基于现有专家的平均值加噪声）
            with torch.no_grad():
                # 复制第一个专家的参数并添加少量噪声
                for new_param, old_param in zip(new_expert.parameters(), self.adaptmlp_list[0].parameters()):
                    new_param.data.copy_(old_param.data + 0.01 * torch.randn_like(old_param.data))

            # 添加到专家列表
            self.adaptmlp_list.append(new_expert)

        # 更新专家数量
        self.experts_num += experts_to_add

        # 更新选择映射
        if self.text_or_image == 'text':
            self.choose_map_text = torch.cat([
                self.choose_map_text,
                torch.zeros(experts_to_add, device=self.choose_map_text.device)
            ])
        else:
            self.choose_map_image = torch.cat([
                self.choose_map_image,
                torch.zeros(experts_to_add, device=self.choose_map_image.device)
            ])

        # 更新路由器参数
        for i in range(self.step):
            old_router = self.router_list[i]
            old_noise = self.w_noise_list[i]

            # 创建新的路由器参数
            new_router = nn.Parameter(
                torch.zeros(
                    old_router.shape[0],
                    self.experts_num,
                    device=old_router.device
                ),
                requires_grad=True
            )
            new_noise = nn.Parameter(
                torch.zeros(
                    old_noise.shape[0],
                    self.experts_num,
                    device=old_noise.device
                ),
                requires_grad=True
            )

            # 复制旧参数
            with torch.no_grad():
                new_router.data[:, :old_expert_count] = old_router.data
                new_noise.data[:, :old_expert_count] = old_noise.data

                # 初始化新专家的路由参数（基于现有专家的平均值加噪声）
                router_mean = old_router.data.mean(dim=1, keepdim=True)
                noise_mean = old_noise.data.mean(dim=1, keepdim=True)

                for j in range(old_expert_count, self.experts_num):
                    new_router.data[:, j] = router_mean.squeeze(1) + 0.01 * torch.randn_like(router_mean.squeeze(1))
                    new_noise.data[:, j] = noise_mean.squeeze(1) + 0.01 * torch.randn_like(noise_mean.squeeze(1))

            # 替换路由器参数
            self.router_list[i] = new_router
            self.w_noise_list[i] = new_noise

        return True

    def update_top_k(self, task_id, total_tasks):
        """
        根据任务进度动态更新top-k值

        参数:
            task_id: 当前任务ID
            total_tasks: 总任务数量

        返回:
            int: 更新后的top-k值
        """
        # 确保task_id和total_tasks有效
        if task_id < 0 or total_tasks <= 0:
            return self.top_k

        # 计算任务进度
        progress = task_id / max(1, total_tasks - 1) if total_tasks > 1 else 0

        # 根据策略计算新的k值
        if self.k_strategy == 'linear':
            # 线性增长
            k = self.initial_k + (self.max_k - self.initial_k) * progress

        elif self.k_strategy == 'step':
            # 阶梯式增长
            step_points = [0.25, 0.5, 0.75]
            k_values = [self.initial_k, self.initial_k + 1, self.initial_k + 2, self.max_k]

            k = k_values[0]
            for i, point in enumerate(step_points):
                if progress >= point:
                    k = k_values[i + 1]

        elif self.k_strategy == 'exp':
            # 指数增长（后期增长更快）
            k = self.initial_k + (self.max_k - self.initial_k) * (progress ** 2)

        elif self.k_strategy == 'log':
            # 对数增长（前期增长更快）
            if task_id == 0:
                k = self.initial_k
            else:
                # 使用对数函数，确保在progress=0时k=initial_k，在progress=1时k接近max_k
                k = self.initial_k + (self.max_k - self.initial_k) * (
                        math.log(1 + 9 * progress) / math.log(10)
                )
        else:
            # 默认不变
            k = self.top_k

        # 确保k是整数且在有效范围内
        new_k = max(self.initial_k, min(self.max_k, round(k)))

        # 更新当前k值
        old_k = self.top_k
        self.top_k = new_k

        return self.top_k

    def set_lora_task(self, task_id):
        """
        设置所有增量LoRA层的当前任务ID

        参数:
            task_id: 当前任务ID
        """
        # 设置专家MLP中的LoRA层
        for expert in self.adaptmlp_list:
            if hasattr(expert[0], 'set_task'):  # c_fc
                expert[0].set_task(task_id)
            if hasattr(expert[3], 'set_task'):  # c_proj
                expert[3].set_task(task_id)

        # 设置共享专家中的LoRA层
        for shared in self.shared_expert:
            if hasattr(shared[0], 'set_task'):  # c_fc
                shared[0].set_task(task_id)
            if hasattr(shared[3], 'set_task'):  # c_proj
                shared[3].set_task(task_id)

    def compute_lora_orthogonal_loss(self):
        """
        计算所有增量LoRA层的正交损失

        返回:
            orth_loss: 所有LoRA层的正交损失总和
        """
        total_orth_loss = torch.tensor(0.0, device=self.mean.device)

        # 收集专家MLP中的LoRA层正交损失
        for expert in self.adaptmlp_list:
            if hasattr(expert[0], 'compute_orthogonal_loss'):  # c_fc
                loss = expert[0].compute_orthogonal_loss()
                # 确保损失在正确的设备上
                loss = loss.to(total_orth_loss.device)
                total_orth_loss += loss
            if hasattr(expert[3], 'compute_orthogonal_loss'):  # c_proj
                loss = expert[3].compute_orthogonal_loss()
                # 确保损失在正确的设备上
                loss = loss.to(total_orth_loss.device)
                total_orth_loss += loss

        # 收集共享专家中的LoRA层正交损失
        for shared in self.shared_expert:
            if hasattr(shared[0], 'compute_orthogonal_loss'):  # c_fc
                loss = shared[0].compute_orthogonal_loss()
                # 确保损失在正确的设备上
                loss = loss.to(total_orth_loss.device)
                total_orth_loss += loss
            if hasattr(shared[3], 'compute_orthogonal_loss'):  # c_proj
                loss = shared[3].compute_orthogonal_loss()
                # 确保损失在正确的设备上
                loss = loss.to(total_orth_loss.device)
                total_orth_loss += loss

        #print("total_orth_loss:", total_orth_loss)
        return total_orth_loss

    def attention(self, x: torch.Tensor):
        if isinstance(x, tuple):
            x = x[0]
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # clean_logits计算的是
        clean_logits = x @ w_gate.to(x)
        # ----------  数值稳定化：零均值 / 单位方差 + 截断  ----------
        if train:  # 仅在训练阶段执行，推理保持原值
            # mean = clean_logits.mean(dim=-1, keepdim=True)
            # std = clean_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)
            # clean_logits = (clean_logits - mean) / std
            pass
           

        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            # 防止 noise_stddev 过小或为 NaN
            noise_stddev = torch.nan_to_num(noise_stddev, nan=1.0)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # 使用当前的top_k值计算门控权重
        # 确保top_k不超过专家数量且至少为1
        current_k = max(1, min(self.top_k, self.experts_num))

        # 确保logits不包含NaN值
        if torch.isnan(logits).any():
            # 如果发现NaN，将其替换为很小的负值
            logits = torch.where(torch.isnan(logits), torch.tensor(-1e10, device=logits.device), logits)

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(current_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :current_k]
        top_k_indices = top_indices[:, :current_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        # TODO:
        top_k_gates = top_k_gates.to(zeros.dtype)  # 强制类型转换

        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, clean_logits

    def compute_auxiliary_loss(self, gates, logits):
        """
        计算MoE的辅助损失：局部熵损失和全局熵损失

        参数:
            gates: 门控矩阵 [batch_size, num_experts]
            logits: 路由器输出的原始logits [batch_size, num_experts]

        返回:
            aux_loss: 辅助损失，包括局部熵和全局熵损失的加权和
        """
        # 1. 计算局部熵损失 - 对每个token的分布计算熵
        # 为防止数值问题，对logits应用softmax得到完整的概率分布
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # 计算每个样本的熵: -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, num_experts]
        entropy_per_sample = -(probs * log_probs).sum(dim=-1)  # [batch_size]

        # 局部熵损失是所有样本熵的平均
        local_entropy = entropy_per_sample.mean()

        # 2. 计算全局熵损失 - 对所有token的平均分布计算熵
        # 计算平均分布
        avg_probs = probs.mean(dim=0)  # [num_experts]

        # 计算平均分布的熵
        global_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()

        # 3. 计算负载平衡损失 - 使用变异系数的平方
        # 计算每个专家的负载
        importance = gates.sum(0)  # [num_experts]
        load_balancing_loss = self.cv_squared(importance)

        # 4. 合并损失
        # 在实际应用中，可能需要调整这些权重
        alpha_local = self.cfg.alpha_local if self.cfg is not None and hasattr(self.cfg,
                                                                               'alpha_local') else 0.1  # 局部熵权重
        alpha_global = self.cfg.alpha_global if self.cfg is not None and hasattr(self.cfg,
                                                                                 'alpha_global') else 0.1  # 全局熵权重
        alpha_load = self.cfg.alpha_load if self.cfg is not None and hasattr(self.cfg, 'alpha_load') else 0.1  # 负载平衡权重

        # 局部熵应该被最大化(负的局部熵被最小化)，全局熵应该被最小化
        aux_loss = -alpha_local * local_entropy + alpha_global * global_entropy + alpha_load * load_balancing_loss

        return aux_loss

    def forward(self, x: torch.Tensor):
        # 声明使用全局变量
        global global_taskid
        global global_gate_collector

        # 确保接收到的是tensor而不是tuple
        if isinstance(x, tuple):
            x = x[0]

        # 初始化辅助损失为设备上的张量
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        orth_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 自注意力处理
        ln_1_output = self.ln_1(x)
        attn_output = self.attention(ln_1_output)
        x = x + attn_output

        if global_taskid is not None:
            # 获取路由输入（第一个token用于决定路由） 每个序列的代表性token特征
            x_re = x.permute(1, 0, 2)[:, 0, :]

            # 若存在 Analyst 模块且已注入 ctx_vector，则先做 Cross-Attention 分析
            rep_feat = x_re
            if self.analyst is not None and hasattr(self, "ctx_vector") and self.ctx_vector is not None:
                rep_feat = self.analyst(x_re, self.ctx_vector)  # (B, 2*d_model)

            # 计算路由权重，现在返回logits用于计算辅助损失
            #print("global_taskid", global_taskid)
            gates, load, logits = self.noisy_top_k_gating(
                rep_feat,
                self.training,
                # self.router_list[global_taskid],
                # self.w_noise_list[global_taskid]
                self.router_list[0],
                self.w_noise_list[0]
            )

            # 收集门控信息用于计算跨模态路由一致性损失和专家迁移稳定性损失
            if hasattr(self, 'text_or_image') and hasattr(self, 'layer_idx'):
                modality = self.text_or_image  # 'image' 或 'text'
                if modality in global_gate_collector:
                    global_gate_collector[modality][self.layer_idx] = {
                        'gates': gates,
                        'logits': logits
                    }

            # 计算辅助损失
            if self.training and self.use_auxiliary_loss:
                # 计算MoE辅助损失
                task_aux_loss = self.compute_auxiliary_loss(gates, logits)
                aux_loss += task_aux_loss
                #print("aux_loss:",aux_loss)

            # 计算专家使用统计
            importance = gates.sum(0)
            nonzero_indices = torch.nonzero(gates)
            counter = Counter(nonzero_indices[:, 1].tolist())
            for number, count in counter.items():
                if self.text_or_image == 'text':
                    self.choose_map_text[number] = self.choose_map_text[number] + count
                else:
                    self.choose_map_image[number] = self.choose_map_image[number] + count

            # 准备输入数据
            batch_size = x.shape[1]
            seq_len = x.shape[0]
            hidden_dim = x.shape[2]

            # 将输入转换为 [batch*seq_len, hidden_dim] 形状便于处理
            x_reshaped = x.permute(1, 0, 2).reshape(-1, hidden_dim)

            # 1. 处理路由专家 (Routed Experts)
            dispatcher = SparseDispatcher(self.experts_num, gates)
            expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))

            expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],x.shape[0], x.shape[2]).to(x),add_residual=False)
                              for i in range(self.experts_num)]
            i = 0
            while i < len(expert_outputs):
                if expert_outputs[i].shape[0] == 0:
                    expert_outputs.pop(i)
                else:
                    expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                    i += 1
            # for i in range(self.experts_num):
            #     if len(expert_inputs[i]) > 0:
            #         # 重新组织形状: [batch_expert, seq_len*hidden] -> [batch_expert*seq_len, hidden]
            #         batch_size_expert = expert_inputs[i].shape[0]
            #
            #         # 将专家输入重塑为三维张量，然后转换为二维以输入MLP
            #         reshaped_input = expert_inputs[i].reshape(batch_size_expert, seq_len, hidden_dim)
            #         # 将序列维度和批次维度合并
            #         flat_input = reshaped_input.reshape(-1, hidden_dim)
            #
            #         # 通过MLP处理 - 现在只返回输出，不返回正交损失
            #         output = self.adaptmlp_list[i](flat_input)
            #
            #         # 恢复原始形状
            #         output = output.reshape(batch_size_expert, seq_len, hidden_dim)
            #         # 重新扁平化为dispatcher期望的形状
            #         output = output.reshape(batch_size_expert, -1)
            #         expert_outputs.append(output)
            #     else:
            #         expert_outputs.append(torch.zeros(0, device=x.device))

            # 基于路由权重组合专家输出
            y_routed = dispatcher.combine(expert_outputs)
            y_routed = y_routed.view(batch_size, seq_len, hidden_dim)

            # 2. 处理共享专家 (Shared Experts)
            shared_outputs = []
            if self.shared_expert_num > 0:
                for i in range(self.shared_expert_num):
                    # 将输入reshape为2D张量，适配Adapter的输入要求
                    x_reshaped = x.permute(1, 0, 2).reshape(-1, x.shape[-1])
                    shared_output = self.shared_expert[i](x_reshaped, add_residual=False)
                    # 将输出reshape回原始形状
                    shared_output = shared_output.reshape(x.shape[1], x.shape[0], x.shape[2]).permute(1, 0, 2)
                    shared_outputs.append(shared_output)
                if self.shared_expert_num > 1:
                    y_shared = torch.stack(shared_outputs).mean(0)
                else:
                    y_shared = shared_outputs[0]
                # 确保y_shared的维度与y匹配
                y_shared = y_shared.permute(1, 0, 2)  # 调整维度顺序以匹配y

            # 3. 融合路由专家和共享专家的输出
            # 简单的加权平均法
            # 对于每个token，共享专家权重为50%，路由专家权重为50%
            if self.shared_expert_num > 0:
                #alpha = 0.5  # 可调整的权重参数
                y_combined =  y_routed + y_shared
            else:
                y_combined = y_routed
            if self.moe_scalar is not None:
                y_combined = y_combined*self.moe_scalar
            # 应用残差连接和原始MLP
            ln_2_output = self.ln_2(x)
            if self.use_mlp:
                mlp_output = self.mlp(ln_2_output)
                x = x + mlp_output + y_combined.permute(1, 0, 2)
            else:
                x = x + y_combined.permute(1, 0, 2)
        else:
            #print(1)
            ln_2_output = self.ln_2(x)
            if self.use_mlp:
                mlp_output = self.mlp(ln_2_output)
                x = x + mlp_output
            else:
                x = x + ln_2_output

        # 返回输出、MoE辅助损失和LoRA正交损失的总和
        # 计算LoRA正交损失
        #if self.use_orth_loss:
            #lora_orth_loss = self.compute_lora_orthogonal_loss()
            #orth_loss += lora_orth_loss

        total_loss = aux_loss + orth_loss
        return x, total_loss


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, text_or_image='text',
                 cfg=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_or_image = text_or_image
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, text_or_image, layer_idx=i, cfg=cfg)
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for block in self.resblocks:
            output = block(x)
            # --- 鲁棒性处理：确保即使block未返回损失，也能正常处理 ---
            if isinstance(output, tuple) and len(output) >= 2:
                x, block_aux_loss = output
                if block_aux_loss is not None:
                    total_aux_loss += block_aux_loss
            else:
                x = output # 如果只返回x，则损失视为0

        return x, total_aux_loss


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 text_or_image='image', cfg=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, text_or_image=text_or_image, cfg=cfg)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.ctx_vector = None  # 将由上层模型注入

    # ---------- Prompt ctx 向量注入接口 ----------
    def set_ctx_vector(self, ctx_vec: torch.Tensor):
        """由上层模型调用，用于在一次 forward 之前传入 Prompt 引导向量。"""
        self.ctx_vector = ctx_vec.detach() if ctx_vec is not None else None

    def forward(self, x: torch.Tensor):
        # 确保接收到的是tensor而不是tuple
        if isinstance(x, tuple):
            x = x[0]

        # 初始化辅助损失
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 嵌入处理
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        # -------- 在进入 Transformer 之前注入 ctx_vector --------
        if self.ctx_vector is not None:
            for blk in self.transformer.resblocks:
                blk.ctx_vector = self.ctx_vector
        else:
            for blk in self.transformer.resblocks:
                if hasattr(blk, 'ctx_vector'):
                    blk.ctx_vector = None

        # transformer可能返回辅助损失
        transformer_output = self.transformer(x)
        x, trans_aux_loss = transformer_output if isinstance(transformer_output, tuple) else (transformer_output, torch.tensor(0.0, device=x.device, dtype=x.dtype))
        if trans_aux_loss is None:
            trans_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_aux_loss += trans_aux_loss

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, total_aux_loss


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 baseline=False,
                 cfg=None
                 ):
        super().__init__()
        self.baseline = baseline
        self.cfg = cfg
        self.historical_gates = {}

        self.context_length = context_length

        # 提前记录 Prompt ctx 维度 (text transformer hidden dim) 到 cfg，后续 AnalystAttention 依赖
        if cfg is not None:
            try:
                # 对 OmegaConf 的 DictConfig 使用 open_dict 临时解锁 struct
                from omegaconf import DictConfig, OmegaConf
                if isinstance(cfg, DictConfig):
                    with OmegaConf.open_dict(cfg):
                        cfg.ctx_dim = transformer_width
                else:
                    # 普通 Namespace / dict
                    cfg.ctx_dim = transformer_width
            except Exception:
                # 若无法写入，忽略，不影响后续；ResidualBlock 将 fallback 到 d_model
                pass

        # task_num cifar100
        self.task_num = 100 / cfg.initial_increment
        # 使用全局门控信息收集器和历史门控信息存储

        # 添加损失权重配置
        self.config = {
            'distill_temperature': 2.0,  # 知识蒸馏温度
            'alignment_temperature': 0.10,  # 对比对齐温度
            'contrastive_temp': 0.1,           # CLIP风格对比损失的温度参数
            'orth_penalty': 0.1,                # LoRA正交约束权重
        }

        # 设置损失权重和类型，如果cfg中未定义，则使用新的默认值
        # cross_modal_loss_type 可选: 'clip_style', 'weighted_l2', 'js_div'
        # stability_loss_type 可选: 'clip_style', 'weighted_l2', 'js_div'
        self.config['cross_modal_weight'] = getattr(cfg, 'cross_modal_weight', 0)
        self.config['stability_weight'] = getattr(cfg, 'stability_weight', 0)
        self.config['cross_modal_loss_type'] = getattr(cfg, 'cross_modal_loss_type', 'clip_style')
        self.config['stability_loss_type'] = getattr(cfg, 'stability_loss_type', 'weighted_l2')

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                text_or_image='image',
                cfg=cfg
            )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_or_image='text',
            cfg=cfg
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def set_historical_gates(self, historical_gates):
        self.historical_gates = historical_gates

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def prepare_for_new_task(self, task_id, total_tasks=None):
        """
        为新任务准备模型，包括增加专家数量和更新top-k值

        参数:`
            task_id: 当前任务ID
            total_tasks: 总任务数量（可选）
        """
        # 如果未提供总任务数，使用默认值
        if total_tasks is None:
            total_tasks = int(self.task_num)

        # 1. 更新图像模态的专家和top-k
        if isinstance(self.visual, VisualTransformer):
            for block in self.visual.transformer.resblocks:
                if hasattr(block, 'expand_experts_for_new_task'):
                    block.expand_experts_for_new_task(task_id)
                if hasattr(block, 'update_top_k'):
                    block.update_top_k(task_id, total_tasks)

        # 2. 更新文本模态的专家和top-k
        for block in self.transformer.resblocks:
            if hasattr(block, 'expand_experts_for_new_task'):
                block.expand_experts_for_new_task(task_id)
            if hasattr(block, 'update_top_k'):
                block.update_top_k(task_id, total_tasks)

        print(f"模型已准备好处理任务 {task_id}，专家数量和top-k值已更新")
    def set_experts_per_task(self, experts_per_task):
        """
        设置每个新任务增加的专家数量

        参数:
            experts_per_task: 每个新任务增加的专家数量
        """
        # 更新图像模态的专家增长配置
        if isinstance(self.visual, VisualTransformer):
            for block in self.visual.transformer.resblocks:
                if hasattr(block, 'experts_per_task'):
                    block.experts_per_task = experts_per_task

        # 更新文本模态的专家增长配置
        for block in self.transformer.resblocks:
            if hasattr(block, 'experts_per_task'):
                block.experts_per_task = experts_per_task

        print(f"已设置每个新任务增加 {experts_per_task} 个专家")

    def set_top_k_strategy(self, max_k=4, strategy='linear'):
        """
        设置top-k增长策略

        参数:
            max_k: 最大top-k值
            strategy: 增长策略，可选 'linear', 'step', 'exp', 'log'
        """
        # 更新图像模态的top-k配置
        if isinstance(self.visual, VisualTransformer):
            for block in self.visual.transformer.resblocks:
                if hasattr(block, 'max_k') and hasattr(block, 'k_strategy'):
                    block.max_k = max_k
                    block.k_strategy = strategy

        # 更新文本模态的top-k配置
        for block in self.transformer.resblocks:
            if hasattr(block, 'max_k') and hasattr(block, 'k_strategy'):
                block.max_k = max_k
                block.k_strategy = strategy

        print(f"已设置top-k增长策略为 {strategy}，最大值为 {max_k}")

    def set_lora_task_id(self, task_id):
        """
        设置所有增量LoRA层的当前任务ID

        参数:
            task_id: 当前任务ID，用于激活对应任务的LoRA参数
        """
        # 1. 设置图像模态的LoRA任务ID
        if isinstance(self.visual, VisualTransformer):
            for block in self.visual.transformer.resblocks:
                if hasattr(block, 'set_lora_task'):
                    block.set_lora_task(task_id)

        # 2. 设置文本模态的LoRA任务ID
        for block in self.transformer.resblocks:
            if hasattr(block, 'set_lora_task'):
                block.set_lora_task(task_id)

        print(f"已设置LoRA任务ID为 {task_id}")



    def set_lora_merge_mode(self, mode='current', task_ids=None):
        """
        设置所有增量LoRA层的合并模式

        参数:
            mode: 合并模式，可选 'current', 'all', 'custom'
            task_ids: 当mode='custom'时使用的任务ID列表
        """
        # 设置图像模态的LoRA合并模式
        if isinstance(self.visual, VisualTransformer):
            for block in self.visual.transformer.resblocks:
                self._set_block_lora_merge_mode(block, mode, task_ids)

        # 设置文本模态的LoRA合并模式
        for block in self.transformer.resblocks:
            self._set_block_lora_merge_mode(block, mode, task_ids)

        print(f"已设置LoRA合并模式为 {mode}")

    def _set_block_lora_merge_mode(self, block, mode, task_ids=None):
        """
        为单个块设置LoRA合并模式

        参数:
            block: ResidualAttentionBlock实例
            mode: 合并模式
            task_ids: 当mode='custom'时使用的任务ID列表
        """
        # 设置专家MLP中的LoRA层
        for expert in block.adaptmlp_list:
            for layer_idx in [0, 3]:  # c_fc和c_proj
                if hasattr(expert[layer_idx], 'set_merge_mode'):
                    expert[layer_idx].set_merge_mode(mode, task_ids)

        # 设置共享专家中的LoRA层
        for shared in block.shared_expert:
            for layer_idx in [0, 3]:  # c_fc和c_proj
                if hasattr(shared[layer_idx], 'set_merge_mode'):
                    shared[layer_idx].set_merge_mode(mode, task_ids)

    def expert_stability_js_div(self, current_gates, previous_gates):
        """
        使用JS散度计算专家迁移稳定性损失

        参数:
            current_gates: 当前任务的门控权重 [curr_batch_size, num_experts]
            previous_gates: 先前任务的门控权重 [prev_batch_size, num_experts]

        返回:
            js_loss: JS散度损失
        """
        # 归一化门控权重
        current_probs = F.softmax(current_gates, dim=-1)  # [curr_batch_size, num_experts]
        previous_probs = F.softmax(previous_gates, dim=-1).detach()  # [prev_batch_size, num_experts]

        # 计算批次级别的平均分布
        curr_mean_probs = current_probs.mean(dim=0)  # [num_experts]
        prev_mean_probs = previous_probs.mean(dim=0)  # [num_experts]

        # 计算JS散度
        mean_probs = (curr_mean_probs + prev_mean_probs) * 0.5

        # KL(curr || mean)
        kl_curr = F.kl_div(
            mean_probs.log(),
            curr_mean_probs,
            reduction='batchmean'
        )

        # KL(prev || mean)
        kl_prev = F.kl_div(
            mean_probs.log(),
            prev_mean_probs,
            reduction='batchmean'
        )

        # JS散度 = 0.5 * (KL(curr || mean) + KL(prev || mean))
        js_div = 0.5 * (kl_curr + kl_prev)
        #print("JS散度损失:", js_div.item())
        return js_div

    def expert_stability_clip_style(self, current_gates, previous_gates):
        """
        使用CLIP风格的对比损失计算专家迁移稳定性损失

        参数:
            current_gates: 当前任务的门控权重 [curr_batch_size, num_experts]
            previous_gates: 先前任务的门控权重 [prev_batch_size, num_experts]

        返回:
            contrastive_loss: CLIP风格的对比损失
        """
        # 归一化门控权重
        current_probs = F.softmax(current_gates, dim=-1)  # [curr_batch_size, num_experts]
        previous_probs = F.softmax(previous_gates, dim=-1).detach()  # [prev_batch_size, num_experts]

        # 计算批次级别的平均分布
        curr_mean_probs = current_probs.mean(dim=0, keepdim=True)  # [1, num_experts]
        prev_mean_probs = previous_probs.mean(dim=0, keepdim=True)  # [1, num_experts]

        # 归一化平均分布
        curr_mean_probs = F.normalize(curr_mean_probs, dim=1)
        prev_mean_probs = F.normalize(prev_mean_probs, dim=1)

        # 计算相似度
        temperature = self.config.get('contrastive_temp', 0.07)
        logits = torch.matmul(curr_mean_probs, prev_mean_probs.t()) / temperature

        # 目标是最大化相似度（最小化负相似度）
        return -logits.mean()

    def expert_transfer_stability_loss(self, current_gates, previous_gates):
        """
        计算专家使用在任务变化时的稳定性损失，根据配置选择不同的损失计算方法

        参数:
            current_gates: 当前任务的门控权重 [curr_batch_size, num_experts]
            previous_gates: 先前任务的门控权重 [prev_batch_size, num_experts]

        返回:
            stability_loss: 专家迁移稳定性损失
        """
        # 根据配置选择损失计算方法
        loss_type = self.config.get('stability_loss_type', 'js_div')

        if loss_type == 'weighted_l2':
            current_probs = F.softmax(current_gates, dim=-1)
            previous_probs = F.softmax(previous_gates, dim=-1).detach()
            
            curr_mean_probs = current_probs.mean(dim=0)
            prev_mean_probs = previous_probs.mean(dim=0)
            
            return self.weighted_distance_loss(curr_mean_probs, prev_mean_probs)
        elif loss_type == 'clip_style':
            return self.expert_stability_clip_style(current_gates, previous_gates)
        else:  # 默认使用JS散度
            return self.expert_stability_js_div(current_gates, previous_gates)

    def weighted_distance_loss(self, p_dist, q_dist):
        """
        使用加权L2距离计算两个概率分布之间的损失。
        权重由两个分布中对应位置的概率最大值确定。
        可以处理实例级 [B, E] 和批次平均级 [E] 的输入。
        """
        # .detach() is equivalent to stop_gradient
        weights = torch.max(p_dist, q_dist).detach()

        # 加权L2距离的平方
        loss_per_instance = (weights * (p_dist - q_dist).pow(2)).sum(dim=-1)

        return loss_per_instance.mean()

    def clip_style_contrastive_loss(self, image_gates, text_gates):
        """
        使用【批量级】余弦相似度损失计算跨模态路由一致性。
        支持两种输入形式：
          1) Tensor → 形状 [B, num_experts] 或 [num_experts]
          2) dict  → {layer_idx: {'probs' 或 'logits': Tensor}}
        """
        # ---- 情况 1: dict 输入（来自早期版本的调用） ----
        if isinstance(image_gates, dict):
            if len(image_gates) == 0 or len(text_gates) == 0:
                return torch.tensor(0.0, device=self.logit_scale.device)
            # 尝试提取概率或 logits 并归一化
            def _extract_probs(g):
                if 'probs' in g:
                    return g['probs']
                elif 'logits' in g:
                    return F.softmax(g['logits'], dim=-1)
                else:
                    return None
            img_list = [_extract_probs(v) for v in image_gates.values() if _extract_probs(v) is not None]
            txt_list = [_extract_probs(v) for v in text_gates.values() if _extract_probs(v) is not None]
            if len(img_list) == 0 or len(txt_list) == 0:
                return torch.tensor(0.0, device=self.logit_scale.device)
            image_probs = torch.stack(img_list, dim=0)
            text_probs  = torch.stack(txt_list, dim=0)
        else:
            # ---- 情况 2: Tensor 输入 ----
            # 处理 [B, E] 或 [E] 形状
            if image_gates.numel() == 0 or text_gates.numel() == 0:
                return torch.tensor(0.0, device=image_gates.device, dtype=image_gates.dtype)
            image_probs = F.softmax(image_gates, dim=-1)
            text_probs  = F.softmax(text_gates,  dim=-1)

        # 计算批次级别平均概率
        img_mean_probs = image_probs.mean(dim=0)
        txt_mean_probs = text_probs.mean(dim=0)

        # 余弦相似度 & 损失
        similarity = F.cosine_similarity(img_mean_probs, txt_mean_probs, dim=0)
        loss = 1.0 - similarity
        return loss

    def js_div_loss(self, image_gates, text_gates):
        """
        使用JS散度计算跨模态路由一致性损失

        参数:
            image_gates: 图像门控权重 [img_batch_size, num_experts]
            text_gates: 文本门控权重 [txt_batch_size, num_experts]

        返回:
            js_loss: JS散度损失
        """
        # 归一化门控权重
        image_probs = F.softmax(image_gates, dim=-1)
        text_probs = F.softmax(text_gates, dim=-1)

        img_mean_probs = image_probs.mean(dim=0)
        txt_mean_probs = text_probs.mean(dim=0)

        mean_probs = (img_mean_probs + txt_mean_probs) / 2
        
        # 为避免log(0)导致NaN，添加一个小的epsilon
        kl_img = F.kl_div(mean_probs.log(), img_mean_probs, reduction='batchmean')
        kl_txt = F.kl_div(mean_probs.log(), txt_mean_probs, reduction='batchmean')

        return (kl_img + kl_txt) / 2

    def cross_modal_routing_consistency_loss(self, image_gates, text_gates):
        """
        计算图像和文本之间路由决策的一致性损失，根据配置选择不同的损失计算方法

        参数:
            image_gates: 图像门控权重 [img_batch_size, num_experts]
            text_gates: 文本门控权重 [txt_batch_size, num_experts]

        返回:
            consistency_loss: 跨模态路由一致性损失
        """
        # 根据配置选择损失计算方法
        loss_type = self.config.get('cross_modal_loss_type', 'js_div')

        if loss_type == 'weighted_l2':
            image_probs = F.softmax(image_gates, dim=-1)
            text_probs = F.softmax(text_gates, dim=-1)
            # --- 修正：先计算批次平均分布 ---
            img_mean_probs = image_probs.mean(dim=0)
            txt_mean_probs = text_probs.mean(dim=0)
            return self.weighted_distance_loss(img_mean_probs, txt_mean_probs)
        elif loss_type == 'clip_style':
            return self.clip_style_contrastive_loss(image_gates, text_gates)
        else:  # 默认使用JS散度
            return self.js_div_loss(image_gates, text_gates)

    def compute_cross_modal_routing_loss(self):
        """
        计算跨模态路由一致性损失

        返回:
            cross_modal_loss: 跨模态路由一致性损失
        """
        global global_gate_collector
        image_gates = global_gate_collector['image']
        text_gates = global_gate_collector['text']

        # 确保两个模态都有门控信息
        if not image_gates or not text_gates:
            return None

        # 找到两个模态共有的层
        common_layers = set(image_gates.keys()).intersection(set(text_gates.keys()))
        if not common_layers:
            return None

        # 计算每一层的跨模态损失并求平均
        losses = []
        for layer_idx in common_layers:
            # img_gates = image_gates[layer_idx]['gates']
            # txt_gates = text_gates[layer_idx]['gates']
            img_gates = image_gates[layer_idx]['logits']
            txt_gates = text_gates[layer_idx]['logits']
            # 直接计算这一层的跨模态损失，不需要检查形状是否匹配
            layer_loss = self.cross_modal_routing_consistency_loss(
                img_gates, txt_gates
            )
            losses.append(layer_loss)

        if not losses:
            return None

        # 返回所有层的平均损失
        return torch.stack(losses).mean()

    def compute_expert_stability_loss(self, current_task_id):
        """
        计算专家迁移稳定性损失

        参数:
            current_task_id: 当前任务ID

        返回:
            stability_loss: 专家迁移稳定性损失
        """
        global global_gate_collector

        prev_task_id = current_task_id - 1

        # 确保有先前任务的门控信息
        if prev_task_id not in self.historical_gates:
            return torch.tensor(0.0, device=self.logit_scale.device)

        prev_gates = self.historical_gates[prev_task_id]
        # current_gates = {
        #     'image': {k: v['gates'] for k, v in global_gate_collector['image'].items()},
        #     'text': {k: v['gates'] for k, v in global_gate_collector['text'].items()}
        # }
        current_gates = {
            'image': {k: v['logits'] for k, v in global_gate_collector['image'].items()},
            'text': {k: v['logits'] for k, v in global_gate_collector['text'].items()}
        }

        # 计算每个模态和层的稳定性损失
        losses = []
        debug_info = []

        # 图像模态
        for layer_idx, curr_gate in current_gates['image'].items():
            if layer_idx in prev_gates['image']:
                prev_gate = prev_gates['image'][layer_idx]
                # 处理不同数量专家的情况
                curr_experts = curr_gate.size(-1)
                prev_experts = prev_gate.size(-1)

                if curr_experts == prev_experts:
                    loss = self.expert_transfer_stability_loss(curr_gate, prev_gate)
                    losses.append(loss)
                else:
                    min_experts = min(curr_experts, prev_experts)
                    loss = self.expert_transfer_stability_loss(
                        curr_gate[..., :min_experts],
                        prev_gate[..., :min_experts]
                    )
                    losses.append(loss)
                debug_info.append(("image", layer_idx, float(loss.detach().cpu())))
            else:
                debug_info.append(("image", layer_idx, "no_prev_layer"))

        # 文本模态
        for layer_idx, curr_gate in current_gates['text'].items():
            if layer_idx in prev_gates['text']:
                prev_gate = prev_gates['text'][layer_idx]
                curr_experts = curr_gate.size(-1)
                prev_experts = prev_gate.size(-1)

                if curr_experts == prev_experts:
                    loss = self.expert_transfer_stability_loss(curr_gate, prev_gate)
                    losses.append(loss)
                else:
                    min_experts = min(curr_experts, prev_experts)
                    loss = self.expert_transfer_stability_loss(
                        curr_gate[..., :min_experts],
                        prev_gate[..., :min_experts]
                    )
                    losses.append(loss)
                debug_info.append(("text", layer_idx, float(loss.detach().cpu())))
            else:
                debug_info.append(("text", layer_idx, "no_prev_layer"))

        # 打印调试信息
        if len(losses) == 0:
            print(f"[Stability Debug] task {current_task_id}: no matched layers, details: {debug_info}")
        else:
            print(f"[Stability Debug] task {current_task_id}: computed {len(losses)} layer losses, details sample: {debug_info[:4]}")

        total_stability_loss = torch.stack(losses).mean() if losses else None
        return total_stability_loss

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, is_train=False):
        # 确保接收到的是tensor而不是tuple
        if isinstance(image, tuple):
            image = image[0]

        # 初始化辅助损失
        total_aux_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)

        # 获取视觉模型的输出
        visual_output = self.visual(image)

        # 处理可能包含辅助损失的输出
        features, vis_aux_loss = visual_output if isinstance(visual_output, tuple) else (visual_output, torch.tensor(0.0, device=image.device, dtype=self.dtype))
        if vis_aux_loss is None:
            vis_aux_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)
        total_aux_loss += vis_aux_loss

        return features, total_aux_loss

    def encode_text(self, text, is_train=False):
        # 确保接收到的是tensor而不是tuple
        if isinstance(text, tuple):
            text = text[0]
        
        # 初始化辅助损失
        total_aux_loss = torch.tensor(0.0, device=text.device, dtype=self.dtype)

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # 获取transformer的输出
        transformer_output = self.transformer(x)

        # 处理可能包含辅助损失的输出
        x, trans_aux_loss = transformer_output if isinstance(transformer_output, tuple) else (transformer_output, torch.tensor(0.0, device=x.device, dtype=self.dtype))
        if trans_aux_loss is None:
            trans_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_aux_loss += trans_aux_loss

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, total_aux_loss

    def forward(self, image=None, text=None, task_id=None, is_train=False):
        # 设置当前任务ID
        global global_taskid
        global global_gate_collector

        # 确保 task_id 是有效值
        if task_id is not None:
            global_taskid = task_id
            global_gate_collector['current_task_id'] = task_id

        # 清空当前收集器
        global_gate_collector['image'] = {}
        global_gate_collector['text'] = {}

        # 确保接收到的参数是tensor而不是tuple
        if isinstance(image, tuple):
            image = image[0]
        if isinstance(text, tuple):
            text = text[0]

        # 初始化总辅助损失
        total_aux_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)

        # 处理只有图像输入的情况
        if image is not None and text is None:
            image_features, img_aux_loss = self.encode_image(image, is_train)
            return image_features, img_aux_loss if img_aux_loss is not None else torch.tensor(0.0, device=image.device, dtype=self.dtype)

        # 处理只有文本输入的情况
        if image is None and text is not None:
            text_features, txt_aux_loss = self.encode_text(text, is_train)
            return text_features, txt_aux_loss if txt_aux_loss is not None else torch.tensor(0.0, device=text.device, dtype=self.dtype)

        # 处理同时有图像和文本输入的情况
        image_features, img_aux_loss = self.encode_image(image, is_train)
        text_features, txt_aux_loss = self.encode_text(text, is_train)

        # 累积辅助损失
        total_aux_loss = (img_aux_loss if img_aux_loss is not None else torch.tensor(0.0, device=image.device, dtype=self.dtype)) + \
                        (txt_aux_loss if txt_aux_loss is not None else torch.tensor(0.0, device=text.device, dtype=self.dtype))

        # 初始化辅助损失字典
        aux_losses = {'moe': total_aux_loss}

        # 如果是训练模式，计算额外的损失
        if is_train:
            print("11111")
            # 1. 计算跨模态路由一致性损失
            cross_modal_loss = self.compute_cross_modal_routing_loss()
            if cross_modal_loss is not None:
                weighted_cross_modal = cross_modal_loss * self.config['cross_modal_weight']
                aux_losses['cross_modal'] = weighted_cross_modal
                # --- 添加调试输出 ---
                print(f"[Debug Loss] Cross-Modal Routing: {cross_modal_loss.item():.4f} (Weighted: {weighted_cross_modal.item():.4f})")

            # 2. 如果有先前任务，计算专家迁移稳定性损失
            if task_id is not None and task_id > 0 and task_id - 1 in self.historical_gates:
                stability_loss = self.compute_expert_stability_loss(task_id)
                if stability_loss is not None:
                    weighted_stability = stability_loss * self.config['stability_weight']
                    aux_losses['stability'] = weighted_stability
                    # --- 添加调试输出 ---
                    print(f"[Debug Loss] Expert Stability: {stability_loss.item():.4f} (Weighted: {weighted_stability.item():.4f})")

            # 保存当前任务的门控信息，用于未来任务
            if task_id is not None:
                self.historical_gates[task_id] = {
                    'image': {k: v['logits'].clone().detach() for k, v in global_gate_collector['image'].items()},
                    'text': {k: v['logits'].clone().detach() for k, v in global_gate_collector['text'].items()}
                }

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # 计算总辅助损失
        total_aux_loss = sum(aux_losses.values())

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text, total_aux_loss  # , aux_losses


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class PrologueWrapper(nn.Module):
    def __init__(self, module, prologue):
        super().__init__()
        self.module = module
        self.prologue = prologue

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output
        if self.prologue is not None:
            x = self.prologue(x)
        if isinstance(output, tuple):
            output = (x,) + output[1:]
        else:
            output = x
        return output


def add_prologue(module, prologue):
    return PrologueWrapper(module, prologue)


class EpilogueWrapper(nn.Module):
    def __init__(self, module, epilogue):
        super().__init__()
        self.module = module
        self.epilogue = epilogue

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output
        if self.epilogue is not None:
            x = self.epilogue(x)
        if isinstance(output, tuple):
            output = (x,) + output[1:]
        else:
            output = x
        return output


def add_epilogue(module, epilogue):
    return EpilogueWrapper(module, epilogue)


def compress_clip_model(model, drop_visual_layers, drop_transformer_layers):
    """
    压缩给定的 CLIP 模型，对模型的 visual 和 transformer 部分分别进行丢层操作。
    参数:
    - model: 预训练的 CLIP 模型
    - drop_visual_layers: 要从 visual 模块中丢弃的层索引列表
    - drop_transformer_layers: 要从 transformer 模块中丢弃的层索引列表
    返回:
    - 压缩后的 CLIP 模型
    """
    # 处理 visual 部分
    # 创建教师模型的深拷贝，保持完整
    # teacher_model = copy.deepcopy(model)
    # teacher_model.eval()  # 设置为评估模式

    model_dtype = next(model.parameters()).dtype  # 获取模型的数据类型
    if isinstance(model.visual, VisualTransformer):
        visual_layers = list(model.visual.transformer.resblocks)
        print("len of visual_layers", len(visual_layers))

        # Adapter
        visual_adapters = visual_layers[:2] + visual_layers[-2:]
        # 教师层，丢弃Adapter
        teacher_visual_layers = visual_layers[2:10]

        # drop_visual_layers = [1, 3, 5, 7]
        student_visual_layers = [layer for i, layer in enumerate(visual_layers[2:10]) if i not in drop_visual_layers]
        print("len of student_visual_layers", len(student_visual_layers))

        # 转换数据类型
        # student_visual_layers = [layer.to(model_dtype) for layer in student_visual_layers]

        # 处理 transformer 部分
        transformer_layers = list(model.transformer.resblocks)
        # Adapter
        transformer_adapters = transformer_layers[:2] + transformer_layers[-2:]
        # Teacher:
        teacher_transformer_layers = transformer_layers[2:10]
        student_transformer_layers = [layer for i, layer in enumerate(transformer_layers[2:10]) if
                                      i not in drop_transformer_layers]
        print("len of student_transformer_layers", len(student_transformer_layers))

        # student_transformer_layers = [layer.to(model_dtype) for layer in student_transformer_layers]

        student_visual_emulator_and_adapter = visual_layers[:2] + student_visual_layers + visual_layers[-2:]
        student_transformer_emulator_and_adapter = transformer_layers[
                                                   :2] + student_transformer_layers + transformer_layers[-2:]

        model.transformer.resblocks = nn.Sequential(*student_transformer_emulator_and_adapter)
        # 转换数据类型
        model.transformer.resblocks = model.transformer.resblocks.to(model_dtype)
        # 重新构建 visual 的 transformer
        model.visual.transformer.resblocks = nn.Sequential(*student_visual_emulator_and_adapter)
        # 转换数据类型
        model.visual.transformer.resblocks = model.visual.transformer.resblocks.to(model_dtype)

        # print("offsite-tuning-CLIP model compressed",model)

        return model  # , teacher_model


import torch
from typing import Dict

import torch
from typing import Dict

def init_moe_experts_from_mlp(model, pretrained_mlp_weights: Dict[str, torch.Tensor], experts_num: int):
    """
    将 state_dict 中的 text 和 image transformer 的预训练 MLP 权重
    切分并注入到 model.transformer.resblocks 和 model.visual.transformer.resblocks
    里的每个 ResidualAttentionBlock.adaptmlp_list 中。

    pretrained_mlp_weights 要包含:
      - "...mlp.c_fc.weight"  shape: [4*d, d]
      - "...mlp.c_fc.bias"    shape: [4*d]
      - "...mlp.c_proj.weight" shape: [d, 4*d]
      - "...mlp.c_proj.bias"   shape: [d]
    experts_num 为 MoE 专家数量
    """
    # 1. 收集所有 mlp 权重，按 modality 和 block idx 分类
    weights = {'text': {}, 'image': {}}
    for full_key, tensor in pretrained_mlp_weights.items():
        if "resblocks" not in full_key or "mlp.c_" not in full_key:
            continue
        # 判断是 text 还是 image
        if full_key.startswith("transformer.resblocks"):
            modality = 'text'
            prefix = "transformer.resblocks"
        elif full_key.startswith("visual.transformer.resblocks"):
            modality = 'image'
            prefix = "visual.transformer.resblocks"
        else:
            continue
        # tail: ["<idx>", "mlp", "c_fc"/"c_proj", "weight"/"bias"]
        tail = full_key[len(prefix) + 1:].split('.')
        try:
            blk_idx = int(tail[0])
            param   = tail[2]  # c_fc or c_proj
            wb      = tail[3]  # weight or bias
        except:
            continue
        slot = weights[modality].setdefault(blk_idx, {})
        slot[f"{param}_{wb}"] = tensor

    # 2. 注入 helper
    def _inject(modality, blocks):
        wdict = weights[modality]
        for blk_idx, block in enumerate(blocks):
            if blk_idx not in wdict:
                print(f"[WARN] no pretrained mlp for {modality} block {blk_idx}, skip")
                continue
            bw = wdict[blk_idx]
            # 检查完整权重
            req = {"c_fc_weight", "c_fc_bias", "c_proj_weight", "c_proj_bias"}
            if not req.issubset(bw.keys()):
                print(f"[WARN] {modality} blk {blk_idx} missing {req - bw.keys()}, skip")
                continue
            # chunk 分配
            W_fc   = bw["c_fc_weight"]    # [4d, d]
            b_fc   = bw["c_fc_bias"]      # [4d]
            W_proj = bw["c_proj_weight"]  # [d, 4d]
            b_proj = bw["c_proj_bias"]    # [d]
            # 切分
            W_fc_chunks   = torch.chunk(W_fc,   experts_num, dim=0)
            b_fc_chunks   = torch.chunk(b_fc,   experts_num, dim=0)
            W_proj_chunks = torch.chunk(W_proj, experts_num, dim=1)
            # 复制
            for exp_idx, expert in enumerate(block.adaptmlp_list):
                fc: torch.nn.Linear   = expert[0]
                proj: torch.nn.Linear = expert[3]
                # shape check
                assert tuple(fc.weight.shape)   == tuple(W_fc_chunks[exp_idx].shape), \
                    f"{modality} blk{blk_idx} exp{exp_idx} c_fc mismatch"
                assert tuple(proj.weight.shape) == tuple(W_proj_chunks[exp_idx].shape), \
                    f"{modality} blk{blk_idx} exp{exp_idx} c_proj mismatch"
                # copy 权重
                fc.weight.data.copy_(W_fc_chunks[exp_idx])
                fc.bias.data.copy_(b_fc_chunks[exp_idx])
                proj.weight.data.copy_(W_proj_chunks[exp_idx])
                proj.bias.data.copy_(b_proj)
            print(f"✅ {modality} block {blk_idx} initialized {len(block.adaptmlp_list)} experts by chunking")

    # 3. 注入 text
    _inject('text',  model.transformer.resblocks)
    # 4. 注入 image
    _inject('image', model.visual.transformer.resblocks)



def build_model(state_dict: dict, compress=True, cfg=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        cfg=cfg
    )
    pretrained_mlp_weights = {
        k: v.clone()
        for k, v in state_dict.items()
        if ("transformer.resblocks" in k or "visual.transformer.resblocks" in k)
           and ("mlp.c_fc" in k or "mlp.c_proj" in k)
    }
    # 初始化MoE专家权重
    #init_moe_experts_from_mlp(model, pretrained_mlp_weights,experts_num=cfg.num_experts)


    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # model.load_state_dict(state_dict, strict=False)
    for p in model.parameters():
        p.data = p.data.float()

    if compress:
        student_model = compress_clip_model(model, [1, 3, 5, 7], [1, 3, 5, 7])
        print("MoE_Adapters model compressed")

        state_dict = torch.load("/home/chenjunwei/Work/CoOp/distill/cc3m/distilled_vit_b16_clip_cc3m_epoch_3.pth")

        student_model.load_state_dict(state_dict, strict=False)
        student_model.float()
        return student_model.eval()
    else:
        model.load_state_dict(state_dict, strict=False)
        return model.eval()
