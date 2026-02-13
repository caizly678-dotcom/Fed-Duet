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
        self.analyst = None

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

    def forward(self, x: torch.Tensor):
        if isinstance(x, tuple):
            x = x[0]

        ln_1_output = self.ln_1(x)
        attn_output = self.attention(ln_1_output)
        x = x + attn_output

        # gating for current inputs
        x_re = x.permute(1, 0, 2)[:, 0, :]
        gates, load, logits = self.noisy_top_k_gating(
            x_re,
            self.training,
            self.router_list[0],
            self.w_noise_list[0],
        )

        batch_size = x.shape[1]
        seq_len = x.shape[0]
        hidden_dim = x.shape[2]

        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))

        expert_outputs = []
        for i in range(self.experts_num):
            if expert_inputs[i].numel() == 0:
                continue
            out = self.adaptmlp_list[i](
                expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(x),
                add_residual=False,
            )
            expert_outputs.append(out.view(out.shape[0], -1))

        y_routed = dispatcher.combine(expert_outputs)
        y_routed = y_routed.view(batch_size, seq_len, hidden_dim)

        if self.shared_expert_num > 0:
            shared_outputs = []
            for i in range(self.shared_expert_num):
                x_reshaped = x.permute(1, 0, 2).reshape(-1, x.shape[-1])
                shared_output = self.shared_expert[i](x_reshaped, add_residual=False)
                shared_output = shared_output.reshape(x.shape[1], x.shape[0], x.shape[2]).permute(1, 0, 2)
                shared_outputs.append(shared_output)
            y_shared = torch.stack(shared_outputs).mean(0) if self.shared_expert_num > 1 else shared_outputs[0]
            y_shared = y_shared.permute(1, 0, 2)
            y_combined = y_routed + y_shared
        else:
            y_combined = y_routed

        if self.moe_scalar is not None:
            y_combined = y_combined * self.moe_scalar

        ln_2_output = self.ln_2(x)
        if self.use_mlp:
            mlp_output = self.mlp(ln_2_output)
            x = x + mlp_output + y_combined.permute(1, 0, 2)
        else:
            x = x + y_combined.permute(1, 0, 2)

        return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)

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

        self.ctx_vector = None

    # ---------- Prompt ctx 向量注入接口 ----------
    def set_ctx_vector(self, ctx_vec: torch.Tensor):
        self.ctx_vector = ctx_vec.detach() if ctx_vec is not None else None

    def forward(self, x: torch.Tensor):
        # 确保接收到的是tensor而不是tuple
        if isinstance(x, tuple):
            x = x[0]

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

        self.context_length = context_length
        self.task_num = 100 / cfg.initial_increment if cfg is not None and hasattr(cfg, 'initial_increment') else 1

        # 设置损失权重和类型，如果cfg中未定义，则使用新的默认值

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

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


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

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # 获取transformer的输出
        transformer_output = self.transformer(x)

        x = transformer_output if not isinstance(transformer_output, tuple) else transformer_output[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, torch.tensor(0.0, device=text.device, dtype=self.dtype)

    def forward(self, image=None, text=None, task_id=None, is_train=False):
        # image only: 返回特征和辅助损失
        if image is not None and text is None:
            return self.encode_image(image, is_train)

        # text only: 返回特征和辅助损失
        if image is None and text is not None:
            return self.encode_text(text, is_train)

        # image + text: 计算相似度日志与总辅助损失
        image_features, img_aux = self.encode_image(image, is_train)
        text_features, txt_aux = self.encode_text(text, is_train)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        total_aux_loss = img_aux + txt_aux
        return logits_per_image, total_aux_loss


def build_model(state_dict: dict, compress=True, cfg=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
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
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        cfg=cfg,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.float()
    return model.eval()
