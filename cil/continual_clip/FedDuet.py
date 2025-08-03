import os.path as osp
import os
import time
import random

import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import copy
from collections import defaultdict

# === Prompt Pool & Gate imports ===
from .prompt_pool import PromptPool, GateNetwork, SparseGateLoss

from continual_clip.sampling import sample_iid, sample_noniid
import clip.clip as clip
from clip import model as clip_module
from continual_clip.utils import get_class_ids_per_task, get_class_names  # 新增
from clip.tokenizer import SimpleTokenizer as _Tokenizer

from . import utils

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    """文本编码器，用于编码prompt"""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        transformer_output = self.transformer(x)  # 这里可能返回元组
        loss = None
        # 提取特征张量（如果是元组，取第一个元素）
        if isinstance(transformer_output, tuple):
            x = transformer_output[0]
            loss = transformer_output[1]
            # print(f"Transformer输出损失: {loss}")
        else:
            x = transformer_output
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        if loss is not None:
            # 如果有损失，返回损失和特征
            return x, loss
        else:
            return x


class PromptLearner(nn.Module):
    """Prompt 学习器（客户端私有的本地 Prompt 组）。不再使用 LoRA，直接训练 8×512 向量。"""

    def __init__(self, cfg, classnames, clip_model, prev_ctx=None):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- 基础超参数 ----------
        n_cls = len(classnames)
        n_ctx = getattr(cfg, "N_CTX", 16)  # 上下文长度
        ctx_init = getattr(cfg, "CTX_INIT", "")  # 初始化文本
        csc = getattr(cfg, "CSC", False)  # 是否使用 CSC（Class-Specific Contexts）默认False
        class_token_position = getattr(cfg, "CLASS_TOKEN_POSITION", "end")  # 类别token位置，默认在末尾

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # ---------- ctx 初始化（直接作为可训练参数） ----------
        if prev_ctx is not None:
            # 如果提供了上一个任务的上下文向量，则使用它
            # print("使用上一个任务的上下文向量")
            ctx_vectors = prev_ctx.to(torch.float32)  # 确保是 FP32
            prompt_prefix = " ".join(["X"] * n_ctx)  # 设置默认的 prompt_prefix
        elif ctx_init:
            # 使用给定词初始化上下文向量
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)  # 将初始化文本转换为token
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].to(device)  # 转换为上下文向量
            ctx_vectors = ctx_vectors.to(torch.float32) # 确保是 FP32
            prompt_prefix = ctx_init
        else:
            # 随机初始化
            if csc:
                # print("初始化类别特定的上下文")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=torch.float32, device=device)
            else:
                # print("初始化通用上下文")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # ---------- 其余 Prompt 编码准备 ----------
        self.ctx = nn.Parameter(ctx_vectors)  # 可优化参数 (FP32)

        # 检查self.ctx是否可以训练
        # print(f"上下文参数可训练: {self.ctx.requires_grad}")
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # 将类别名称转换为token
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)  # 获取token的嵌入向量

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    # 兼容旧接口
    @property
    def ctx_base(self):
        return self.ctx

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # 将 ctx 转换为与模型其余部分匹配的 dtype (FP16)
        ctx = ctx.to(prefix.dtype)

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                                   dim=1)  # 最后生成的prompt是[CLS, ctx前半部分, 类别token, ctx后半部分, EOS]
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


# --------------------------------------------------
# CustomCLIP
# --------------------------------------------------

class CustomCLIP(nn.Module):
    """自定义CLIP模型，集成PromptLearner和TextEncoder"""

    def __init__(self, cfg, classnames, clip_model, prev_ctx=None, prev_fusion_state=None, historical_gates=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, prev_ctx)
        self.cfg = cfg
        self.n_class = len(classnames)
        self.classnames = classnames

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.clip_model = clip_model
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.ln_final = clip_model.ln_final
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.text_projection = clip_model.text_projection

        # ---- 新增：专家融合组件 (pFedMoAP logic) ----
        self.expert_prompts: torch.Tensor = None  # (K_expert, 512)

        # === 本地特征融合 Gating 网络（客户端私有） ===
        # 为避免与服务器端 GateNetwork 混淆，此处命名为 fusion_gating
        feature_dim = clip_model.ln_final.weight.shape[0]
        gating_embed_dim = getattr(cfg, "gating_embed_dim", 128)
        num_heads = getattr(cfg, "gating_heads", 8)
        scaling = getattr(cfg, "gating_scaling", 10.0)

        # Ensure feature_dim is divisible by gating_embed_dim for integer reduce_times
        if feature_dim % gating_embed_dim != 0:
            raise ValueError(f"Feature dim ({feature_dim}) must be divisible by gating_embed_dim ({gating_embed_dim})")
        self.reduce_times = feature_dim // gating_embed_dim

        self.fusion_gating = MultiheadAttention(gating_embed_dim, num_heads=num_heads, scaling=scaling, dtype=torch.float32)

        # 如果上一个任务提供了 fusion gating 参数，则加载
        if prev_fusion_state is not None:
            try:
                self.fusion_gating.load_state_dict(prev_fusion_state, strict=False)
                print("[CustomCLIP] 已加载上一任务的 fusion_gating 状态")
            except Exception as e:
                print(f"[CustomCLIP] 加载 fusion_gating 失败: {e}")

        # 融合系数 λ，控制 local vs global logits 权重
        self.lmbda = getattr(cfg, "ctx_lmbda", 0.5)

        # --------- 全局/非本地 prompt 支持 ---------
        self.nonlocal_ctx = None  # 服务器下发的上下文列表 (List[Tensor] or Tensor)
        self.nonlocal_text_features = []  # 存储处理后的全局 prompt 文本特征

        # 显式设置历史门控信息
        if historical_gates is not None and hasattr(self.clip_model, 'set_historical_gates'):
            self.clip_model.set_historical_gates(historical_gates)

        # MoE 相关参数
        self.experts_num = cfg.num_experts

        # ---- Helper: recursively set historical gates to the base CLIP ----
        if historical_gates is not None:
            base = self.clip_model
            depth_guard = 0
            while hasattr(base, 'clip_model') and depth_guard < 5:
                base = base.clip_model
                depth_guard += 1
            if hasattr(base, 'set_historical_gates'):
                base.set_historical_gates(historical_gates)

    @property
    def historical_gates(self):
        base = self.clip_model
        depth_guard = 0
        while hasattr(base, 'clip_model') and depth_guard < 5:
            base = base.clip_model
            depth_guard += 1
        return getattr(base, 'historical_gates', {})

    # === 新增：将损失计算委托给内部的 clip_model ===
    def compute_cross_modal_routing_loss(self):
        base = self.clip_model
        depth_guard = 0
        while hasattr(base, 'clip_model') and depth_guard < 5:
            base = base.clip_model
            depth_guard += 1
        if hasattr(base, 'compute_cross_modal_routing_loss'):
            return base.compute_cross_modal_routing_loss()
        return None

    def compute_expert_stability_loss(self, task_id):
        base = self.clip_model
        depth_guard = 0
        while hasattr(base, 'clip_model') and depth_guard < 5:
            base = base.clip_model
            depth_guard += 1
        if hasattr(base, 'compute_expert_stability_loss'):
            return base.compute_expert_stability_loss(task_id)
        return None

    def pool(self, t: torch.Tensor):
        """Feature pooling by slicing."""
        if len(t.shape) == 4:
            return t[:, :, :, ::self.reduce_times]
        if len(t.shape) == 3:
            return t[:, :, ::self.reduce_times]
        if len(t.shape) == 2:
            return t[:, ::self.reduce_times]
        return None

    # ======== Non-local prompt 相关 ========
    def load_ctx(self, ctx: torch.Tensor):
        """将给定 ctx (n_ctx, d) 加载到 prompt_learner 以便编码。"""
        state_dict = self.prompt_learner.state_dict()
        state_dict['ctx'] = ctx
        self.prompt_learner.load_state_dict(state_dict, strict=False)

    def _compute_nonlocal_text_features(self):
        if self.nonlocal_ctx is None:
            self.nonlocal_text_features = []
            return

        temp_local_state = copy.deepcopy(self.prompt_learner.state_dict())
        self.nonlocal_text_features = []

        # 统一成列表
        if not isinstance(self.nonlocal_ctx, list):
            self.nonlocal_ctx = [self.nonlocal_ctx]

        for ctx in self.nonlocal_ctx:
            # load nonlocal ctx
            self.load_ctx(ctx)

            # compute nonlocal text features
            with torch.no_grad():
                text_output = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
                text_features = text_output if isinstance(text_output, torch.Tensor) else text_output[0]

                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = self.pool(text_features)
                self.nonlocal_text_features.append(text_features.detach())

        # 恢复本地 ctx
        self.prompt_learner.load_state_dict(temp_local_state)

    def update_prompt_learner(self, prev_ctx=None, new_classnames=None):
        """更新prompt learner的ctx参数"""
        # 当类别列表发生变化或显式提供新的 ctx 时，都需要重新初始化 PromptLearner。
        # 如果没有显式提供 prev_ctx，则默认沿用当前 prompt_learner 的 ctx 权重，
        # 这样可以避免打乱已学习的 prompt 表示。

        need_reinit = (new_classnames is not None) or (prev_ctx is not None)

        if not need_reinit:
            return  # 无需任何更新

        # 如果未显式传入 prev_ctx，则使用现有的 ctx 作为初始化
        if prev_ctx is None:
            prev_ctx = self.prompt_learner.ctx.detach().clone()

        # 如果未显式传入 new_classnames，则保持现有类别
        if new_classnames is None:
            new_classnames = self.classnames

        # print(f"[PromptLearner 更新] 旧类别数: {len(self.classnames)}, 新类别数: {len(new_classnames)}")

        # 重新构建 PromptLearner
        self.prompt_learner = PromptLearner(self.cfg, new_classnames, self.clip_model, prev_ctx)
        self.classnames = new_classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # print(f"[PromptLearner 更新] tokenized_prompts 形状: {self.tokenized_prompts.shape}")

    def forward(self, image, text=None, task_id=None, is_train=True, prev_ctx=None):
        """前向传播。
        Args:
            image (torch.Tensor): 输入图像张量。
            text (torch.Tensor, optional): 文本tokens。如果为None，则使用prompt_learner生成的prompts。
            task_id (int, optional): 当前任务ID，用于模型内部记录和损失计算。
            is_train (bool): 是否为训练模式。
        """
        # --- 任务ID设置：为模型内部的MoE层等组件设置当前任务ID ---
        if task_id is not None:
            clip_module.global_taskid = task_id
            clip_module.global_gate_collector['current_task_id'] = task_id
            clip_module.global_gate_collector['image'] = {}
            clip_module.global_gate_collector['text'] = {}

        # -------- 计算有效 ctx 用于 Cross-Attention 分析师 --------
        with torch.no_grad():
            ctx_eff_vec = self.prompt_learner.ctx.detach().clone()  # [n_ctx, d_ctx]

        # 将 ctx 向量注入 image_encoder 供 ResidualAttentionBlock 使用--Cross Attention
        if hasattr(self.image_encoder, "set_ctx_vector"):
            self.image_encoder.set_ctx_vector(ctx_eff_vec)

        # -------- 图像编码 (B, D) --------
        image_output = self.image_encoder(image.type(self.dtype))
        if isinstance(image_output, tuple):
            image_features, image_loss = image_output
        else:
            image_features = image_output
            image_loss = None

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, D)

        # -------- 本地 prompt 文本特征 (分块处理以节省显存) --------
        local_prompts = self.prompt_learner()
        tokenized = self.tokenized_prompts
        
        # 当类别数量很大时 (如DomainNet)，一次性编码所有文本prompt会消耗大量显存。
        # 这里通过分块处理来降低峰值显存占用。
        text_batch_size = getattr(self.cfg, "text_batch_size", 256)
        num_prompts = local_prompts.shape[0]
        
        all_text_features = []
        all_text_losses = []
        
        for i in range(0, num_prompts, text_batch_size):
            prompt_chunk = local_prompts[i : i + text_batch_size]
            token_chunk = tokenized[i : i + text_batch_size]
            
            text_output_chunk = self.text_encoder(prompt_chunk, token_chunk)
            
            if isinstance(text_output_chunk, tuple):
                text_features_chunk, text_loss_chunk = text_output_chunk
                if text_loss_chunk is not None:
                    all_text_losses.append(text_loss_chunk)
            else:
                text_features_chunk = text_output_chunk
            
            all_text_features.append(text_features_chunk)

        text_features = torch.cat(all_text_features, dim=0)
        text_loss = torch.stack(all_text_losses).mean() if all_text_losses else None
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # ---- 本地 logits ----
        local_logits = logit_scale * image_features @ text_features.t()  # (B, n_cls)

        # ---- 若存在全局专家，则使用 Gating 进行特征融合 ----
        if self.training and self.nonlocal_text_features:
            # q: 来自图像的特征, (n_cls, B, d_pooled)
            q = self.pool(image_features).repeat(self.n_class, 1, 1)

            # k, v: 来自本地和全局的文本特征, (n_cls, n_experts, d_pooled)
            # 将本地 text_features 和 nonlocal_text_features 堆叠
            # nonlocal_text_features 是一个 list of (n_cls, d), 需要堆叠成 (n_experts, n_cls, d)
            # 然后 permute 成 (n_cls, n_experts, d_pooled)
            global_expert_feats = torch.stack(self.nonlocal_text_features, dim=0)  # (n_experts, n_cls, d_pooled)
            k = v = torch.cat([
                self.pool(text_features).unsqueeze(1),  # (n_cls, 1, d_pooled)
                global_expert_feats.permute(1, 0, 2)  # (n_cls, n_experts, d_pooled)
            ], dim=1)

            # 通过 gating 融合特征
            # 注意：为保证数值稳定性，输入到 attention 的 Q,K,V 强制转为 float32
            orig_dtype = q.dtype
            new_features, _ = self.fusion_gating(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))  # (n_cls, B, d_pooled)
            new_features = new_features.to(orig_dtype)
            
            new_features = new_features.permute(1, 2, 0)  # (B, d_pooled, n_cls)

            # 计算融合后的 logits
            fused_logits = logit_scale * torch.bmm(self.pool(image_features).unsqueeze(1), new_features).squeeze(1)

            # 加权求和
            logits = self.lmbda * local_logits + (1 - self.lmbda) * fused_logits
        else:
            logits = local_logits

        # 合并所有损失
        total_loss = None
        if image_loss is not None or text_loss is not None:
            total_loss = 0.0
            if image_loss is not None:
                total_loss += image_loss
            if text_loss is not None:
                total_loss += text_loss

        return logits, total_loss

    # ---------- 新增：服务器下发全局专家 Prompt ----------
    @torch.no_grad()
    def set_global_experts(self, expert_ctx_list):
        """服务器下发全局专家 Prompt ctx 列表 (List[Tensor] 或 Tensor)。"""
        self.nonlocal_ctx = expert_ctx_list
        self._compute_nonlocal_text_features()


# --------------------------------------------------
# FedDuetTrainer
# --------------------------------------------------

class FedDuetTrainer:
    """实现FedDuet联邦学习的核心训练器"""

    def __init__(self, cfg, global_model, client_model, train_dataset, eval_dataset, task_id, texts, prev_ctx=None,
                 prev_fusion_state=None, prev_mean_acc_history=None, classes_names=None, prev_client_states=None):
        self.cfg = cfg
        self.global_model = CustomCLIP(cfg, texts, global_model, prev_ctx, prev_fusion_state)
        self.client_model = CustomCLIP(cfg, texts, client_model, prev_ctx, prev_fusion_state)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_id = task_id
        self.texts = texts
        self.classes_names = classes_names  # 全局类别名称列表
        # 使用 cfg.device 指定的 GPU；若为字符串则转换为 torch.device
        if hasattr(cfg, "device"):
            self.device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将 global_model 固定在 CPU，client_model 仍放在训练用 GPU
        self.global_model.to(torch.device("cpu"))
        self.client_model.to(self.device)

        # -------- 显存管理：清理缓存 --------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 联邦学习参数
        self.iid = getattr(cfg, "iid", True)
        self.num_clients = getattr(cfg, "num_clients", 5)
        self.com_rounds = getattr(cfg, "com", 10)
        self.client_epochs = getattr(cfg, "client_epochs", 5)
        self.current_round = 0
        self.metrics = defaultdict(list)

        # ---- 聚合策略 (用于消融实验) ----
        self.aggregation_strategy = getattr(cfg, "aggregation_strategy", "FedDuet")
        print(f"[聚合策略] 当前使用: {self.aggregation_strategy}")

        # 历史 mean_acc 记录
        self.mean_acc_history = prev_mean_acc_history.copy() if prev_mean_acc_history else []

        # ---- 保存每个客户端的最终个性化状态（prompt, MoE experts等） ----
        self.final_client_states = [{} for _ in range(self.num_clients)]

        # 创建数据划分
        if self.iid:
            self.dict_users = sample_iid(train_dataset[task_id:task_id + 1], self.num_clients)
        else:
            self.dict_users = sample_noniid(train_dataset[task_id:task_id + 1], self.num_clients)

        # 创建客户端数据加载器
        self.clients_loaders = []
        self.client_sizes = []
        for uid in range(self.num_clients):
            client_indices = list(self.dict_users[uid])
            client_subset = Subset(train_dataset[task_id:task_id + 1], client_indices)
            client_loader = DataLoader(
                client_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=getattr(cfg, "num_workers", 1),
                drop_last=True
            )
            self.clients_loaders.append(client_loader)
            self.client_sizes.append(len(client_subset))

        self.trainable_params = []
        self.trainable_params_names = []

        # ---------- 消融实验开关 ----------
        self.ablation_no_moe_adapter = getattr(cfg, "ablation_no_moe_adapter", False)
        if self.ablation_no_moe_adapter:
            print("[消融实验] 已启用 NO_MOE_ADAPTER，MoE adapters 将不会被使用或训练。")


        # ---------- 全局Prompt池初始化 ----------
        self.prompt_pool_size = getattr(cfg, "prompt_pool_size", 64)  # 全局 Prompt 池大小
        with torch.no_grad():
            base_ctx = self.global_model.prompt_learner.ctx.detach().clone()  # (n_ctx,d)

        # ---- 根据场景选择用于 KMeans 初始化的类别名称文件 ----
        if getattr(cfg, "scenario", "class") == "domain":
            class_file_path = "cil/dataset_reqs/domainnet_classes.txt"
        else:
            class_file_path = "cil/dataset_reqs/imagenet1000_classes.txt"

        self.prompt_pool = PromptPool(
            K=self.prompt_pool_size,
            base_prompt=base_ctx,
            init_by_kmeans=getattr(cfg, "init_pool_by_kmeans", True),
            clip_model=self.global_model.clip_model,
            class_file=class_file_path
        )

        # 每个客户端一次通信下发的专家数量
        self.num_experts_per_client = getattr(cfg, "num_experts", 8)

        # -------- 门控网络 (Gate Network) - 服务器端的决策者 --------
        self.feature_dim = base_ctx.shape[-1]
        gate_hidden = getattr(cfg, "gate_hidden_dim", 512)
        self.gate = GateNetwork(self.feature_dim, self.prompt_pool_size, gate_hidden).to(self.device)
        self.gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=getattr(cfg, "gate_lr", 1e-3))

        # ----- Gate 训练缓冲区 (feat, prompt_idx, loss) -----
        self._gate_train_buffer = []

        # 用于推理时缓存客户端特征
        self.client_features = [None] * self.num_clients

        moe_keywords = ("adaptmlp_list", "router", "noise", "shared_expert")
        prompt_keywords = ("prompt_learner", "fusion_gating")  # prompt参数关键字

        # === 根据 cfg.unfreeze_moe 控制是否训练 MoE 参数 ===
        unfreeze_moe = getattr(cfg, "unfreeze_moe", True)
        if not unfreeze_moe:
            print("cfg.unfreeze_moe=False -> 冻结所有 MoE 相关参数")
            for model in (self.global_model, self.client_model):
                for n, p in model.named_parameters():
                    if any(k in n for k in moe_keywords):
                        p.requires_grad = False
        else:
            print("cfg.unfreeze_moe=True -> 允许训练 MoE 相关参数")

        # 联邦通信时的上传策略
        self.upload_moe_params = getattr(cfg, "upload_moe_params", True)  # 是否上传 MoE 参数
        self.upload_prompt_params = getattr(cfg, "upload_prompt_params", False)  # 是否上传 Prompt 参数（默认不上传，实现个性化）

        # 如果明确禁止上传 MoE 参数，则冻结相关参数（前提：未显式要求 unfreeze）
        if (not self.upload_moe_params) and (not unfreeze_moe):
            for model in (self.global_model, self.client_model):
                for n, p in model.named_parameters():
                    if any(k in n for k in moe_keywords):
                        p.requires_grad = False

        # 统计在联邦聚合时需要上传的参数（trainable_params_names）
        for name, param in self.global_model.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结参数

            # 判断是否跳过 prompt 参数
            if (not self.upload_prompt_params) and any(pk in name for pk in prompt_keywords):
                print("跳过上传 prompt 参数:", name)
                continue  # 不上传 prompt

            # 判断是否跳过 MoE 参数
            if (not self.upload_moe_params) and any(mk in name for mk in moe_keywords):
                print("跳过上传 MoE 参数:", name)
                continue  # 不上传 MoE

            self.trainable_params_names.append(name)
        # print("可训练参数名字：",self.trainable_params_names)

        # 为后续同步准备：保存 prompt 相关关键字列表，便于在加载全局模型时保留本地 personalized prompt
        self._prompt_keywords = prompt_keywords

        # 训练稳定性参数
        self.gradient_clip_norm = getattr(cfg, "gradient_clip_norm", 1.0)

        # --------- 预创建客户端测试加载器（同FedAvg） ---------
        self.clients_test_loaders = []
        if self.eval_dataset is not None:
            # --- 新增：一次性为所有任务的测试集进行客户端划分 ---
            self.dict_users_test_all_tasks = []
            for t in range(len(self.eval_dataset)): # 假设 eval_dataset 是一个可以获取总任务数的对象
                task_dataset = self.eval_dataset[t:t+1]
                if self.iid:
                    self.dict_users_test_all_tasks.append(sample_iid(task_dataset, self.num_clients))
                else:
                    self.dict_users_test_all_tasks.append(sample_noniid(task_dataset, self.num_clients))
            # ---------------------------------------------------

            for uid in range(self.num_clients):
                client_indices = list(self.dict_users[uid])
                client_subset_test = Subset(self.eval_dataset[task_id:task_id + 1], client_indices)
                test_loader = DataLoader(
                    client_subset_test,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=getattr(cfg, "num_workers", 4),
                    drop_last=False
                )
                self.clients_test_loaders.append(test_loader)
        else:
            self.clients_test_loaders = self.clients_loaders

        # 统计 MoE 参数名，便于聚合
        self._moe_param_names = [n for n, _ in self.global_model.named_parameters() if
                                 any(k in n for k in moe_keywords)]

    def _train_client(self, client_id, train_loader, prompt_idx):
        """训练单个客户端模型"""
        self.client_model.train()

        # ---- Step 1: 根据通信轮次，动态冻结/解冻参数（两阶段训练） ----
        prompt_keywords = ("prompt_learner", "fusion_gating")  # 训练目标：prompt 和 fusion_gating
        moe_keywords = ("adaptmlp_list", "router", "noise", "shared_expert")

        # 默认冻结所有参数
        for p in self.client_model.parameters():
            p.requires_grad = False

        if self.ablation_no_moe_adapter:
            # --- 消融实验：只训练 prompt 和 gating ---
            print(f"[Client {client_id}] Ablation Mode: 训练 PromptLearner 和 Gating")
            for name, p in self.client_model.named_parameters():
                if any(k in name for k in prompt_keywords):
                    p.requires_grad = True
        elif self.current_round < self.com_rounds // 2:
            # --- 阶段一：训练共享 MoE ---
            print(f"[Client {client_id}] Round {self.current_round}: 训练 MoE")
            for name, p in self.client_model.named_parameters():
                if any(k in name for k in moe_keywords):
                    p.requires_grad = True
        else:
            # --- 阶段二：训练私有 prompt 和 gating ---
            print(f"[Client {client_id}] Round {self.current_round}: 训练 PromptLearner 和 Gating")
            for name, p in self.client_model.named_parameters():
                if any(k in name for k in prompt_keywords):
                    p.requires_grad = True

        total_params = sum(p.numel() for p in self.client_model.parameters())
        trainable_params = sum(p.numel() for p in self.client_model.parameters() if p.requires_grad)
        print(f"总参数: {total_params}, 可训练参数数: {trainable_params}, 占比: {100 * trainable_params / total_params:.2f}%")

        # ---- Step 2: compute feature summary f_i over few batches ----
        # ------------- (1) 计算客户端特征摘要 f_i -------------
        def _feature_summary(max_batches: int = 10):
            """返回差分隐私保护后的特征摘要 f_i^dp"""
            feats = []
            cnt = 0
            for imgs, _, _ in train_loader:
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    out = self.client_model.image_encoder(imgs.type(self.client_model.dtype))
                    if isinstance(out, tuple):
                        out = out[0]
                    out = out / out.norm(dim=-1, keepdim=True)
                    feats.append(out.mean(dim=0))
                cnt += 1
                if cnt >= max_batches:
                    break

            feat_vec = torch.stack(feats).mean(dim=0)  # (d,)

            # ----- DP 处理（可选） -----
            if getattr(self.cfg, "enable_dp", False):
                clip_norm = getattr(self.cfg, "dp_clip", 1.0)
                noise_mul = getattr(self.cfg, "dp_noise_multiplier", 0.1)

                norm = feat_vec.norm()
                if norm > clip_norm:
                    feat_vec = feat_vec * (clip_norm / norm)

                if noise_mul > 0:
                    noise = torch.randn_like(feat_vec) * (clip_norm * noise_mul)
                    feat_vec = feat_vec + noise

                print(f"[Client {client_id}] DP: clip={clip_norm}, noise_mul={noise_mul}")

            return feat_vec

        feat_summary = _feature_summary(max_batches=getattr(self.cfg, "summary_batches", 1))  # (d,)

        # 将 (f_i, prompt_idx) 加入 Gate 训练缓冲区（label 为一热）
        # self._gate_train_buffer.append((feat_summary.detach().cpu(), prompt_idx)) # 旧逻辑

        # 记录所选 prompt id
        selected_pid = prompt_idx

        trainable_params = [p for p in self.client_model.parameters() if p.requires_grad]

        # total = sum(p.numel() for p in self.client_model.parameters())
        # 输出可训练参数的名字
        # print(f"可训练参数的name: {[name for name, p in self.client_model.named_parameters() if p.requires_grad]}")
        # print(f"客户端 {client_id} 模型参数总数: {total}")
        # trainable = sum(p.numel() for p in self.client_model.parameters() if p.requires_grad)
        # print(f"Trainable params on Training: {trainable} ({100 * trainable / total:.2f}%)")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.lr,
            weight_decay=getattr(self.cfg, "weight_decay", 0.01)
        )
        scaler = torch.cuda.amp.GradScaler()

        # --- 梯度累积优化 ---
        accumulation_steps = getattr(self.cfg, "gradient_accumulation_steps", 1)
        if accumulation_steps > 1:
            print(f"[显存优化] 启用梯度累积, 步数为: {accumulation_steps}")

        total_iterations = len(train_loader) * self.client_epochs
        scheduler = utils.cosine_lr(optimizer, self.cfg.lr, 30, total_iterations)
        train_iter = iter(train_loader)
        progress_bar = tqdm(range(total_iterations), desc=f"客户端 {client_id} 训练 (任务 {self.task_id})")

        client_metrics = defaultdict(list)

        # 初始化滑动平均准确率和损失
        running_accuracy = 0.0
        running_loss = 0.0
        
        optimizer.zero_grad() # 在训练循环前置零梯度

        for iteration in range(total_iterations):
            try:
                inputs, targets, task_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            # 标签偏移处理
            if getattr(self.cfg, "scenario", "class") == "class" and hasattr(self.cfg, "increment"):
                shift = self.task_id * self.cfg.increment
                targets = targets - shift

            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()  # cross_entropy 需要 int64 类型标签

            # --- 新增：手动清空门控信息收集器 ---
            if hasattr(self.client_model, 'clip_model') and hasattr(self.client_model.clip_model, 'global_gate_collector'):
                self.client_model.clip_model.global_gate_collector['image'] = {}
                self.client_model.clip_model.global_gate_collector['text'] = {}
            # ------------------------------------

            with torch.cuda.amp.autocast():
                # 注意这里模型传入的是输入图片，没有传入额外的参数如text_tokens，确保模型使用自己的tokenized_prompts
                output, moe_loss = self.client_model(inputs, task_id=self.task_id)

                cls_loss = F.cross_entropy(output, targets, label_smoothing=getattr(self.cfg, "ls", 0.0))
                total_loss = cls_loss
                if moe_loss is not None:
                    total_loss += moe_loss

                # --- 新增：计算并添加跨模态和稳定性损失 ---
                # 修正：直接在 client_model (CustomCLIP) 上调用，由它委托给内部的 clip_model
                # 1. 跨模态路由一致性损失
                cross_modal_loss = self.client_model.compute_cross_modal_routing_loss()
                if cross_modal_loss is not None:
                    weighted_cross_modal = cross_modal_loss * self.cfg.cross_modal_weight
                    # print(f"[Debug Cross-Modal Loss] Client {client_id} | Total Cross-Modal Loss: {weighted_cross_modal.item():.4f}")
                    total_loss += weighted_cross_modal

                # 2. 专家迁移稳定性损失
                if self.task_id > 0:
                    stability_loss = self.client_model.compute_expert_stability_loss(self.task_id)
                    if stability_loss is not None:
                        weighted_stability = stability_loss * self.cfg.stability_weight
                        # print(f"[Debug Stability Loss] Client {client_id} | Total Stability Loss: {weighted_stability.item():.4f}")
                        total_loss += weighted_stability
                # ------------------------------------------

                # --- 梯度累积：对损失进行归一化 ---
                if accumulation_steps > 1:
                    total_loss = total_loss / accumulation_steps

                with torch.no_grad():
                    _, predicted = torch.max(output, 1)
                    batch_accuracy = (predicted == targets).float().mean().item()
                    # 更新滑动平均
                    running_accuracy = 0.9 * running_accuracy + 0.1 * batch_accuracy
                    running_loss = 0.9 * running_loss + 0.1 * total_loss.item() * accumulation_steps # 恢复原损失大小用于日志
            
            # 反向传播，累积梯度
            scaler.scale(total_loss).backward()

            # --- 在累积了足够步数后，执行优化器步骤 ---
            if (iteration + 1) % accumulation_steps == 0 or (iteration + 1) == total_iterations:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 为下一轮累积清空梯度

            # 使用scheduler更新学习率
            scheduler(iteration)

            # 记录指标
            client_metrics["loss"].append(total_loss.item() * accumulation_steps) # 记录恢复后的损失
            client_metrics["accuracy"].append(batch_accuracy)

            if moe_loss is not None:
                if isinstance(moe_loss, float):
                    moe_loss = torch.tensor(moe_loss, device=self.device)
                client_metrics["moe_loss"].append(moe_loss.item())
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{running_loss:.4f}",
                'acc': f"{running_accuracy:.4f}",
                'moe_loss': f"{moe_loss.item() if moe_loss is not None else 0:.4f}"
            })
            progress_bar.update(1)

            # --- 显存优化：手动删除不再需要的张量 ---
            del inputs, targets, output, total_loss
            if 'moe_loss' in locals(): del moe_loss
            if 'cross_modal_loss' in locals(): del cross_modal_loss
            if 'stability_loss' in locals(): del stability_loss

        progress_bar.close()
        # 计算平均指标
        avg_metrics = {
            "loss": np.mean(client_metrics["loss"]) if client_metrics["loss"] else 0.0,
            "accuracy": np.mean(client_metrics["accuracy"]) if client_metrics["accuracy"] else 0.0
        }
        # ----- 新增：将(特征, prompt_id列表, loss)加入Gate训练缓冲区 -----
        # 收集数据用于 GateNetwork 训练，使用完整的 expert indices 列表
        self._gate_train_buffer.append((feat_summary.detach().cpu(), prompt_idx, avg_metrics["loss"]))

        print(f"[客户端 {client_id}, 任务 {self.task_id}] 已完成 {total_iterations} 次迭代")
        print(f"最终指标 - 损失: {avg_metrics['loss']:.6f}, 准确率: {avg_metrics['accuracy']:.6f}")

        # 将客户端指标汇总到全局指标
        for key, values in client_metrics.items():
            if values:  # 确保有值
                self.metrics[key].extend(values)

        return avg_metrics, feat_summary.detach()

    def train(self):
        """执行联邦平均训练过程"""
        self.global_model.train()

        for global_round in range(self.com_rounds):
            self.current_round = global_round
            print(f"\n=== 全局通信轮次 [{global_round + 1}/{self.com_rounds}] ===")

            # 客户端训练
            client_states = []
            client_metrics = []

            for client_id, client_loader in enumerate(self.clients_loaders):
                print(f"\n--- 客户端 {client_id + 1}/{len(self.clients_loaders)} 训练 ---")

                # ---------- 模型参数管理：下发与恢复 ----------
                # 1. 备份所有潜在的个性化参数（Prompts 和 本地Experts）。
                #    这些参数后续可能会根据聚合策略被恢复，以覆盖从服务器下载的全局参数。
                personalized_state = {
                    k: v.clone()
                    for k, v in self.client_model.state_dict().items()
                    if any(pk in k for pk in ("prompt_learner", "fusion_gating", "adaptmlp_list"))
                }

                # 2. 从服务器下载最新的全局模型。
                # FIX: 将全局模型的状态字典移动到客户端设备后再加载
                global_state_on_device = {k: v.to(self.device) for k, v in self.global_model.state_dict().items()}
                self.client_model.load_state_dict(global_state_on_device, strict=False)

                # 3. 根据聚合策略，决定是否恢复本地的个性化参数。
                #    如果不恢复，则客户端将使用服务器聚合后的参数进行本轮训练。
                params_to_restore = {}
                is_first_half = self.current_round < self.com_rounds // 2
                is_final_round = self.current_round == self.com_rounds - 1

                # a) 判断是否恢复 Prompt 相关参数
                should_restore_prompts = False
                if self.aggregation_strategy in ["pFedMoP", "ablation_b"]:
                    should_restore_prompts = not is_final_round
                elif self.aggregation_strategy in ["ablation_a", "ablation_c"]:
                    should_restore_prompts = is_first_half

                if should_restore_prompts:
                    prompt_params = {k: v for k, v in personalized_state.items() if
                                     any(pk in k for pk in self._prompt_keywords)}
                    params_to_restore.update(prompt_params)

                # b) 判断是否恢复本地 MoE Expert 参数
                should_restore_local_experts = False
                if self.aggregation_strategy in ["pFedMoP", "ablation_a"]:
                    should_restore_local_experts = not is_final_round
                elif self.aggregation_strategy in ["ablation_b", "ablation_c"]:
                    should_restore_local_experts = not is_first_half  # Restore in the second half

                if should_restore_local_experts and self.upload_moe_params:
                    expert_params = {k: v for k, v in personalized_state.items() if 'adaptmlp_list' in k}
                    params_to_restore.update(expert_params)

                # 4. 执行恢复操作并打印日志
                if not params_to_restore:
                    print(f"[Client {client_id}] 使用服务器聚合后的所有参数进行训练。")
                else:
                    print(f"[Client {client_id}] 恢复部分个性化参数，其余使用服务器聚合参数。")
                    if 'prompt_learner.ctx' in params_to_restore:
                        print("  - 将恢复: 本地个性化Prompt参数")
                    else:
                        print("  - 将使用: 服务器聚合Prompt参数")

                    if self.upload_moe_params:
                        if 'adaptmlp_list.0.fc1.weight' in params_to_restore:
                            print("  - 将恢复: 本地个性化MoE专家参数")
                        else:
                            print("  - 将使用: 服务器聚合MoE专家参数")

                    current_state = self.client_model.state_dict()
                    # FIX: 将需要恢复的个性化参数也移动到客户端设备
                    params_to_restore_on_device = {k: v.to(self.device) for k, v in params_to_restore.items()}
                    current_state.update(params_to_restore_on_device)
                    self.client_model.load_state_dict(current_state)


                # --------- 随机选择全局专家 Prompt 列表 ---------
                indices = random.sample(range(self.prompt_pool_size), self.num_experts_per_client)
                expert_ctx_list = [self.prompt_pool.prompts[idx].to(self.device) for idx in indices]
                print(f"[Server] Round {global_round} | Client {client_id} 分配专家索引: {indices}")

                # --------- 服务器端 Gate 推理基于 feature ---------
                if self.client_features[client_id] is not None:
                    with torch.no_grad():
                        logits_pred = self.gate(self.client_features[client_id].to(self.device).to(torch.float32))
                    top_indices = logits_pred.argsort(descending=True)[:self.num_experts_per_client]
                    indices = top_indices.cpu().tolist()
                else:
                    # cold start
                    indices = random.sample(range(self.prompt_pool_size), self.num_experts_per_client)

                expert_ctx_list = [self.prompt_pool.prompts[idx].to(self.device) for idx in indices]
                print(f"[Server] Round {global_round} | Client {client_id} 分配专家索引: {indices}")

                # 下发 global experts
                self.client_model.set_global_experts(expert_ctx_list)

                # 训练客户端模型
                client_metric, feat_summary = self._train_client(client_id, client_loader, indices)
                client_metrics.append(client_metric)

                # 保存 feature 用于下一轮 gating 推理
                self.client_features[client_id] = feat_summary.detach().cpu()

                # ---- 状态收集 (根据消融策略调整) ----
                is_final_round = global_round == self.com_rounds - 1

                # 1. 总是收集MoE参数
                moe_state = {n: p.detach().cpu() for n, p in self.client_model.named_parameters() if any(k in n for k in self._moe_param_names)}
                client_state_for_agg = {'moe': moe_state}

                # 2. 根据策略和轮次决定是否收集个性化参数用于本轮聚合
                p_params_for_current_agg = {}

                # 策略 A/C: 在后半程的每一轮都收集prompt/gating
                if self.aggregation_strategy in ["ablation_a", "ablation_c"] and self.current_round >= self.com_rounds // 2:
                    prompt_p_state = {
                        n: p.detach().cpu() for n, p in self.client_model.named_parameters()
                        if any(k in n for k in ("prompt_learner", "fusion_gating"))
                    }
                    p_params_for_current_agg.update(prompt_p_state)

                # 最终轮总是收集所有个性化参数，以确保为下一任务正确传递知识
                # 这会覆盖上面收集的任何子集，确保最终聚合的完整性
                if is_final_round:
                    print(f"[Client {client_id}] 最后一轮，收集所有个性化参数用于最终聚合。")
                    final_p_state = {
                        n: p.detach().cpu()
                        for n, p in self.client_model.named_parameters()
                        if any(k in n for k in ("prompt_learner", "fusion_gating", "adaptmlp_list"))
                    }
                    p_params_for_current_agg.update(final_p_state)

                # 如果有需要聚合的个性化参数，则添加到待上传状态中
                if p_params_for_current_agg:
                    client_state_for_agg['personalized'] = p_params_for_current_agg

                client_states.append(client_state_for_agg)


                # 保存该客户端最新的个性化参数状态 (prompt, fusion_gating, and personalized MoE experts)
                personalized_keywords = ("prompt_learner", "fusion_gating", "adaptmlp_list")
                final_p_state = {
                    n: p.detach().cpu().clone()
                    for n, p in self.client_model.named_parameters()
                    if any(k in n for k in personalized_keywords)
                }
                self.final_client_states[client_id] = final_p_state


            # 模型聚合
            print(f"\n--- 聚合全局模型 (策略: {self.aggregation_strategy}) ---")

            global_state = self.global_model.state_dict()
            total_size = sum(self.client_sizes)
            client_weights = [size / total_size for size in self.client_sizes]

            # --------- 1. FedAvg 聚合 MoE 参数 (前半程更新) ---------
            if self.current_round < self.com_rounds // 2:
                print("\n--- 聚合 MoE 参数 ---")
                # 定义要聚合的共享参数关键字 (普通专家 adaptmlp_list 被排除)
                shared_moe_keywords = ("shared_expert", "router", "noise")
                # 策略 B/C: 在前半程的每一轮都聚合本地专家
                aggregate_local_experts_every_round = self.aggregation_strategy in ["ablation_b", "ablation_c"]

                for key in self._moe_param_names:
                    is_shared_moe = any(k in key for k in shared_moe_keywords)
                    is_local_expert = 'adaptmlp_list' in key

                    should_aggregate = is_shared_moe
                    if aggregate_local_experts_every_round and is_local_expert:
                        should_aggregate = True

                    if should_aggregate:
                        if all('moe' in state and key in state['moe'] for state in client_states):
                            # print(f"聚合共享MoE参数: {key}") # 用于调试
                            weighted_sum = torch.zeros_like(global_state[key])
                            for i, state in enumerate(client_states):
                                weighted_sum += client_weights[i] * state['moe'][key].to(global_state[key].device)
                            global_state[key] = weighted_sum

            # --------- 2. FedAvg 聚合 Personalized 参数 (根据策略和轮次) ---------
            # 只有当 'personalized' 键存在于客户端状态中时，才执行此聚合
            # 我们的状态收集逻辑确保了只有在需要聚合时才会添加此键
            if client_states and 'personalized' in client_states[0]:
                print("\n--- 聚合 Personalized 参数 ---")
                # 从第一个客户端获取要聚合的参数名列表作为代表
                personal_params_to_agg = client_states[0]['personalized'].keys()

                for key in personal_params_to_agg:
                    # 确保所有客户端都提供了这个参数
                    if all('personalized' in state and key in state['personalized'] for state in client_states):
                        weighted_sum = torch.zeros_like(global_state[key])
                        for i, state in enumerate(client_states):
                            weighted_sum += client_weights[i] * state['personalized'][key].to(global_state[key].device)
                        global_state[key] = weighted_sum
                        # print(f"已聚合个性化参数: {key}")

            # 加载聚合后的参数
            self.global_model.load_state_dict(global_state)

            # 打印训练指标
            # 打印训练指标
            for metric_name in ["loss", "accuracy"]:
                # 确保 values 是浮点数列表
                values = [m[metric_name] for m in client_metrics]

                # 如果 values 是序列（列表），转换为平均值
                if isinstance(values[0], (list, tuple)):
                    values = [numpy.mean(v) for v in values]

                weighted_avg = sum(v * w for v, w in zip(values, client_weights))
                print(f"全局轮次 {global_round + 1} 平均{metric_name}: {weighted_avg:.6f}")

            # ---- 训练 GateNetwork ----
            self._update_gate_network()
        # 任务完成后评估各客户端
        mean_acc = self.evaluate_clients()

        # -------- 任务结束：收集全局 ctx 和 fusion_gating 状态 --------
        ctx_state = self.global_model.prompt_learner.ctx.detach().clone()
        fusion_state = self.global_model.fusion_gating.state_dict()

        # --- 新增：在任务结束时，统一保存历史门控信息 ---
        if self.task_id is not None:
            try:
                # 为了获取最终的门控分布，需要用一个样本进行一次前向传播
                dummy_loader = self.clients_loaders[0]
                dummy_images, _, _ = next(iter(dummy_loader))
                dummy_images = dummy_images.to(self.device)

                with torch.no_grad():
                    self.global_model.eval()
                    # 手动清空，以防有旧数据
                    global_gate_collector['image'] = {}
                    global_gate_collector['text'] = {}
                    # 运行一次前向传播来填充收集器
                    self.global_model(dummy_images)

                # 找到最底层的 CLIP 模型来保存
                base = self.global_model.clip_model
                depth_guard = 0
                while hasattr(base, 'clip_model') and depth_guard < 5:
                    base = base.clip_model
                    depth_guard += 1

                if hasattr(base, 'historical_gates'):
                    base.historical_gates[self.task_id] = {
                        'image': {k: v['logits'].clone().detach() for k, v in global_gate_collector['image'].items()},
                        'text': {k: v['logits'].clone().detach() for k, v in global_gate_collector['text'].items()}
                    }
                    print(f"[Trainer] 已为任务 {self.task_id} 保存最终的历史门控信息。")

            except Exception as e:
                print(f"[Trainer] 警告：为任务 {self.task_id} 保存历史门控信息失败。错误: {e}")
        # ----------------------------------------------------

        # 返回训练后的全局模型、上下文状态、融合门状态、准确率历史和客户端个性化状态
        return self.global_model.clip_model, self.global_model.prompt_learner.ctx.detach().clone(), \
               self.global_model.fusion_gating.state_dict(), self.mean_acc_history, self.final_client_states, self.global_model.clip_model.historical_gates

    # ---------------- 评估函数 ----------------
    def evaluate_clients(self):

        from torch.utils.data import ConcatDataset, Subset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_accs = []  # 每个客户端准确率
        total_samples = []  # 用于打印样本数

        if self.classes_names is not None:
            # Domain-IL: 所有任务共享完整类别集，无需根据 class_order 划分
            if getattr(self.cfg, "scenario", "class") == "domain":
                seen_class_names = self.classes_names
            else:
                try:
                    class_ids_per_task = list(get_class_ids_per_task(self.cfg))
                    seen_class_ids = []
                    for t in range(self.task_id + 1):
                        seen_class_ids.extend(class_ids_per_task[t])
                    seen_class_names = get_class_names(self.classes_names, seen_class_ids)
                except Exception as e:
                    print(f"[FedDuet] 警告: 因获取类别ID失败，将回退使用所有类别名称进行评估。错误: {e}")
                    seen_class_names = self.classes_names
        else:
            seen_class_names = self.texts  # 回退到当前任务类别

        # -------- 逐客户端评估 ---------
        for client_id in range(self.num_clients):
            subsets = []
            for t in range(self.task_id + 1):
                task_dataset = self.eval_dataset[t:t + 1]
                # --- 使用预先划分好的固定测试集索引 ---
                dict_users_test = self.dict_users_test_all_tasks[t]
                client_indices = list(dict_users_test[client_id])
                if client_indices: # 确保索引列表不为空
                    subsets.append(Subset(task_dataset, client_indices))

            if not subsets:
                print(f"[评估警告] 客户端 {client_id} 在任务 0-{self.task_id} 上没有测试数据，跳过评估。")
                client_accs.append(0.0) # 或者 np.nan
                total_samples.append(0)
                continue

            concat_test = ConcatDataset(subsets)
            test_loader = DataLoader(concat_test, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=getattr(self.cfg, "num_workers", 1))

            # --- 为每个客户端构建其完整的、最新的模型 ---
            # 1. 深度拷贝最终的全局模型（包含聚合后的共享参数）
            local_model = copy.deepcopy(self.global_model).to(device)

            # 2. 加载该客户端最终的个性化参数
            p_state = self.final_client_states[client_id]
            if p_state:
                # 将保存在CPU的参数加载到当前设备
                p_state_on_device = {k: v.to(device) for k, v in p_state.items()}
                local_model.load_state_dict(p_state_on_device, strict=False)


            # 若存在全局类别列表，则扩展 PromptLearner 以覆盖所有已见类别
            if self.classes_names is not None:
                try:
                    prev_ctx_tmp = local_model.prompt_learner.ctx.detach().clone()
                except AttributeError:
                    prev_ctx_tmp = None
                if hasattr(local_model, 'update_prompt_learner'):
                    local_model.update_prompt_learner(prev_ctx=prev_ctx_tmp, new_classnames=seen_class_names)

            local_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels, _ in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = local_model(imgs)[0] if isinstance(local_model(imgs), tuple) else local_model(imgs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = 100.0 * correct / total if total else 0.0
            client_accs.append(acc)
            total_samples.append(total)

        # -------- 打印统计信息 --------
        mean_acc = sum(client_accs) / len(client_accs) if client_accs else 0.0

        print(f"[FedDuet] 任务 {self.task_id} | 平均准确率: {mean_acc:.2f}%")

        # --- 使用内存中的历史记录计算 avg_acc ---
        prev_accs = self.mean_acc_history  # 之前任务的记录
        total_tasks_done = len(prev_accs) + 1
        avg_acc = (sum(prev_accs) + mean_acc) / total_tasks_done

        # 记录并更新历史
        self.mean_acc_history.append(mean_acc)

        # 记录日志，包含累计avg_acc
        import json, os
        dir_name = os.path.dirname(self.cfg.log_path)
        base = os.path.basename(self.cfg.log_path)
        path = os.path.join(dir_name, f"fedduet_client_{base}")

        log_entry = {
            "task": self.task_id,
            "acc": round(mean_acc, 2),
            "client_acc": [round(a, 2) for a in client_accs],
            "avg_acc": round(avg_acc, 2)
        }

        with open(path, 'a+') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"[FedDuet] 任务 {self.task_id} | 累计平均准确率 (avg_acc): {avg_acc:.2f}%")

        return mean_acc

    # ---------------- 全局Prompt池更新 ----------------
    def _update_prompt_pool(self):
        """全局Prompt池不直接通过梯度更新，而是通过聚合客户端上传的专家/prompt。此函数占位。"""
        return

    # ---------------- 门控网络(Gate Network)训练 ----------------
    def _update_gate_network(self):
        """使用客户端上传的 <特征, 专家索引> 对来训练服务器端的门控网络。"""
        if not self._gate_train_buffer:
            return

        feats = torch.stack([f for f, _, _ in self._gate_train_buffer]).to(self.device)
        target_indices_list = [indices for _, indices, _ in self._gate_train_buffer]
        losses = torch.tensor([loss for _, _, loss in self._gate_train_buffer], device=self.device, dtype=torch.float)

        # 创建多标签 one-hot 编码的目标
        targets = torch.zeros(len(target_indices_list), self.prompt_pool_size, device=self.device)
        for i, indices in enumerate(target_indices_list):
            if isinstance(indices, int):  # 兼容旧的单个索引格式
                indices = [indices]
            targets[i, indices] = 1.0

        # 根据客户端损失计算权重（损失越小，权重越高）
        eps = 1e-6
        weights = 1.0 / (losses + eps)
        weights = weights / weights.mean()  # 归一化

        # 前向传播
        logits = self.gate(feats.to(torch.float32))  # (客户端数量, prompt_pool_size)

        # 计算加权的多标签损失 (BCEWithLogitsLoss)
        # 逐样本计算损失，然后加权
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(dim=1)  # 在类别维度上平均
        loss = (loss_bce * weights).mean()  # 在批次维度上加权平均

        self.gate_optimizer.zero_grad()
        loss.backward()
        self.gate_optimizer.step()

        print(f"[Gate] 更新完成，loss={loss.item():.4f}, batch={feats.size(0)}")

        self._gate_train_buffer.clear()


def fedduet_train(global_model, train_dataset, eval_dataset, cfg, texts, task_id, client_model,
                             prev_ctx=None, prev_fusion_state=None, prev_mean_acc_history=None, classes_names=None,
                             prev_client_states=None, historical_gates=None):
    """FedDuet训练的入口函数，负责初始化和启动训练流程。"""
    if hasattr(cfg, "seed"):
        seed = cfg.seed
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    print("\n=== FedDuet 训练配置 ===")
    print(f"联邦通信轮次: {getattr(cfg, 'com', 10)}")
    print(f"客户端本地训练轮次: {getattr(cfg, 'client_epochs', 5)}")
    print(f"学习率: {getattr(cfg, 'lr', 3e-5)}")
    print(f"客户端数量: {getattr(cfg, 'num_clients', 5)}")
    print(f"数据分布: {'IID' if getattr(cfg, 'iid', True) else 'Non-IID'}")
    print(f"类别数量: {len(texts)}")
    # print(f"类别列表: {texts}")
    # print(f"任务ID: {task_id}")

    wrapped_global_model = CustomCLIP(cfg, texts, global_model, prev_ctx, prev_fusion_state, historical_gates)
    wrapped_client_model = CustomCLIP(cfg, texts, client_model, prev_ctx, prev_fusion_state, historical_gates)
    
    trainer = FedDuetTrainer(
        cfg=cfg,
        global_model=wrapped_global_model,
        client_model=wrapped_client_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_id=task_id,
        texts=texts,
        prev_ctx=prev_ctx,
        prev_fusion_state=prev_fusion_state,
        prev_mean_acc_history=prev_mean_acc_history,
        classes_names=classes_names,
        prev_client_states=prev_client_states
    )

    trained_model, ctx_state, fusion_state, mean_acc_history, final_client_states, updated_historical_gates = trainer.train()

    return trained_model, ctx_state, fusion_state, mean_acc_history, final_client_states, updated_historical_gates


class MultiheadAttention(nn.Module):
    """一个标准的多头注意力实现，用于特征融合。"""
    def __init__(self, d_model, num_heads, dropout=0.2, scaling=1.0, dtype=torch.float32):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # self.scaling = self.embed_dim ** -0.5
        self.scaling = scaling
        self.dtype = dtype

        self.W_q = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_k = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_v = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_o = nn.Linear(d_model, d_model, dtype=self.dtype)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, torch.mean(attn_probs, dim=1)

