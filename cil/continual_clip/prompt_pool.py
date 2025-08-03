"""Prompt Pool 模块
====================

本文件实现了以下组件：

1. PromptPool
   - 维护 K 个可被客户端选用的 prompt 上下文向量 (n_ctx, ctx_dim)。
   - 支持按权重或 Top-m 的方式合成加权 prompt 向量。
   - 当前实现将 prompt 存为 *buffer*，即默认 **不在服务器端训练**。
     客户端若使用 LoRA 等方法细化 prompt，可将增量上载，由服务器聚合后
     更新 `self.prompts`（目前尚未实现聚合逻辑，后续可扩展）。

2. GateNetwork
   - 一个简单的两层 MLP，将客户端特征摘要 fᵢ → K 维 logits。

3. SparseGateLoss
   - MoE 中常用的 importance loss，用于鼓励 Gate 的稀疏分配，避免所有
     权重集中到同一个 prompt。

提示：
若要支持 PromptPool 的 *联邦更新*（即 pₖ → pₖ′），可以：
  • 让客户端训练 LoRA ΔA, ΔB（或全量 Δpₖ）；
  • 回传至服务器后对相应 prompt 做 FedAvg；
  • 更新后调用 `prompt_pool.prompts.copy_(new_tensor)`。

"""
import os

import torch
import torch.nn as nn
from typing import Optional, List

# 可选：若安装 scikit-learn 可用 KMeans 聚类做初始化
try:
    from sklearn.cluster import KMeans
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


# ---------------------------------------------------------------------------
# PromptPool
# ---------------------------------------------------------------------------
#
# 设计理念：
#   Server 持有一个大小为 K 的 prompt 池，每个 prompt 为长度 n_ctx 的 token
#   上下文向量。Client 依据 Gate 输出的权重 w 选择或加权组合 Prompt。
# ---------------------------------------------------------------------------
class PromptPool(nn.Module):
    """A pool that stores K prompt context tensors of shape (n_ctx, ctx_dim).

    Args:
        K (int): number of prompt candidates in the pool.
        base_prompt (torch.Tensor): a context tensor of shape (n_ctx, ctx_dim) used
            to initialize the pool when no pretrained prompts are provided.
        pretrained_prompts (Optional[torch.Tensor]): optional tensor of shape
            (K, n_ctx, ctx_dim) to initialize the pool. If provided its shape must
            match (K, n_ctx, ctx_dim).
    """

    def __init__(self, K: int, base_prompt: torch.Tensor,
                 pretrained_prompts: Optional[torch.Tensor] = None,
                 init_by_kmeans: bool = True,
                 clip_model=None,
                 class_file: str = "cil/dataset_reqs/imagenet1000_classes.txt"):
        super().__init__()
        n_ctx, ctx_dim = base_prompt.shape
        if pretrained_prompts is not None:
            assert pretrained_prompts.shape == (K, n_ctx, ctx_dim), (
                "pretrained_prompts should have shape (K, n_ctx, ctx_dim)")
            prompt_data = pretrained_prompts.clone().detach()
        else:
            if init_by_kmeans and _HAS_SK and clip_model is not None and os.path.exists(class_file):
                try:
                    prompt_data = self._init_kmeans(K, n_ctx, ctx_dim, clip_model, class_file)
                except Exception as e:
                    print("[PromptPool] KMeans 初始化失败，fallback 随机:", e)
                    prompt_data = self._random_init(K, base_prompt)
            else:
                prompt_data = self._random_init(K, base_prompt)

        # 以 *buffer* 形式保存，意味着默认不参与梯度更新。
        # 若需要在服务器端微调，可改为 nn.Parameter。
        self.register_buffer("prompts", prompt_data)  # (K, n_ctx, ctx_dim)

    @staticmethod
    def _random_init(K:int, base_prompt:torch.Tensor):
        prompt_data = base_prompt.clone().detach().unsqueeze(0).repeat(K,1,1)
        prompt_data = prompt_data + 0.02*torch.randn_like(prompt_data)
        return prompt_data

    @staticmethod
    def _tokenize_names(names:List[str], clip_model):
        import clip
        tokenized = torch.cat([clip.tokenize(n) for n in names]).to(next(clip_model.parameters()).device)
        with torch.no_grad():
            embeds = clip_model.token_embedding(tokenized).type(clip_model.dtype)
        return embeds

    def _init_kmeans(self, K, n_ctx, ctx_dim, clip_model, class_file):
        import os, random
        with open(class_file,'r') as rf:
            names=[l.strip() for l in rf if l.strip()]
        if len(names)<K:
            raise RuntimeError("classname list shorter than K")
        # 取全部类别
        embeds = self._tokenize_names(names, clip_model)[:,1:1+n_ctx,:]  # (N,n_ctx,d)
        vecs = embeds.mean(dim=1).cpu().numpy()
        kmeans = KMeans(n_clusters=K, random_state=0).fit(vecs)
        centers=torch.tensor(kmeans.cluster_centers_,dtype=clip_model.dtype) # (K,d)
        prompt_data=centers.unsqueeze(1).repeat(1,n_ctx,1)
        return prompt_data

    @torch.no_grad()
    def get_prompt(self, weights: torch.Tensor, top_m: Optional[int] = None):
        """Return a weighted prompt given selection probabilities.
        Args:
            weights (tensor): shape (K,) after softmax.
            top_m (int, optional): if specified, only keep the top-m weights,
                renormalise and combine.
        Returns:
            tensor of shape (n_ctx, ctx_dim)
        """
        # -------- Top-m 策略（可选） --------
        if top_m is not None and top_m < weights.numel():
            # zero out weights except top-m
            topk = torch.topk(weights, top_m)
            mask = torch.zeros_like(weights)
            mask[topk.indices] = 1.0
            weights = weights * mask
            weights = weights / weights.sum()
        # -------- 加权求和得到组合 prompt --------
        ctx = torch.einsum('k,knd->nd', weights, self.prompts)
        return ctx


# ---------------------------------------------------------------------------
# GateNetwork
# ---------------------------------------------------------------------------
# 输入：客户端特征摘要向量 fᵢ (dim=d)
# 输出：对 K 个 prompt 的 logits
# ---------------------------------------------------------------------------
class GateNetwork(nn.Module):
    """Simple MLP gate that maps a client feature vector to logits over K prompts."""

    def __init__(self, in_dim: int, K: int, hidden_dim: int = 512):
        super().__init__()
        # 两层 MLP + ReLU
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# SparseGateLoss
# ---------------------------------------------------------------------------
# 重要性正则：鼓励 Gate 的平均分配向量接近稀疏。
# ---------------------------------------------------------------------------
class SparseGateLoss(nn.Module):
    """Importance loss to encourage sparse gate activations (similar to MoE)."""

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, probs: torch.Tensor):
        """probs: (B, K) softmax outputs."""
        # 平均重要性 importance_k = mean_b p_{b,k}
        importance = probs.mean(dim=0)
        # KL 与均匀分布等价的负熵，可产生稀疏效果
        loss = self.alpha * (importance * torch.log(importance + 1e-8)).sum()
        return loss 