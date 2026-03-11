import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
        
class MultiScaleAnomalyAttention(nn.Module):
    """
    Multi-scale variant:
      - Runs attention at several local radii (neighborhoods) and an optional global head.
      - Each scale has its own learnable gain on sigma (sigma_scale) and fusion weight.
      - Returns fused context; if output_attention=True, also returns per-scale series/prior/sigma.
    """
    def __init__(
        self,
        win_size,
        radii,
        use_global=True,
        mask_flag=True,
        scale=None,
        attention_dropout=0.0,
        output_attention=False,
    ):
        super().__init__()
        self.radii = list(sorted(set(int(r) for r in radii if r > 0)))
        self.use_global = use_global
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # distance matrix for priors / local masks
        self.register_buffer(
            "distances",
            torch.arange(win_size).float().unsqueeze(0) - torch.arange(win_size).float().unsqueeze(1)
        )
        self.distances = self.distances.abs()  # [L,L]
        # print("distances_init", self.distances)
        # learnable per-scale fusion weights (including optional global)
        n_scales = len(self.radii) + (1 if self.use_global else 0)
        self.fusion_logits = nn.Parameter(torch.zeros(n_scales))

        # learnable per-scale sigma multipliers (softplus to keep positive)
        self.sigma_scale_logits = nn.Parameter(torch.zeros(n_scales))

    # def _maybe_resize_distances(self, L, device):
    #     if self.distances.size(0) != L:
    #         d = torch.arange(L, device=device).float()
    #         return (d[None, :] - d[:, None]).abs()
    #     return self.distances.to(device)

    def _masked_softmax(self, logits, mask_bool):
        logits = logits.masked_fill(mask_bool, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(self, queries, keys, values, sigma, attn_mask):
        """
        queries,keys,values: [B, L, H, E]
        sigma: [B, L, H]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        device = queries.device
        scale = self.scale or 1. / sqrt(E)

        # base attention logits
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B,H,L,S(=L)]
        # base mask (causal)
        base_mask = None
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=device)
            base_mask = attn_mask.mask  # [B,1,L,L]

        # common sigma preproc
        sigma_bhl = sigma.transpose(1, 2)  # [B,H,L]
        sigma_bhll = sigma_bhl.unsqueeze(-1).repeat(1, 1, 1, L)  # [B,H,L,L]
        # print("distances_forward", self.distances)
        if self.distances.size(0) != L:
            d = torch.arange(L, device=device).float()
            self.distances = (d[None, :] - d[:, None]).abs().to("cuda")
        
        # distances = self._maybe_resize_distances(L, device)  # [L,L]

        per_scale_series = []
        per_scale_prior = []
        per_scale_sigma = []

        # softplus multipliers and fusion weights
        sigma_mult = F.softplus(self.sigma_scale_logits) + 1e-5  # [n_scales]
        fuse_w = torch.softmax(self.fusion_logits, dim=0)         # [n_scales]

        fused_context = 0.0
        scale_idx = 0

        def get_attention(local_mask_bool, sigma_scale_scalar, idx_out):
            # logits + masks
            attn_logits = scale * scores
            if base_mask is not None:
                attn_logits = attn_logits.masked_fill(base_mask, float("-inf"))
            if local_mask_bool is not None:
                attn_logits = attn_logits.masked_fill(local_mask_bool, float("-inf"))

            # series association
            series = self.dropout(torch.softmax(attn_logits, dim=-1))  # [B,H,L,L]

            # prior association (Gaussian with scaled sigma)
            sigma_scaled = (torch.sigmoid(sigma_bhl * 5) + 1e-5)
            sigma_scaled = torch.pow(3, sigma_scaled) - 1
            sigma_scaled = sigma_scaled * sigma_scale_scalar
            sigma_scaled = sigma_scaled.unsqueeze(-1).repeat(1, 1, 1, L)  # [B,H,L,L]

            prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)  # [B,H,L,L]
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma_scaled) * torch.exp(-prior ** 2 / (2 * (sigma_scaled ** 2)))

            # context
            V = torch.einsum("bhls,bshd->blhd", series, values)  # [B,L,H,D]

            # collect
            per_scale_series.append(series if self.output_attention else None)
            per_scale_prior.append(prior if self.output_attention else None)
            per_scale_sigma.append(sigma_scaled if self.output_attention else None)

            return V  # [B,L,H,D]

        # local scales (neighborhood masks)
        # Build one static distance mask (broadcastable), then compare to each radius.
        dist_mat = self.distances.to(device)  # [L,L]
        dist_mask_template = dist_mat.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        if base_mask is not None:
            # ensure shape compatibility; base_mask is [B,1,L,L]
            pass

        for r in self.radii:
            # mask positions farther than r
            local_mask_bool = dist_mask_template > r  # [1,1,L,L]
            local_mask_bool = local_mask_bool.expand(B, 1, L, L)  # [B,1,L,L]
            V_r = get_attention(local_mask_bool, sigma_mult[scale_idx], scale_idx)
            fused_context = fused_context + fuse_w[scale_idx] * V_r
            scale_idx += 1

        # optional global head (no local distance mask)
        if self.use_global:
            V_g = get_attention(local_mask_bool=None, sigma_scale_scalar=sigma_mult[scale_idx], idx_out=scale_idx)
            fused_context = fused_context + fuse_w[scale_idx] * V_g

        # fuse heads (sum), then reshape to [B,L, H*D]
        out = fused_context.contiguous()

        if self.output_attention:
            return (out, per_scale_series, per_scale_prior, per_scale_sigma)
        else:
            return (out, None)

class MultiScaleAttentionLayer(nn.Module):
    """
    Same interface as AttentionLayer, but expects a MultiScaleAnomalyAttention inner module.
    Returns fused output; if output_attention=True, series/prior/sigma are lists per-scale.
    """
    def __init__(self, attention: MultiScaleAnomalyAttention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.output_attention = getattr(attention, "output_attention", False)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series_list, prior_list, sigma_list = self.inner_attention(
            queries, keys, values, sigma, attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        # For compatibility with encoder that expects tensors, we pass lists through.
        return out, series_list, prior_list, sigma_list


# ---------------- USAGE (example) ----------------
# In your model builder, swap the single-scale layer with the multi-scale one:
#
# attention = MultiScaleAnomalyAttention(
#     win_size=win_size,
#     radii=(16, 64, 256),     # token radii for local windows
#     use_global=True,
#     mask_flag=False,         # keep False to match original unless you need causal streaming
#     attention_dropout=dropout,
#     output_attention=output_attention,
# )
# layer = MultiScaleAttentionLayer(attention, d_model, n_heads)