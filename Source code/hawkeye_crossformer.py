
# hawkeye_crossformer.py
# MIT License
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------
# Utilities
# --------------------

def make_episodes(X: torch.Tensor, len_eps: int, stride: int) -> Tuple[torch.Tensor, int]:
    """
    Turn a [B, T, D] tensor into episodes along time for each dimension.
    Returns Z with shape [B, E, D, len_eps] where E is number of episodes.
    """
    B, T, D = X.shape
    # print("B T D", B, T, D)
    E = 1 + (T - len_eps) // stride if T >= len_eps else 0
    if E <= 0:
        return torch.empty(B, 0, D, len_eps, device=X.device), 0
    wins = []
    for i in range(0, T - len_eps + 1, stride):
        wins.append(X[:, i:i+len_eps, :].unsqueeze(1))  # [B,1,len_eps,D]
    W = torch.cat(wins, dim=1)  # [B, E, len_eps, D]
    W = W.permute(0, 1, 3, 2).contiguous()  # [B, E, D, len_eps]
    # print("E len_eps", E, len_eps)
    return W, E

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        # print(x, self.pe)
        return x + self.pe[:L].unsqueeze(0)

# --------------------
# Two-Stage Attention
# --------------------

class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pos = PositionalEncoding(d_model)

    def forward(self, x):
        # x: [B, E, D, C]
        B, E, D, C = x.shape
        z = x.permute(0,2,1,3).contiguous().view(B*D, E, C)  # [B*D, E, C]
        z = self.pos(z)
        z = self.encoder(z)  # [B*D, E, C]
        z = z.view(B, D, E, C).permute(0,2,1,3).contiguous()  # [B, E, D, C]
        return z

class RouterAttention(nn.Module):
    def __init__(self, d_model: int, routers: int = 3, nhead: int = 1, dropout: float = 0.0):
        super().__init__()
        self.routers = routers
        self.router_tokens = nn.Parameter(torch.randn(routers, d_model) * 0.02)
        self.attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x):
        # x: [B, E, D, C]
        B, E, D, C = x.shape
        z = x.view(B*E, D, C)  # [B*E, D, C]
        routers = self.router_tokens.unsqueeze(0).expand(B*E, -1, -1)  # [B*E, R, C]
        r, _ = self.attn1(query=routers, key=z, value=z)  # [B*E, R, C]
        r = self.norm1(routers + r)
        r = r + self.mlp(r)
        z2, _ = self.attn2(query=z, key=r, value=r)       # [B*E, D, C]
        z2 = self.norm2(z + z2)
        z2 = z2 + self.mlp(z2)
        out = z2.view(B, E, D, C)
        return out

class TSA(nn.Module):
    def __init__(self, d_model: int, nhead_time: int, nhead_dim: int, dim_ff: int, dropout: float, routers: int):
        super().__init__()
        self.temporal = TemporalBlock(d_model, nhead_time, dim_ff, dropout)
        self.router = RouterAttention(d_model, routers=routers, nhead=nhead_dim, dropout=dropout)

    def forward(self, x):
        x = self.temporal(x)
        x = self.router(x)
        return x

# --------------------
# Hierarchical Encoder-Decoder
# --------------------

class MergeEpisodes(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fuse = nn.Linear(2*d_model, d_model)

    def forward(self, x):
        B, E, D, C = x.shape
        if E < 2:
            return x, x
        if E % 2 == 1:
            x = x[:, :-1, :, :]
            E = x.shape[1]
        x1 = x[:, 0::2, :, :]
        x2 = x[:, 1::2, :, :]
        z = torch.cat([x1, x2], dim=-1)
        z = self.fuse(z)
        return z, x

class HEDLayer(nn.Module):
    def __init__(self, d_model: int, nhead_time: int, nhead_dim: int, dim_ff: int, dropout: float, routers: int):
        super().__init__()
        self.merge = MergeEpisodes(d_model)
        self.tsa = TSA(d_model, nhead_time, nhead_dim, dim_ff, dropout, routers)

    def forward(self, x):
        z_coarse, cache = self.merge(x)
        z_coarse = self.tsa(z_coarse)
        return z_coarse, cache

class HEDDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, enc_caches: List[torch.Tensor]) -> torch.Tensor:
        target_E = enc_caches[0].shape[1]
        out = torch.zeros_like(enc_caches[0])
        for z in enc_caches[::-1]:
            E = z.shape[1]
            rep = max(1, target_E // max(1,E))
            z_up = z.repeat_interleave(rep, dim=1)
            if z_up.shape[1] > target_E:
                z_up = z_up[:, :target_E, :, :]
            elif z_up.shape[1] < target_E:
                pad = target_E - z_up.shape[1]
                z_up = torch.nn.functional.pad(z_up, (0,0,0,0,0,pad))
            out = out + self.proj(z_up)
        return out

# --------------------
# Full Model
# --------------------

from dataclasses import dataclass

@dataclass
class HawkeyeCfg:
    len_eps: int = 24
    stride_eps: int = 12
    d_model: int = 64
    nhead_time: int = 4
    nhead_dim: int = 1
    dim_ff: int = 128
    dropout: float = 0.1
    routers: int = 3
    hed_layers: int = 2
    device: str = "cpu"

class DSWEmbed(nn.Module):
    def __init__(self, in_dim: int, d_model: int, len_eps: int):
        super().__init__()
        self.proj = nn.Linear(len_eps, d_model)
        self.dim_pe = nn.Parameter(torch.randn(1, 1, in_dim, d_model) * 0.01)
        self.time_pe = PositionalEncoding(d_model)

    def forward(self, episodes):
        z = self.proj(episodes)           # [B,E,D,d_model]
        z = z + self.dim_pe
        B,E,D,C = z.shape
        z_flat = z.view(B*D, E, C)
        z_flat = self.time_pe(z_flat).view(B, E, D, C)
        return z_flat

class CrossformerHED(nn.Module):
    def __init__(self, in_dim: int, cfg: HawkeyeCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = DSWEmbed(in_dim, cfg.d_model, cfg.len_eps)
        self.hed_blocks = nn.ModuleList([HEDLayer(cfg.d_model, cfg.nhead_time, cfg.nhead_dim, cfg.dim_ff, cfg.dropout, cfg.routers) for _ in range(cfg.hed_layers)])
        self.decoder = HEDDecoder(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x):
        episodes, E = make_episodes(x, self.cfg.len_eps, self.cfg.stride_eps)
        if E == 0:
            raise ValueError("Time series too short for current episode settings.")
        z = self.embed(episodes)
        caches = []
        cur = z
        for blk in self.hed_blocks:
            cur, cache = blk(cur)
            caches.append(cache)
        dec_out = self.decoder(caches)  # [B,E_fine,D,C]
        yhat = self.head(dec_out).squeeze(-1)  # [B,E_fine,D]
        return yhat, episodes

# --------------------
# Dataset & Training
# --------------------

class SeqForecastDataset(Dataset):
    def __init__(self, X: np.ndarray, len_eps: int, stride: int, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()
        Xn = self.scaler.fit_transform(X)
        self.X = torch.tensor(Xn, dtype=torch.float32)
        self.len_eps = len_eps
        self.stride = stride

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X

@dataclass
class TrainCfg:
    batch_size: int = 1
    lr: float = 1e-3
    epochs: int = 10
    device: str = "cpu"
    len_eps: int = 24
    stride_eps: int = 12
    d_model: int = 64
    nhead_time: int = 4
    nhead_dim: int = 1
    dim_ff: int = 128
    dropout: float = 0.1
    routers: int = 3
    hed_layers: int = 2

class HawkeyeTrainer:
    def __init__(self, input_dim: int, cfg: TrainCfg):
        self.cfg = cfg
        self.model = CrossformerHED(input_dim, HawkeyeCfg(
            len_eps=cfg.len_eps, stride_eps=cfg.stride_eps, d_model=cfg.d_model,
            nhead_time=cfg.nhead_time, nhead_dim=cfg.nhead_dim, dim_ff=cfg.dim_ff,
            dropout=cfg.dropout, routers=cfg.routers, hed_layers=cfg.hed_layers, device=cfg.device
        )).to(cfg.device)
        self.scaler = StandardScaler()

    def fit(self, series: np.ndarray):
        ds = SeqForecastDataset(series, self.cfg.len_eps, self.cfg.stride_eps, scaler=self.scaler)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.cfg.epochs):
            for xb in dl:
                xb = xb.to(self.cfg.device)
                yhat, episodes = self.model(xb)  # [B,E,D], [B,E,D,L]
                B,E,D,L = episodes.shape
                targets = []
                for e in range(E):
                    start = e * self.cfg.stride_eps
                    ti = min(start + L, xb.shape[1]-1)  # next step (clip)
                    targets.append(xb[:, ti, :])
                ytrue = torch.stack(targets, dim=1)  # [B,E,D]
                loss = loss_fn(yhat, ytrue)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {"train_mse": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def forecast_series(self, series: np.ndarray):
        Xn = self.scaler.transform(series)
        x = torch.tensor(Xn, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        self.model.eval()
        yhat, episodes = self.model(x)  # [1,E,D], [1,E,D,L]
        yhat = yhat.squeeze(0).cpu().numpy()
        E = yhat.shape[0]
        L = episodes.shape[-1]
        starts = np.arange(E) * self.cfg.stride_eps
        tgt_idx = starts + L
        T, D = series.shape
        preds = np.zeros_like(yhat)
        gts = np.zeros_like(yhat)
        for e in range(E):
            ti = tgt_idx[e]
            if ti >= T:
                ti = T - 1
            preds[e] = self.scaler.inverse_transform(yhat[e].reshape(1,-1)).ravel()
            gts[e]   = series[ti]
        residuals = gts - preds
        return preds, residuals, starts, tgt_idx

# --------------------
# Residual clustering
# --------------------

def residual_cluster_scores(residuals: np.ndarray, k: int = 3):
    E, D = residuals.shape
    if E == 0:
        return np.zeros(0), np.zeros(0, dtype=int), {}
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(residuals)
    labels = km.labels_
    norms = []
    for c in range(k):
        if np.any(labels==c):
            norms.append(np.linalg.norm(residuals[labels==c], axis=1).mean())
        else:
            norms.append(-1)
    anomalous = int(np.argmax(norms))
    mapping = {c: ("anomalous" if c==anomalous else "normalish") for c in range(k)}
    centroids = km.cluster_centers_
    d_anom = np.linalg.norm(residuals - centroids[anomalous], axis=1)
    s = (d_anom - d_anom.min()) / (d_anom.max() - d_anom.min() + 1e-8)
    return s, labels, mapping
