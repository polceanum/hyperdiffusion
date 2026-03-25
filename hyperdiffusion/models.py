from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetArchitecture:
    in_dim: int = 2
    hidden_dim: int = 64
    out_dim: int = 1
    depth: int = 4

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self.depth < 2:
            raise ValueError("Target architecture depth must be at least 2.")
        shapes: Dict[str, Tuple[int, ...]] = {
            "w1": (self.hidden_dim, self.in_dim),
            "b1": (self.hidden_dim,),
        }
        for layer_idx in range(2, self.depth):
            shapes[f"w{layer_idx}"] = (self.hidden_dim, self.hidden_dim)
            shapes[f"b{layer_idx}"] = (self.hidden_dim,)
        shapes[f"w{self.depth}"] = (self.out_dim, self.hidden_dim)
        shapes[f"b{self.depth}"] = (self.out_dim,)
        return shapes

    @property
    def num_params(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes.values())


class DeepSetEncoder(nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 1, hidden_dim: int = 128, cond_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.item_mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([support_x, support_y], dim=-1)
        h = self.item_mlp(h)
        pooled = h.mean(dim=1)
        context = self.context_mlp(pooled)
        latent = self.latent_mlp(context)
        return context, latent


class AttentionSetBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(self.norm_q(q), self.norm_kv(x), self.norm_kv(x), need_weights=False)
        return pooled[:, 0]


class AttentionSetEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 1,
        hidden_dim: int = 128,
        cond_dim: int = 64,
        latent_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.item_proj = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            AttentionSetBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.pool = AttentionPool(hidden_dim, num_heads=num_heads)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([support_x, support_y], dim=-1)
        h = self.item_proj(h)
        for layer in self.layers:
            h = layer(h)
        pooled = self.pool(h)
        context = self.context_mlp(pooled)
        latent = self.latent_mlp(context)
        return context, latent


class HyperDecoder(nn.Module):
    def __init__(self, arch: TargetArchitecture, latent_dim: int = 32, cond_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.arch = arch
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, arch.num_params),
        )

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = self.backbone(torch.cat([z, context], dim=-1))
        return unpack_parameters(flat, self.arch)


class ResidualFiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.affine = nn.Linear(cond_dim, 2 * dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DiffusionDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 32, cond_dim: int = 64, hidden_dim: int = 128, depth: int = 4, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualFiLMBlock(hidden_dim, hidden_dim) for _ in range(depth)])
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_embed = sinusoidal_time_embedding(t, self.time_dim)
        cond = self.cond_proj(torch.cat([context, t_embed], dim=-1))
        h = self.in_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


class HyperNetworkSystem(nn.Module):
    def __init__(
        self,
        arch: TargetArchitecture,
        cond_dim: int = 64,
        latent_dim: int = 32,
        encoder_hidden: int = 128,
        decoder_hidden: int = 256,
        encoder_type: str = "attention",
        attention_heads: int = 4,
        attention_layers: int = 3,
        x_dim: int = 2,
        y_dim: int = 1,
    ):
        super().__init__()
        encoder_type = encoder_type.lower()
        if encoder_type == "attention":
            self.encoder = AttentionSetEncoder(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dim=encoder_hidden,
                cond_dim=cond_dim,
                latent_dim=latent_dim,
                num_heads=attention_heads,
                num_layers=attention_layers,
            )
        elif encoder_type == "deepset":
            self.encoder = DeepSetEncoder(x_dim=x_dim, y_dim=y_dim, hidden_dim=encoder_hidden, cond_dim=cond_dim, latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.decoder = HyperDecoder(arch=arch, latent_dim=latent_dim, cond_dim=cond_dim, hidden_dim=decoder_hidden)

    def encode(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(support_x, support_y)

    def decode(self, z: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(z, context)

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        context, latent = self.encode(support_x, support_y)
        params = self.decode(latent, context)
        return context, latent, params


def unpack_parameters(flat: torch.Tensor, arch: TargetArchitecture) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    start = 0
    for name, shape in arch.shapes.items():
        n = math.prod(shape)
        params[name] = flat[:, start : start + n].reshape(flat.shape[0], *shape)
        start += n
    return params


def functional_target_network(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    layer_ids: List[int] = sorted({int(name[1:]) for name in params if name.startswith("w")})
    h = x
    last_layer = max(layer_ids)
    for layer_idx in layer_ids:
        w = params[f"w{layer_idx}"]
        b = params[f"b{layer_idx}"]
        h = torch.einsum("bni,boi->bno", h, w) + b.unsqueeze(1)
        if layer_idx != last_layer:
            h = F.relu(h)
    return h


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
