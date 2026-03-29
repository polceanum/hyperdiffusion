"""Direct baseline: end-to-end prediction without fast weights/hypernetworks."""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import AttentionSetEncoder


class DirectPredictor(nn.Module):
    """End-to-end model: encode support set, predict query outputs directly (no weights)."""

    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 1,
        encoder_hidden: int = 128,
        cond_dim: int = 64,
        attention_heads: int = 4,
        attention_layers: int = 3,
    ):
        super().__init__()
        # Attention set encoder: processes support set to context vector
        self.encoder = AttentionSetEncoder(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dim=encoder_hidden,
            cond_dim=cond_dim,
            latent_dim=cond_dim,  # For direct mode, latent_dim = cond_dim (we only use context)
            num_heads=attention_heads,
            num_layers=attention_layers,
        )
        
        # Direct prediction head: concat query_x with context -> query predictions
        self.pred_head = nn.Sequential(
            nn.Linear(x_dim + cond_dim, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, y_dim),
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_x: (batch, support_size, x_dim)
            support_y: (batch, support_size, y_dim)
            query_x: (batch, query_size, x_dim)

        Returns:
            pred: (batch, query_size, y_dim)
        """
        # Encode support set to context
        context, _ = self.encoder(support_x, support_y)  # (batch, cond_dim)
        
        # Expand context for all query points
        batch_size, query_size = query_x.shape[:2]
        context_expanded = context.unsqueeze(1).expand(batch_size, query_size, -1)  # (batch, query_size, cond_dim)
        
        # Concatenate query inputs with context
        combined = torch.cat([query_x, context_expanded], dim=-1)  # (batch, query_size, x_dim + cond_dim)
        
        # Predict outputs
        pred = self.pred_head(combined)  # (batch, query_size, y_dim)
        return pred


class DirectTextProjector(nn.Module):
    """Projects text embeddings to context vector."""

    def __init__(self, text_dim: int = 768, cond_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        return self.proj(text_embed)


class DirectSystem(nn.Module):
    """Direct prediction system supporting multiple encoding modalities."""

    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 1,
        encoder_hidden: int = 128,
        cond_dim: int = 64,
        attention_heads: int = 4,
        attention_layers: int = 3,
        text_dim: int = 768,
        text_mix_alpha: float = 0.5,
        num_families: int = 1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.text_mix_alpha = text_mix_alpha
        
        # Main predictor
        self.predictor = DirectPredictor(
            x_dim=x_dim,
            y_dim=y_dim,
            encoder_hidden=encoder_hidden,
            cond_dim=cond_dim,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
        )
        
        # Text projection for text/hybrid modes
        self.text_projector = DirectTextProjector(text_dim=text_dim, cond_dim=cond_dim)

    def encode_support(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        """Encode support set to context."""
        context, _ = self.predictor.encoder(support_x, support_y)
        return context

    def encode_text(self, text_embed: torch.Tensor) -> torch.Tensor:
        """Project text embedding to context."""
        return self.text_projector(text_embed)

    def encode_oracle(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert one-hot to context (oracle mode)."""
        # Simple projection of one-hot to context space
        proj = nn.Linear(one_hot.shape[-1], self.cond_dim, device=one_hot.device)
        return proj(one_hot)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional external context.
        
        Args:
            support_x: (batch, support_size, x_dim)
            support_y: (batch, support_size, y_dim)
            query_x: (batch, query_size, x_dim)
            context: (batch, cond_dim) optional external context (if provided, ignores support_x/y)
        
        Returns:
            pred: (batch, query_size, y_dim)
        """
        if context is None:
            context = self.encode_support(support_x, support_y)
        
        # Use predictor forward to compute predictions
        batch_size, query_size = query_x.shape[:2]
        context_expanded = context.unsqueeze(1).expand(batch_size, query_size, -1)
        combined = torch.cat([query_x, context_expanded], dim=-1)
        pred = self.predictor.pred_head(combined)
        return pred

