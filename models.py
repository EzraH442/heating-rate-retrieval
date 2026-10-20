import torch
from torch import nn


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_sizes):
        super().__init__()
        layers = []
        d = input_dim
        for size in hidden_dim_sizes:
            layers.extend([nn.Linear(d, size), nn.Tanh()])
            d = size

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self, num_bands=16, d_model=64, n_heads=4, num_levels=37, n_transformer_layers=2
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_bands = num_bands

        self.flux_embed = nn.Embedding(num_bands, d_model)
        self.flux_value = nn.Linear(1, d_model)
        self.level_embed = nn.Embedding(num_levels, d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,
            dim_feedforward=128,
        )
        self.vertical_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # final output projection
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, fluxes):
        B = fluxes.shape[0]
        device = fluxes.device

        # flux embedding, each band is a "token"
        # (16) -> (16, d_model) -> (B, 16, d_model)
        flux_ids = torch.arange(self.num_bands, device=device)
        kv = self.flux_embed(flux_ids).unsqueeze(0) + self.flux_value(
            fluxes.unsqueeze(-1)
        )

        # pressure levels as query vectors
        # (37) -> (37, d_model) -> (B, 37, d_model)
        level_ids = torch.arange(self.num_levels, device=device)
        q = self.level_embed(level_ids).unsqueeze(0).expand(B, -1, -1)

        # x-attn flux profile with pressure level embeddings
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        x = self.norm_cross(q + attn_out)

        # self-attn over pressure levels
        x = self.vertical_transformer(x)  # (B, 37, d_model)

        # final map to final profile values
        predictions = self.out_proj(x)  # Shape: (B, 37, 1)

        return predictions
