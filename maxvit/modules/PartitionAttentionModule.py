import torch
from maxvit.functions import drop_path
from .RelativePositionalAttentionModule import RelativePositionalAttentionModule


class PartitionAttentionModule(torch.nn.Module):
    def __init__(
        self, dimension: int, num_heads: int, P: int, mode: str, drop: float = 0.0
    ):
        super().__init__()
        self.P = P
        self.mode = mode
        self.drop = drop

        self.norm1 = torch.nn.LayerNorm(dimension)
        self.attn = RelativePositionalAttentionModule(dimension, num_heads, P)
        self.norm2 = torch.nn.LayerNorm(dimension)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dimension, dimension * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dimension * 4, dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P, gH, gW = self.P, H // self.P, W // self.P

        if self.mode == "window":
            x = (
                x.view(B, C, gH, P, gW, P)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(-1, P * P, C)
            )
        else:
            x = (
                x.view(B, C, P, gH, P, gW)
                .permute(0, 3, 5, 2, 4, 1)
                .reshape(-1, P * P, C)
            )

        x = x + drop_path(self.attn(self.norm1(x)), self.drop, self.training)
        x = x + drop_path(self.mlp(self.norm2(x)), self.drop, self.training)

        if self.mode == "window":
            x = x.view(B, gH, gW, P, P, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        else:
            x = x.view(B, gH, gW, P, P, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, H, W)
        return x
