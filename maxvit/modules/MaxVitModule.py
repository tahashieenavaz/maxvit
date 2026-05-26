import torch
from .MobileInvertedModule import MobileInvertedModule
from .PartitionAttentionModule import PartitionAttentionModule


class MaxVitModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        P: int,
        num_heads: int,
        stride: int = 1,
        drop: float = 0.0,
    ):
        super().__init__()
        self.mobile_inverted_convolutional = MobileInvertedModule(
            input_dimension, output_dimension, stride, drop=drop
        )
        self.window_attention = PartitionAttentionModule(
            output_dimension, num_heads, P, "window", drop
        )
        self.grid_attention = PartitionAttentionModule(
            output_dimension, num_heads, P, "grid", drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grid_attention(
            self.window_attention(self.mobile_inverted_convolutional(x))
        )
