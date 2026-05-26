import torch
from torch.nn.functional import silu


class SqueezeExciteModule(torch.nn.Module):
    def __init__(self, in_dimension: int, squeeze_dimension: int):
        super().__init__()
        self.alpha = torch.nn.Conv2d(in_dimension, squeeze_dimension, 1)
        self.beta = torch.nn.Conv2d(squeeze_dimension, in_dimension, 1)

    def get_scale(self, *, mean: torch.Tensor) -> torch.Tensor:
        scale = self.alpha(mean)
        scale = silu(scale)
        scale = self.beta(scale)
        return torch.sigmoid(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        scale = self.get_scale(mean)
        return x * scale
