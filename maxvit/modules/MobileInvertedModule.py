import torch
from maxvit.functions import drop_path
from .SqueezeExciteModule import SqueezeExciteModule


class MobileInvertedModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        stride: int = 1,
        expansion: float = 4.0,
        squeeze_ratio: float = 0.25,
        path_drop: float = 0.0,
    ):
        super().__init__()
        self.path_drop = path_drop
        mid_dim = int(output_dimension * expansion)
        sqz_dim = int(output_dimension * squeeze_ratio)
        self.projection = (
            torch.nn.Sequential(
                (
                    torch.nn.AvgPool2d(3, stride, 1)
                    if stride == 2
                    else torch.nn.Identity()
                ),
                torch.nn.Conv2d(input_dimension, output_dimension, 1),
            )
            if stride != 1 or input_dimension != output_dimension
            else torch.nn.Identity()
        )

        self.block = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_dimension),
            torch.nn.Conv2d(input_dimension, mid_dim, 1, bias=False),
            torch.nn.BatchNorm2d(mid_dim),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid_dim, mid_dim, 3, stride, 1, groups=mid_dim, bias=False),
            torch.nn.BatchNorm2d(mid_dim),
            torch.nn.GELU(),
            SqueezeExciteModule(mid_dim, sqz_dim),
            torch.nn.Conv2d(mid_dim, output_dimension, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x) + drop_path(
            self.block(x), self.path_drop, self.training
        )
