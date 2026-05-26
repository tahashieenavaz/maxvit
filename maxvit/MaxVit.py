import torch
from maxvit.modules import MaxVitModule


class MaxVit(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        stem_dimension: int = 64,
        P: int = 7,
        head_dim: int = 32,
        dimensions: list = [64, 128, 256, 512],
        layers: list = [2, 2, 5, 2],
        drop_rate: float = 0.2,
    ):
        super().__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels, stem_dimension, 3, stride=2, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(stem_dimension),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                stem_dimension, stem_dimension, 3, stride=1, padding=1, bias=True
            ),
        )

        self.blocks = torch.nn.ModuleList()
        dp_rates = torch.linspace(0, drop_rate, sum(layers)).tolist()

        in_dim, idx = stem_dimension, 0
        for out_dim, num_layers in zip(dimensions, layers):
            for i in range(num_layers):
                stride = 2 if i == 0 else 1
                self.blocks.append(
                    MaxVitModule(
                        in_dim, out_dim, P, out_dim // head_dim, stride, dp_rates[idx]
                    )
                )
                in_dim = out_dim
                idx += 1

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.LayerNorm(dimensions[-1]),
            torch.nn.Linear(dimensions[-1], dimensions[-1]),
            torch.nn.Tanh(),
            torch.nn.Linear(dimensions[-1], num_classes, bias=False),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
