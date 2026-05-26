import torch


class RelativePositionalAttentionModule(torch.nn.Module):
    def __init__(self, dimension: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dimension // num_heads) ** -0.5
        self.qkv = torch.nn.Linear(dimension, dimension * 3)
        self.projection = torch.nn.Linear(dimension, dimension)
        self.bias_table = torch.nn.Parameter(
            torch.empty((2 * window_size - 1) ** 2, num_heads)
        )
        torch.nn.init.trunc_normal_(self.bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid([torch.arange(window_size)] * 2, indexing="ij")
        )
        coords = coords.flatten(1)
        rel_coords = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        rel_coords += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_idx", rel_coords.sum(-1).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.bias_table[self.rel_idx].view(N, N, -1).permute(2, 0, 1)
        attention = (attention + bias).softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(B, N, C)
        return self.projection(output)
