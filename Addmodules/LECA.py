import torch.nn as nn

__all__ = ['LECA']


class LECA(nn.Module):


    def __init__(self, in_channels, reduction=16, strip_kernel_size=5):
        super().__init__()
        self.strip_pool_h = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, strip_kernel_size),
            padding=(0, strip_kernel_size // 2),
            groups=in_channels,
            bias=True,
        )
        self.strip_pool_w = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(strip_kernel_size, 1),
            padding=(strip_kernel_size // 2, 0),
            groups=in_channels,
            bias=True,
        )
        self.spatial_act = nn.Sigmoid()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_interact = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.channel_act = nn.Sigmoid()

    def forward(self, x):
        spatial_h = self.strip_pool_h(x)
        spatial_w = self.strip_pool_w(x)
        spatial_weight = self.spatial_act(spatial_h + spatial_w)
        out_spatial = x * spatial_weight

        channel_ctx = self.global_pool(out_spatial)
        channel_ctx = channel_ctx.squeeze(-1).transpose(-1, -2)
        channel_weight = self.channel_act(self.channel_interact(channel_ctx))
        channel_weight = channel_weight.transpose(-1, -2).unsqueeze(-1)

        return out_spatial * channel_weight
