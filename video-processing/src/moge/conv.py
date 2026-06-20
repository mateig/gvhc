"""Conv building blocks for the MoGe neck and heads."""

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(channels, hidden_channels, 3, padding=1, padding_mode="replicate"),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, padding_mode="replicate"),
        )
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip(x)


def _make_resampler(in_ch: int, out_ch: int, scale_factor: int, kind: str) -> nn.Sequential:
    if kind == "conv_transpose":
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, scale_factor, stride=scale_factor),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
        )
    if kind == "bilinear":
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate"),
        )
    raise ValueError(f"Unsupported resampler type: {kind}")


class ConvStack(nn.Module):
    def __init__(
        self,
        dim_in: list[int | None],
        dim_res_blocks: list[int],
        dim_out: list[int | None] | None,
        resamplers: list[str],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        n = len(dim_res_blocks)
        if dim_out is None:
            dim_out = [None] * n
        if num_res_blocks is None:
            num_res_blocks = [1] * n

        self.input_blocks = nn.ModuleList(
            [
                nn.Conv2d(d_in, d_res, 1) if d_in is not None else nn.Identity()
                for d_in, d_res in zip(dim_in, dim_res_blocks, strict=False)
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                _make_resampler(d_prev, d_next, 2, rtype)
                for d_prev, d_next, rtype in zip(
                    dim_res_blocks[:-1], dim_res_blocks[1:], resamplers, strict=False
                )
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        ResidualConvBlock(d, dim_times_res_block_hidden * d)
                        for _ in range(num_res_blocks[i])
                    )
                )
                for i, d in enumerate(dim_res_blocks)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                nn.Conv2d(d_res, d_out, 1) if d_out is not None else nn.Identity()
                for d_out, d_res in zip(dim_out, dim_res_blocks, strict=False)
            ]
        )

    def forward(self, in_features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        x = None
        for i in range(len(self.res_blocks)):
            feat = self.input_blocks[i](in_features[i])
            x = feat if i == 0 else x + feat
            x = self.res_blocks[i](x)
            out.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out


class ScaleMLP(nn.Sequential):
    def __init__(self, dims: list[int]):
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-2], dims[1:-1], strict=False):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        super().__init__(*layers)
