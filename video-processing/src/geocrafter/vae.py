"""Point-map VAE: encodes priors into the latent and decodes depth/mask/intrinsics."""

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, Encoder
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import SpatioTemporalResBlock
from diffusers.models.unets.unet_3d_blocks import (
    MidBlockTemporalDecoder,
    UpBlockTemporalDecoder,
)


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class PMapTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: tuple[int, ...] = (1, 1, 1),
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], 3, padding=1)
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_out_channels))
        out_ch = reversed_channels[0]
        for i in range(len(block_out_channels)):
            prev_ch = out_ch
            out_ch = reversed_channels[i]
            self.up_blocks.append(
                UpBlockTemporalDecoder(
                    num_layers=layers_per_block + 1,
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    add_upsample=i != len(block_out_channels) - 1,
                )
            )

        half_ch = block_out_channels[0] // 2
        self.out_blocks = nn.ModuleList()
        self.time_conv_outs = nn.ModuleList()
        for ch in out_channels:
            self.out_blocks.append(
                nn.ModuleList(
                    [
                        nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(block_out_channels[0], half_ch, 3, padding=1),
                        SpatioTemporalResBlock(
                            in_channels=half_ch,
                            out_channels=half_ch,
                            temb_channels=None,
                            eps=1e-6,
                            temporal_eps=1e-5,
                            merge_factor=0.0,
                            merge_strategy="learned",
                            switch_spatial_to_temporal_mix=True,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(half_ch, ch, 1),
                    ]
                )
            )
            self.time_conv_outs.append(nn.Conv3d(ch, ch, (3, 1, 1), padding=(1, 0, 0)))

    def forward(
        self,
        sample: torch.Tensor,
        image_only_indicator: torch.Tensor,
        num_frames: int = 1,
    ) -> list[torch.Tensor]:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
        for up_block in self.up_blocks:
            sample = up_block(sample, image_only_indicator=image_only_indicator)

        output = []
        for out_block, time_conv in zip(self.out_blocks, self.time_conv_outs, strict=False):
            x = sample
            for layer in out_block:
                x = (
                    layer(x, None, image_only_indicator)
                    if isinstance(layer, SpatioTemporalResBlock)
                    else layer(x)
                )
            bf, c, h, w = x.shape
            x = x.reshape(bf // num_frames, num_frames, c, h, w).permute(0, 2, 1, 3, 4)
            x = time_conv(x).permute(0, 2, 1, 3, 4).reshape(bf, c, h, w)
            output.append(x)

        return output


class PMapVAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 4,
        enc_down_block_types: tuple[str, ...] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        enc_block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        enc_layers_per_block: int = 2,
        dec_block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        dec_layers_per_block: int = 2,
        out_channels: tuple[int, ...] = (1, 1, 1),
        mid_block_add_attention: bool = True,
        offset_scale_factor: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=enc_down_block_types,
            block_out_channels=enc_block_out_channels,
            layers_per_block=enc_layers_per_block,
            double_z=False,
            mid_block_add_attention=mid_block_add_attention,
        )
        _zero_module(self.encoder.conv_out)
        self.offset_scale_factor = offset_scale_factor
        self.decoder = PMapTemporalDecoder(
            in_channels=latent_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            out_channels=out_channels,
        )

    def encode(
        self, x: torch.Tensor, latent_dist: DiagonalGaussianDistribution
    ) -> DiagonalGaussianDistribution:
        offset = self.encoder(x) * self.offset_scale_factor
        mean, logvar = torch.chunk(latent_dist.parameters, 2, dim=1)
        return DiagonalGaussianDistribution(torch.cat([mean + offset, logvar], dim=1))

    def decode(self, z: torch.Tensor, num_frames: int) -> list[torch.Tensor]:
        batch_size = z.shape[0] // num_frames
        indicator = torch.zeros(batch_size, num_frames, device=z.device)
        return self.decoder(z, num_frames=num_frames, image_only_indicator=indicator)
