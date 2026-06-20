"""DINOv2 ViT encoder used as the MoGe image backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        return self.proj(
            F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(B, N, C)
        )


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, init_values) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        N = self.pos_embed.shape[1] - 1
        if x.shape[1] - 1 == N and w == h:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        M = int(math.sqrt(N))

        patch_pos = (
            F.interpolate(
                patch_pos.reshape(1, M, M, -1).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=False,
                scale_factor=(
                    (h // self.patch_size + 0.1) / M,
                    (w // self.patch_size + 0.1) / M,
                ),
            )
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
        )
        return torch.cat([cls_pos.expand(patch_pos.shape[0], -1, -1), patch_pos], dim=1)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), self.patch_embed(x)], dim=1)
        return x + self.interpolate_pos_encoding(x, h, w)

    def get_intermediate_layers(
        self, x: torch.Tensor, layer_indices: list[int]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        x = self.prepare_tokens(x)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                outputs.append(x)
        outputs = [self.norm(o) for o in outputs]
        return tuple((o[:, 1:], o[:, 0]) for o in outputs)


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        intermediate_layers: list[int],
        dim_out: int,
    ):
        super().__init__()
        assert backbone == "dinov2_vitl14"
        self.intermediate_layers = intermediate_layers
        self.backbone = DinoVisionTransformer(
            embed_dim=1024, depth=24, num_heads=16, init_values=1.0
        )

        self.output_projections = nn.ModuleList(
            [nn.Conv2d(1024, dim_out, 1) for _ in range(len(intermediate_layers))]
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self, images: torch.Tensor, token_rows: int, token_cols: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images_14 = F.interpolate(
            images,
            (token_rows * 14, token_cols * 14),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images_14 = (images_14 - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(images_14, self.intermediate_layers)

        x = torch.stack(
            [
                proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
                for proj, (feat, _) in zip(self.output_projections, features, strict=False)
            ],
            dim=1,
        ).sum(dim=1)

        return x, features[-1][1]
