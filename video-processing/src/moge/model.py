"""MoGe model: DINOv2 encoder into conv heads for points and scale."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvStack, ScaleMLP
from .geometry import normalized_view_plane_uv
from .vit import DINOv2Encoder


class MoGeModel(nn.Module):
    def __init__(
        self,
        encoder: dict,
        neck: dict,
        points_head: dict,
        scale_head: dict,
        mask_head: dict,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DINOv2Encoder(**encoder)
        self.neck = ConvStack(**neck)
        self.points_head = ConvStack(**points_head)
        self.mask_head = ConvStack(**mask_head)
        self.scale_head = ScaleMLP(**scale_head)

    def forward(
        self, images: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, img_h, img_w = images.shape
        aspect = img_w / img_h

        token_h = round((num_tokens / aspect) ** 0.5)
        token_w = round((num_tokens * aspect) ** 0.5)

        enc_feat, cls_token = self.encoder(images, token_h, token_w)
        features: list[torch.Tensor | None] = [enc_feat, None, None, None, None]

        for level in range(5):
            uv = normalized_view_plane_uv(
                token_w * 2**level,
                token_h * 2**level,
            ).to(images.device, images.dtype)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
            features[level] = (
                torch.cat([features[level], uv], dim=1) if features[level] is not None else uv
            )

        features = self.neck(features)

        points = self.points_head(features)[-1]
        mask = self.mask_head(features)[-1]
        scale = self.scale_head(cls_token)

        points = F.interpolate(points, (img_h, img_w), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, (img_h, img_w), mode="bilinear", align_corners=False)

        points = points.permute(0, 2, 3, 1)
        xy, z = points[..., :2], torch.exp(points[..., 2:3])
        points = torch.cat([xy * z, z], dim=-1)

        mask = mask.squeeze(1).sigmoid()
        scale = scale.squeeze(1).exp()

        return points, scale, mask
