"""GeoCrafter video-diffusion pipeline: video + priors -> depth, intrinsics."""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)
from transformers import CLIPVisionModelWithProjection

from .geometry import normalized_meshgrid
from .unet import UNet
from .vae import PMapVAE

VAE_SCALE_FACTOR = 8


class GeoCrafterPipeline:
    def __init__(
        self,
        image_encoder: CLIPVisionModelWithProjection,
        vae: AutoencoderKLTemporalDecoder,
        unet: UNet,
        pmap_vae: PMapVAE,
        scheduler: EulerDiscreteScheduler,
    ):
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.pmap_vae = pmap_vae
        self.scheduler = scheduler

    def encode_clip(self, video: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = _resize_with_antialiasing(video, (224, 224))
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return torch.cat(
            [
                self.image_encoder(x[i : i + batch_size]).image_embeds
                for i in range(0, x.shape[0], batch_size)
            ]
        )

    def encode_vae(self, video: torch.Tensor, batch_size: int) -> torch.Tensor:
        return torch.cat(
            [
                self.vae.encode(video[i : i + batch_size]).latent_dist.mode()
                for i in range(0, video.shape[0], batch_size)
            ]
        )

    def encode_priors(
        self,
        disparity: torch.Tensor,
        point_map: torch.Tensor,
        intrinsic_map: torch.Tensor,
        valid_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        n = point_map.shape[0]
        pseudo_image = disparity[:, None].repeat(1, 3, 1, 1)
        intr_norm = torch.norm(intrinsic_map[:, 2:4], p=2, dim=1)

        latents = []
        for i in range(0, n, batch_size):
            s = slice(i, i + batch_size)
            dist = self.vae.encode(pseudo_image[s]).latent_dist
            dist = self.pmap_vae.encode(
                torch.cat(
                    [
                        intr_norm[s, None],
                        point_map[s, 2:3],
                        disparity[s, None],
                        valid_mask[s, None],
                    ],
                    dim=1,
                ),
                dist,
            )
            latents.append(dist.mode())
        return torch.cat(latents) * self.vae.config.scaling_factor

    def decode_latents(
        self, latents: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = latents.shape[0]
        decoded = [
            self.pmap_vae.decode(latents[i : i + batch_size], num_frames=min(batch_size, n - i))
            for i in range(0, n, batch_size)
        ]
        return (
            torch.cat([d[0] for d in decoded]),
            torch.cat([d[1] for d in decoded]),
            torch.cat([d[2] for d in decoded]),
        )

    def denoise(
        self,
        video_latents: torch.Tensor,
        prior_latents: torch.Tensor,
        video_embeddings: torch.Tensor,
        num_frames: int,
        latent_h: int,
        latent_w: int,
        num_inference_steps: int,
        window_size: int,
        overlap: int,
    ) -> torch.Tensor:
        scheduler = self.scheduler
        device = video_latents.device

        if num_frames <= window_size:
            window_size, overlap = num_frames, 0
        stride = window_size - overlap

        scheduler.set_timesteps(num_inference_steps, device=device)
        added_time_ids = torch.tensor([[7, 127, 0.02]], device=device)

        latents_init = (
            torch.randn(1, window_size, 4, latent_h, latent_w, device=device)
            * scheduler.init_noise_sigma
        )
        latents_all = None

        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device).view(1, overlap, 1, 1, 1)

        idx_start = 0
        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            scheduler.set_timesteps(num_inference_steps, device=device)

            cur_latents = latents_init[:, : idx_end - idx_start].clone()
            latents_init = torch.cat([latents_init[:, -overlap:], latents_init[:, :stride]], dim=1)

            vid_lat = video_latents[:, idx_start:idx_end]
            pri_lat = prior_latents[:, idx_start:idx_end]
            vid_emb = video_embeddings[:, idx_start:idx_end]

            for i, t in enumerate(scheduler.timesteps):
                if latents_all is not None and i == 0:
                    cur_latents[:, :overlap] = (
                        latents_all[:, -overlap:]
                        + cur_latents[:, :overlap]
                        / scheduler.init_noise_sigma
                        * scheduler.sigmas[i]
                    )

                model_input = scheduler.scale_model_input(cur_latents, t)
                model_input = torch.cat([model_input, vid_lat, pri_lat], dim=2)

                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=vid_emb,
                    added_time_ids=added_time_ids,
                )[0]

                cur_latents = scheduler.step(noise_pred, t, cur_latents).prev_sample

            if latents_all is None:
                latents_all = cur_latents.clone()
            else:
                if overlap > 0:
                    latents_all[:, -overlap:] = cur_latents[:, :overlap] * weights + latents_all[
                        :, -overlap:
                    ] * (1 - weights)
                latents_all = torch.cat([latents_all, cur_latents[:, overlap:]], dim=1)

            idx_start += stride

        return latents_all.squeeze(0) / self.vae.config.scaling_factor

    def __call__(
        self,
        video: torch.Tensor,
        disparity: torch.Tensor,
        point_map: torch.Tensor,
        intrinsic_map: torch.Tensor,
        valid_mask: torch.Tensor,
        process_resolution: int,
        num_inference_steps: int,
        window_size: int,
        overlap: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, C, H, W = video.shape

        w = round(process_resolution / 64) * 64
        h = round(H * process_resolution / W / 64) * 64
        size = (h, w)

        video = F.interpolate(
            video, size, mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        disparity = F.interpolate(
            disparity[:, None], size, mode="bilinear", align_corners=False
        ).squeeze(1)
        point_map = F.interpolate(point_map, size, mode="bilinear", align_corners=False)
        intrinsic_map = F.interpolate(intrinsic_map, size, mode="bilinear", align_corners=False)
        valid_mask = F.interpolate(
            valid_mask[:, None], size, mode="bilinear", align_corners=False
        ).squeeze(1)

        video = video * 2.0 - 1.0
        video_embeddings = self.encode_clip(video, batch_size).unsqueeze(0)
        video_latents = self.encode_vae(video, batch_size).unsqueeze(0)
        prior_latents = self.encode_priors(
            disparity, point_map, intrinsic_map, valid_mask, batch_size
        ).unsqueeze(0)

        latents = self.denoise(
            video_latents,
            prior_latents,
            video_embeddings,
            num_frames=T,
            latent_h=h // VAE_SCALE_FACTOR,
            latent_w=w // VAE_SCALE_FACTOR,
            num_inference_steps=num_inference_steps,
            window_size=window_size,
            overlap=overlap,
        )

        return self.decode_latents(latents, batch_size)


def reconstruct_points(
    intr: torch.Tensor,
    log_depth: torch.Tensor,
    vmask: torch.Tensor,
    depth_norm: float,
    out_size: tuple[int, int],
) -> torch.Tensor:
    H, W = out_size
    log_depth = F.interpolate(
        log_depth.clamp_max(10).exp(),
        (H, W),
        mode="bilinear",
        align_corners=False,
    ).log()
    intr = F.interpolate(intr, (H, W), mode="bilinear", align_corners=False)
    vmask = F.interpolate(vmask, (H, W), mode="bilinear", align_corners=False)
    valid = vmask.squeeze(1) > 0

    diag = (H**2 + W**2) ** 0.5
    intr = torch.cat([intr * W / diag, intr * H / diag], dim=1)

    valid_f = valid.float().unsqueeze(1)
    denom = valid_f.mean().clamp_min(1e-4)
    nf = torch.stack(
        [
            (intr[:, 0:1] * valid_f).mean() / denom,
            (intr[:, 1:2] * valid_f).mean() / denom,
        ]
    ).view(1, 2, 1, 1)

    grid = normalized_meshgrid(H, W, intr.device).permute(2, 0, 1).unsqueeze(0)
    z = log_depth.squeeze(1).clamp_max(10).exp() * depth_norm
    xy = nf * grid * z.unsqueeze(1)
    points = torch.cat([xy, z.unsqueeze(1)], dim=1).permute(0, 2, 3, 1)

    nan = torch.tensor(float("nan"), device=points.device, dtype=points.dtype)
    return torch.where(valid.unsqueeze(-1), points, nan)
