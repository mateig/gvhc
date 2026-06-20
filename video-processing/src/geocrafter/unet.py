"""Spatiotemporal UNet forward used by the GeoCrafter denoiser."""

import torch
from diffusers import UNetSpatioTemporalConditionModel


class UNet(UNetSpatioTemporalConditionModel):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int | float,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.int64, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        emb = emb + self.add_embedding(time_embeds)

        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.flatten(0, 1).unsqueeze(1)
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        down_block_res_samples = (sample,)
        for block in self.down_blocks:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res = block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res

        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        for block in self.up_blocks:
            res = down_block_res_samples[-len(block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(block.resnets)]
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        return (sample,)
