"""Direct-action WTW+DELTA policy without teacher-student distillation."""

from __future__ import annotations

import torch
from torch import nn

from .wtw_delta_model import WtwDeltaResidualModel


class WtwDeltaDirectActionModel(WtwDeltaResidualModel):
    """Fuse a frozen WTW action prior and DELTA terrain features directly.

    The WTW policy remains a stable prior, but DELTA is connected to the final
    action head instead of being multiplied by a hand-tuned residual scale or a
    post-hoc map-confidence gate.  The fusion layer is initialized as an
    identity pass-through of the WTW action, so the new task starts with the
    pretrained gait and can learn terrain-dependent changes with PPO.
    """

    def __init__(self, *args, **kwargs):
        fusion_hidden_dims = tuple(kwargs.get("residual_hidden_dims", (256, 128)))
        super().__init__(*args, **kwargs)
        # These modules are unused by direct fusion.  Removing them keeps their
        # parameters out of the optimizer and avoids accidentally using residual
        # scaling from the parent model.
        del self.residual_head
        del self.residual_gate

        fusion_input_dim = self.output_dim + self.delta_proprio_dim + 64
        layers: list[nn.Module] = []
        input_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ELU()))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, self.output_dim))
        self.action_fusion = nn.Sequential(*layers)
        fusion_output = self.action_fusion[-1]
        assert isinstance(fusion_output, nn.Linear)
        with torch.no_grad():
            fusion_output.weight.zero_()
            fusion_output.bias.zero_()

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del masks, hidden_state
        base, proprio, terrain = self._observation(obs)
        terrain_latent = self.delta(proprio, terrain)
        prior = self.wtw_policy(base)
        fusion_input = torch.cat((prior, proprio, terrain_latent), dim=-1)
        # There is no hand-tuned DELTA scale.  The zero-initialized nonlinear
        # correction starts as an exact WTW pass-through, then learns the full
        # terrain-dependent action change.  Tanh prevents early PPO updates from
        # immediately overwhelming the pretrained gait.
        delta_action = torch.tanh(self.action_fusion(fusion_input))
        mean = prior + delta_action

        with torch.no_grad():
            self.last_diagnostics = {
                "DELTA/direct_action_rms": mean.square().mean().sqrt().detach(),
                "DELTA/direct_action_delta_rms": delta_action.square()
                .mean()
                .sqrt()
                .detach(),
                "DELTA/direct_action_prior_rms": prior.square().mean().sqrt().detach(),
                "DELTA/direct_action_delta_ratio": (
                    delta_action.square().mean().sqrt()
                    / prior.square().mean().sqrt().clamp_min(1.0e-6)
                ).detach(),
                "DELTA/encoder_output_rms": terrain_latent.square()
                .mean()
                .sqrt()
                .detach(),
            }
            for key, value in self.delta.last_stats.items():
                self.last_diagnostics[key] = value.detach()

        if self.distribution is None:
            return mean
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def debug_visualize(self, env, visualizer) -> None:
        """Reuse DELTA attention drawing for the direct-action model."""
        super().debug_visualize(env, visualizer)
