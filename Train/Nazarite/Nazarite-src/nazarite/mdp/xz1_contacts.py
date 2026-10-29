"""Identify actual foot/pit-floor contacts without copying contacts to CPU."""

# Warp kernel annotations construct array descriptors at runtime.
# pyright: reportInvalidTypeForm=false

import warp as wp

from nazarite.config.robot_config.xz1_cfg import XZ1_FOOT_GEOMS
from nazarite.config.train_config.XZ1_env_cfgs.him_obstacles import GAP_BOTTOM_PREFIX


@wp.kernel
def _foot_bottom_contacts(
    nacon: wp.array(dtype=int),
    worlds: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    distances: wp.array(dtype=float),
    kinds: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    index = wp.tid()
    if index < nacon[0]:
        world = worlds[index]
        a, b = pairs[index][0], pairs[index][1]
        if (
            world >= 0
            and world < hits.shape[0]
            and a >= 0
            and b >= 0
            and a < kinds.shape[0]
            and b < kinds.shape[0]
            and distances[index] <= 0.0
        ):
            if (kinds[a] == 1 and kinds[b] == 2) or (kinds[a] == 2 and kinds[b] == 1):
                wp.atomic_max(hits, world, 1)


class GapBottomContact:
    """Match named pit floors to feet, ignoring walls and unused contact slots.

    Like the existing foot force checks, this samples the last physics substep
    of the control step. No terrain-row-specific geometry IDs are hardcoded.
    """

    def __init__(self, model, num_envs, device):
        foot_names = {f"robot/{name}" for name in XZ1_FOOT_GEOMS}
        kinds = [0] * model.ngeom
        for index in range(model.ngeom):
            name = model.geom(index).name
            if name in foot_names:
                kinds[index] = 1
            elif name.startswith(GAP_BOTTOM_PREFIX):
                kinds[index] = 2
        if 1 not in kinds or 2 not in kinds:
            raise ValueError("Gap contact check requires named feet and pit bottoms")
        self.kinds = wp.array(kinds, dtype=int, device=str(device))
        self.hits = wp.zeros(num_envs, dtype=int, device=str(device))

    def __call__(self, data):
        self.hits.zero_()
        contact = data.contact
        wp.launch(
            _foot_bottom_contacts,
            dim=contact.worldid.shape[0],
            inputs=[
                data.nacon,
                contact.worldid,
                contact.geom,
                contact.dist,
                self.kinds,
                self.hits,
            ],
            device=self.hits.device,
        )
        return wp.to_torch(self.hits).bool()
