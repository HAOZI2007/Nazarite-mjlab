"""Observation terms used by the DELTA terrain policy."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def delta_privileged_terrain_map(
    env,
    sensor_name: str = "delta_privileged_scan",
    map_height: int = 16,
    map_width: int = 26,
    scan_width: int = 51,
    x_range_m: tuple[float, float] = (0.0, 2.5),
    y_range_m: tuple[float, float] = (-0.8, 0.8),
    z_scale_m: float = 0.8,
    map_channels: int = 5,
    coordinate_normalization: str = "bev",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Build a privileged local map from downward raycasts.

    ``map_channels=3`` returns the paper-compatible ``x, y, z`` map.  The
    legacy five-channel support/confidence representation remains available for
    older tasks.
    """
    sensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    hit_w = sensor.data.hit_pos_w
    distances = sensor.data.distances
    batch = hit_w.shape[0]
    if hit_w.ndim != 3 or hit_w.shape[-1] != 3:
        raise ValueError(f"{sensor_name} returned invalid hit shape {tuple(hit_w.shape)}")

    # Project ray hits into the current body frame before rasterizing. This avoids
    # assuming a particular source grid (the old implementation hard-coded a
    # 5 m/0.1 m scan and then selected columns, which was wrong for 1.5 m BEVs).
    hit_w = hit_w.reshape(batch, -1, 3)
    distances = distances.reshape(batch, -1)
    base_pos = asset.data.root_link_pos_w[:, None, :]
    base_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, hit_w.shape[1], -1)
    hit_b = quat_apply_inverse(base_quat, hit_w - base_pos)
    valid = torch.isfinite(hit_b).all(dim=-1) & (distances >= 0.0)
    x0, x1 = map(float, x_range_m)
    y0, y1 = map(float, y_range_m)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("x_range_m and y_range_m must be increasing")
    cells = map_height * map_width
    col = torch.floor((hit_b[..., 0] - x0) / (x1 - x0) * map_width).long()
    row = torch.floor((y1 - hit_b[..., 1]) / (y1 - y0) * map_height).long()
    inside = valid & (col >= 0) & (col < map_width) & (row >= 0) & (row < map_height)
    cell = row.clamp(0, map_height - 1) * map_width + col.clamp(0, map_width - 1)
    offset = torch.arange(batch, device=hit_w.device).view(batch, 1) * cells
    flat_cell = (cell + offset).reshape(-1)
    flat_valid = inside.reshape(-1)
    if map_channels not in (3, 5):
        raise ValueError("Privileged DELTA maps support 3 or 5 channels")
    if coordinate_normalization not in ("bev", "paper"):
        raise ValueError("coordinate_normalization must be 'bev' or 'paper'")
    flat_z = hit_b[..., 2].clamp(-0.8, 0.8).reshape(-1)
    support_flat = torch.zeros(batch * cells, device=hit_w.device, dtype=hit_w.dtype)
    support_flat.scatter_reduce_(0, flat_cell, flat_valid.to(hit_w.dtype), reduce="amax", include_self=True)
    # Do not initialize the height accumulator to zero: in the base frame a
    # normal ground hit is negative z, so a zero-initialized ``amax`` silently
    # erases the entire floor and all downward terrain.  Use -inf while
    # rasterizing and fill genuinely empty cells only after the reduction.
    z_flat = torch.full_like(support_flat, -torch.inf)
    z_values = torch.where(flat_valid, flat_z, torch.full_like(flat_z, -torch.inf))
    z_flat.scatter_reduce_(0, flat_cell, z_values, reduce="amax", include_self=True)
    z_flat = torch.where(torch.isfinite(z_flat), z_flat, torch.zeros_like(z_flat))
    support = support_flat.reshape(batch, map_height, map_width)
    z = z_flat.reshape(batch, map_height, map_width)
    x_centers = torch.linspace(
        x0 + 0.5 * (x1 - x0) / map_width,
        x1 - 0.5 * (x1 - x0) / map_width,
        map_width,
        device=z.device,
        dtype=z.dtype,
    )
    y_centers = torch.linspace(
        y1 - 0.5 * (y1 - y0) / map_height,
        y0 + 0.5 * (y1 - y0) / map_height,
        map_height,
        device=z.device,
        dtype=z.dtype,
    )
    if coordinate_normalization == "paper":
        x = x_centers / 1.25
        y = y_centers / 0.75
        z = z / 0.6
    else:
        x = (x_centers - x0) / (x1 - x0) * 2.0 - 1.0
        y = (y_centers - y0) / (y1 - y0) * 2.0 - 1.0
        z = z / max(float(z_scale_m), 1.0e-6)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).expand(batch, -1, -1, -1)
    output = torch.cat((xy, z[..., None]), dim=-1)
    if map_channels == 5:
        confidence = support.clone()
        output = torch.cat((output, support[..., None], confidence[..., None]), dim=-1)
    output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    camera_map = getattr(env, "_delta_map_cache", None)
    # Support/confidence diagnostics only exist for the legacy five-channel
    # camera map.  In the paper-compatible three-channel x/y/z path there are
    # no support/confidence planes to compare, so skip this block entirely.
    if (
        map_channels >= 5
        and isinstance(camera_map, torch.Tensor)
        and camera_map.shape == output.shape
        and camera_map.shape[-1] >= 5
        and hasattr(env, "extras")
        and "log" in env.extras
    ):
        gt_support = output[..., 3] > 0.5
        camera_support = camera_map[..., 3] > 0.5
        camera_observed = camera_map[..., 4] > 0.25
        intersection = (gt_support & camera_support).sum(dim=(-1, -2)).float()
        union = (gt_support | camera_support).sum(dim=(-1, -2)).float()
        true_support = gt_support.sum(dim=(-1, -2)).float()
        predicted_support = camera_support.sum(dim=(-1, -2)).float()
        gt_gap = ~gt_support
        predicted_gap = ~camera_support
        visible_gt_gap = gt_gap & camera_observed
        height_mask = gt_support & camera_support & camera_observed
        height_error = torch.abs(camera_map[..., 2] - output[..., 2]) * 0.8
        height_mae = (height_error * height_mask).sum(dim=(-1, -2)) / (
            height_mask.sum(dim=(-1, -2)).clamp_min(1)
        )
        gap_recall = (visible_gt_gap & predicted_gap).sum(dim=(-1, -2)).float() / (
            visible_gt_gap.sum(dim=(-1, -2)).clamp_min(1)
        )
        env.extras["log"]["DELTA/map_height_mae_m"] = height_mae.mean()
        env.extras["log"]["DELTA/map_support_iou"] = (
            intersection / union.clamp_min(1.0)
        ).mean()
        env.extras["log"]["DELTA/map_support_precision"] = (
            intersection / predicted_support.clamp_min(1.0)
        ).mean()
        env.extras["log"]["DELTA/map_support_recall"] = (
            intersection / true_support.clamp_min(1.0)
        ).mean()
        env.extras["log"]["DELTA/map_gap_recall"] = gap_recall.mean()
    # Reward computation on the next control step uses this simulator-ground-
    # truth cache.  The deployed actor never receives it in the camera task.
    env._delta_privileged_map_cache = output.detach()
    return output.reshape(batch, map_height * map_width * map_channels)


def delta_elevation_map(
    env,
    sensor_name: str = "terrain_scan",
    height: int = 26,
    width: int = 16,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return a flattened robot-frame DELTA terrain map.

    The raycast sensor stores hit points in world coordinates.  We transform them
    into the current base frame so the policy sees a translation/rotation-local
    map.  Misses are represented by zero and can be randomized during training.

    ObservationManager concatenates all terms along the feature dimension, so this
    term must expose ``[B, H*W*3]`` rather than ``[B, H, W, 3]``.  The DELTA model
    reshapes it back to ``[B, H, W, 3]`` before passing it to the encoder.
    """
    sensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    hit_w = sensor.data.hit_pos_w
    distances = sensor.data.distances
    b = hit_w.shape[0]
    if hit_w.numel() != b * height * width * 3:
        raise ValueError(
            f"{sensor_name} returned {hit_w.shape}; expected [B,{height * width},3]"
        )
    hit_w = hit_w.reshape(b, height * width, 3)
    distances = distances.reshape(b, height * width)
    base_pos = asset.data.root_link_pos_w[:, None, :]
    base_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, height * width, -1)
    hit_b = quat_apply_inverse(base_quat, hit_w - base_pos)
    valid = distances >= 0
    hit_b = torch.where(valid[..., None], hit_b, torch.zeros_like(hit_b))
    # Match the paper's coordinate normalization before the model sees the map.
    hit_b = hit_b.reshape(b, height, width, 3)
    hit_b[..., 0] = hit_b[..., 0] / 1.25
    hit_b[..., 1] = hit_b[..., 1] / 0.75
    hit_b[..., 2] = hit_b[..., 2].clamp(-0.8, 0.8) / 0.6
    hit_b = torch.nan_to_num(hit_b, nan=0.0, posinf=0.0, neginf=0.0)
    return hit_b.reshape(b, height * width * 3)


def delta_depth_image(
    env,
    sensor_name: str = "delta_depth_camera",
    map_height: int = 16,
    map_width: int = 26,
    max_depth: float = 5.0,
    horizontal_fov_deg: float | None = None,
    vertical_fov_deg: float | None = None,
    camera_pos_b: tuple[float, float, float] = (0.30, 0.0, 0.12),
    camera_pitch_deg: float = 20.0,
    camera_roll_deg: float = 0.0,
    fx_px: float | None = None,
    fy_px: float | None = None,
    cx_px: float | None = None,
    cy_px: float | None = None,
    project_to_bev: bool = False,
    x_range_m: tuple[float, float] = (0.0, 3.0),
    y_range_m: tuple[float, float] = (-1.0, 1.0),
    z_scale_m: float = 0.8,
    map_channels: int = 3,
    fill_kernel_size: int = 5,
    coordinate_normalization: str = "bev",
    x_scale_m: float = 1.25,
    y_scale_m: float = 0.75,
    z_normalization_m: float = 0.6,
    ablation_mode: str = "none",
) -> torch.Tensor:
    """Convert one head-mounted depth image to a DELTA terrain map.

    With ``project_to_bev=True`` the perspective image is splatted into a
    regular robot-frame x-y grid.  This is the representation expected by the
    deformable DELTA sampler.  The optional fourth channel is a valid-obstacle
    mask; it is deliberately opt-in so old three-channel DELTA checkpoints keep
    their observation contract.
    """
    sensor: CameraSensor = env.scene[sensor_name]
    depth_data = sensor.data.depth
    if depth_data is None:
        raise RuntimeError(f"Camera '{sensor_name}' has no depth output")
    if depth_data.ndim != 4 or depth_data.shape[-1] != 1:
        raise ValueError(f"Expected depth [B,H,W,1], got {tuple(depth_data.shape)}")

    depth = depth_data[..., 0].float()
    batch, image_height, image_width = depth.shape
    device, dtype = depth.device, depth.dtype
    explicit_intrinsics = (fx_px, fy_px, cx_px, cy_px)
    if any(value is not None for value in explicit_intrinsics):
        if not all(value is not None for value in explicit_intrinsics):
            raise ValueError("fx_px, fy_px, cx_px, and cy_px must be provided together")
        assert fx_px is not None and fy_px is not None
        assert cx_px is not None and cy_px is not None
        if fx_px <= 0.0 or fy_px <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        fx = depth.new_tensor(fx_px)
        fy = depth.new_tensor(fy_px)
        cx = depth.new_tensor(cx_px)
        cy = depth.new_tensor(cy_px)
    elif sensor.cfg.focal_length_px is not None:
        assert sensor.cfg.principal_point_px is not None
        scale_x = image_width / sensor.cfg.width
        scale_y = image_height / sensor.cfg.height
        fx = depth.new_tensor(sensor.cfg.focal_length_px[0] * scale_x)
        fy = depth.new_tensor(sensor.cfg.focal_length_px[1] * scale_y)
        cx = depth.new_tensor(sensor.cfg.principal_point_px[0] * scale_x)
        cy = depth.new_tensor(sensor.cfg.principal_point_px[1] * scale_y)
    else:
        # Backward-compatible fallback for cameras configured only by FOV.
        horizontal_fov_deg = 87.0 if horizontal_fov_deg is None else horizontal_fov_deg
        vertical_fov_deg = 58.0 if vertical_fov_deg is None else vertical_fov_deg
        if not 0.0 < horizontal_fov_deg < 180.0:
            raise ValueError("horizontal_fov_deg must be between 0 and 180")
        if not 0.0 < vertical_fov_deg < 180.0:
            raise ValueError("vertical_fov_deg must be between 0 and 180")
        half_pi = torch.pi / 360.0
        fx = image_width / (
            2.0 * torch.tan(depth.new_tensor(horizontal_fov_deg * half_pi))
        )
        fy = image_height / (
            2.0 * torch.tan(depth.new_tensor(vertical_fov_deg * half_pi))
        )
        cx = depth.new_tensor((image_width - 1) * 0.5)
        cy = depth.new_tensor((image_height - 1) * 0.5)
    u = torch.arange(image_width, device=device, dtype=dtype)
    v = torch.arange(image_height, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    valid = torch.isfinite(depth) & (depth > 1.0e-3) & (depth < max_depth)
    z = depth.clamp(min=0.0, max=max_depth)
    x_right = (uu[None] - cx) * z / fx
    y_down = (vv[None] - cy) * z / fy

    # Match sim2sim exactly: construct the camera-to-body rotation once and
    # rotate camera-frame points with it. This keeps horizontal and portrait
    # streams, including optical-axis roll, under one coordinate convention.
    # MuJoCo camera coordinates are +X right, +Y up, -Z forward.
    pitch = depth.new_tensor(camera_pitch_deg * torch.pi / 180.0)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    rotation_y = torch.stack(
        (
            torch.stack((cp, cp * 0.0, sp)),
            torch.stack((cp * 0.0, cp * 0.0 + 1.0, cp * 0.0)),
            torch.stack((-sp, cp * 0.0, cp)),
        )
    )
    # At zero pitch: camera +X -> body -Y, camera +Y -> body +Z,
    # camera +Z -> body -X, which maps camera -Z to body +X.
    base_rotation = depth.new_tensor(
        ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    pitched_rotation = rotation_y @ base_rotation
    optical_axis = -pitched_rotation[:, 2]
    roll = depth.new_tensor(camera_roll_deg * torch.pi / 180.0)
    cr, sr = torch.cos(roll), torch.sin(roll)
    skew = torch.stack(
        (
            torch.stack((optical_axis[0] * 0.0, -optical_axis[2], optical_axis[1])),
            torch.stack((optical_axis[2], optical_axis[0] * 0.0, -optical_axis[0])),
            torch.stack((-optical_axis[1], optical_axis[0], optical_axis[0] * 0.0)),
        )
    )
    roll_rotation = (
        torch.eye(3, device=device, dtype=dtype) * cr
        + (1.0 - cr) * torch.outer(optical_axis, optical_axis)
        + sr * skew
    )
    camera_points = torch.stack((x_right, -y_down, -z), dim=-1)
    body_points = torch.einsum(
        "bhwj,ij->bhwi", camera_points, roll_rotation @ pitched_rotation
    )
    points = body_points + depth.new_tensor(camera_pos_b)
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    if project_to_bev:
        if map_channels not in (3, 4, 5):
            raise ValueError("BEV DELTA maps support 3, 4, or 5 channels")
        if coordinate_normalization not in ("bev", "paper"):
            raise ValueError("coordinate_normalization must be 'bev' or 'paper'")
        if x_scale_m <= 0.0 or y_scale_m <= 0.0 or z_normalization_m <= 0.0:
            raise ValueError("Map coordinate normalization scales must be positive")
        x0, x1 = (float(x_range_m[0]), float(x_range_m[1]))
        y0, y1 = (float(y_range_m[0]), float(y_range_m[1]))
        if not x1 > x0 or not y1 > y0:
            raise ValueError("x_range_m and y_range_m must be increasing")
        if z_scale_m <= 0.0:
            raise ValueError("z_scale_m must be positive")
        if fill_kernel_size < 1 or fill_kernel_size % 2 == 0:
            raise ValueError("fill_kernel_size must be a positive odd integer")

        # Rows run from +y to -y so grid_sample's row orientation matches the
        # robot-frame convention used by the attention sampler.
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        col = torch.floor((x - x0) / (x1 - x0) * map_width).long()
        row = torch.floor((y1 - y) / (y1 - y0) * map_height).long()
        inside = (
            valid
            & (col >= 0)
            & (col < map_width)
            & (row >= 0)
            & (row < map_height)
            & torch.isfinite(z)
        )

        cells = map_height * map_width
        cell = row.clamp(0, map_height - 1) * map_width + col.clamp(0, map_width - 1)
        batch_offset = torch.arange(batch, device=device).view(batch, 1, 1) * cells
        flat_cell = (cell + batch_offset).reshape(-1)
        flat_valid = inside.reshape(-1)
        flat_z = z.clamp(-0.8, 0.8).reshape(-1)

        # Use the highest observed surface in a cell.  This preserves obstacle
        # tops when a pixel also sees the ground behind the obstacle.
        bev_z = torch.zeros((batch * cells,), device=device, dtype=dtype)
        valid_values = torch.where(flat_valid, flat_z, torch.zeros_like(flat_z))
        bev_z.scatter_reduce_(
            0, flat_cell, valid_values, reduce="amax", include_self=True
        )
        bev_valid = torch.zeros((batch * cells,), device=device, dtype=dtype)
        bev_valid.scatter_add_(0, flat_cell, flat_valid.to(dtype))
        bev_valid = (bev_valid > 0.0).to(dtype)
        bev_z = bev_z.reshape(batch, map_height, map_width)
        bev_valid = bev_valid.reshape(batch, map_height, map_width)
        raw_bev_valid = bev_valid.clone()

        # The paper exposes only x/y/z to DELTA and does not use a learned
        # inpainting stage.  Keep raw splatted heights for the three-channel
        # task; legacy four/five-channel maps retain the old optional fill.
        if map_channels != 3:
            valid_f = bev_valid[:, None]
            z_f = bev_z[:, None] * valid_f
            local_kernel = int(fill_kernel_size)
            local_sum = torch.nn.functional.avg_pool2d(
                z_f, local_kernel, stride=1, padding=local_kernel // 2
            ) * (local_kernel**2)
            local_count = torch.nn.functional.avg_pool2d(
                valid_f, local_kernel, stride=1, padding=local_kernel // 2
            ) * (local_kernel**2)
            local_z = local_sum / local_count.clamp_min(1.0)
            hole = bev_valid <= 0.0
            bev_z = torch.where(hole, local_z[:, 0], bev_z)
            bev_valid = torch.maximum(
                bev_valid,
                (local_count[:, 0] > 0.0).to(dtype) * 0.5,
            )

        x_centers = torch.linspace(
            x0 + 0.5 * (x1 - x0) / map_width,
            x1 - 0.5 * (x1 - x0) / map_width,
            map_width,
            device=device,
            dtype=dtype,
        )
        y_centers = torch.linspace(
            y1 - 0.5 * (y1 - y0) / map_height,
            y0 + 0.5 * (y1 - y0) / map_height,
            map_height,
            device=device,
            dtype=dtype,
        )
        if coordinate_normalization == "paper":
            x_grid = x_centers / float(x_scale_m)
            y_grid = y_centers / float(y_scale_m)
            z_grid = bev_z / float(z_normalization_m)
        else:
            # Legacy maps normalize x/y to the local BEV ROI and z by z_scale_m.
            x_grid = (x_centers - x0) / (x1 - x0) * 2.0 - 1.0
            y_grid = (y_centers - y0) / (y1 - y0) * 2.0 - 1.0
            z_grid = bev_z / float(z_scale_m)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        xy = torch.stack((xx, yy), dim=-1).expand(batch, -1, -1, -1)
        output = torch.cat((xy, z_grid[..., None]), dim=-1)
        if map_channels == 4:
            output = torch.cat((output, bev_valid[..., None]), dim=-1)
        elif map_channels == 5:
            # Channel 3 is physical support (a raw depth hit); channel 4 is
            # observation confidence after local hole filling.  Keeping these
            # separate prevents a stepping-stone gap from becoming indistinguishable
            # from an interpolated floor cell.
            output = torch.cat(
                (output, raw_bev_valid[..., None], bev_valid[..., None]), dim=-1
            )
        # Keep the latest map available to DELTA-specific rewards.  Rewards are
        # evaluated before the next observation pass, so this is intentionally a
        # one-step-lagged cache; it is preferable to duplicating the camera
        # projection inside every reward term and remains synchronized with the
        # policy input during normal stepping.
        cached_map = output.detach()
        env._delta_map_cache = cached_map
        if getattr(env, "delta_visual_debug_enabled", False):
            # Keep a detached, play-only snapshot for the Viser diagnostic panel.
            # Training does not retain image-sized tensors beyond the observation.
            env._delta_visual_debug = {
                "depth": depth.detach(),
                "bev_z": bev_z.detach(),
                "support": raw_bev_valid.detach(),
                "confidence": bev_valid.detach(),
            }
        if hasattr(env, "extras") and "log" in env.extras:
            finite_depth_ratio = valid.to(dtype).mean()
            inside_ratio = inside.to(dtype).mean()
            inside_points = points[inside]
            env.extras["log"]["DELTA/raw_map_occupancy_ratio"] = raw_bev_valid.mean()
            env.extras["log"]["DELTA/filled_map_ratio"] = bev_valid.mean()
            env.extras["log"]["DELTA/depth_valid_ratio"] = finite_depth_ratio
            env.extras["log"]["DELTA/bev_inside_ratio"] = inside_ratio
            # These four scalars expose perspective under-sampling without keeping a
            # per-cell profile in training memory. Columns are x (near -> far), rows
            # are y (+left -> -right) in the camera-facing BEV convention.
            x_mid = max(1, map_width // 2)
            y_mid = max(1, map_height // 2)
            env.extras["log"]["DELTA/support_x_near_ratio"] = raw_bev_valid[
                ..., :x_mid
            ].mean()
            env.extras["log"]["DELTA/support_x_far_ratio"] = raw_bev_valid[
                ..., x_mid:
            ].mean()
            env.extras["log"]["DELTA/support_y_left_ratio"] = raw_bev_valid[
                ..., :y_mid, :
            ].mean()
            env.extras["log"]["DELTA/support_y_right_ratio"] = raw_bev_valid[
                ..., y_mid:, :
            ].mean()
            env.extras["log"]["DELTA/map_observed_z_span"] = (
                bev_z.amax(dim=(-1, -2)) - bev_z.amin(dim=(-1, -2))
            ).mean()
            if inside_points.numel() > 0:
                env.extras["log"]["DELTA/projected_x_mean"] = inside_points[:, 0].mean()
                env.extras["log"]["DELTA/projected_y_mean"] = inside_points[:, 1].mean()
                env.extras["log"]["DELTA/projected_x_span"] = (
                    inside_points[:, 0].amax() - inside_points[:, 0].amin()
                )
                env.extras["log"]["DELTA/projected_y_span"] = (
                    inside_points[:, 1].amax() - inside_points[:, 1].amin()
                )
        if ablation_mode == "zero":
            output = output.clone()
            output[..., 2:] = 0.0
        elif ablation_mode == "shuffle":
            output = output.clone()
            if batch > 1:
                output[..., 2:] = output.roll(shifts=1, dims=0)[..., 2:]
            else:
                output[..., 2:] = output.flip(dims=(2,))[..., 2:]
        elif ablation_mode != "none":
            raise ValueError(
                "ablation_mode must be one of: 'none', 'zero', or 'shuffle'"
            )
        return output.reshape(batch, map_height * map_width * map_channels)

    points = points.permute(0, 3, 1, 2)
    points = torch.nn.functional.interpolate(
        points,
        size=(map_height, map_width),
        mode="bilinear",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    points[..., 0] = points[..., 0].clamp(0.0, max_depth) / max_depth
    points[..., 1] = points[..., 1].clamp(-1.5, 1.5) / 1.5
    points[..., 2] = points[..., 2].clamp(-0.8, 0.8) / 0.8
    return points.reshape(batch, map_height * map_width * 3)
