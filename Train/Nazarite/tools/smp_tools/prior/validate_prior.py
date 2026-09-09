"""Validate a trained Go2 SMP prior before using it for downstream PPO.

The audit checks three independent properties:

1. clean reference motion has lower epsilon-prediction error than temporally
   shuffled and joint-perturbed motion;
2. unconditional DDPM samples are finite and stay near the training support;
3. generated final states have valid Go2 joint limits, root poses, and MuJoCo FK.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from nazarite.config.train_config.train_algorithm.smp.prior.feature_to_state import (
    rotation_6d_to_matrix,
    rotation_6d_to_quaternion,
    split_features,
)
from nazarite.config.train_config.train_algorithm.smp.prior.features import (
    GO2_SMP_FEATURE_DIM,
    GO2_SMP_JOINT_NAMES,
)
from nazarite.config.train_config.train_algorithm.smp.prior.model import (
    DiffusionDenoiser,
)
from nazarite.config.train_config.train_algorithm.smp.prior.runtime import (
    load_prior,
    sample_prior,
)
from nazarite.config.train_config.train_algorithm.smp.prior.scheduler import (
    DDPMScheduler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GO2_XML = PROJECT_ROOT / "MJCF-Manager/Robots/GO2/xmls/go2.xml"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-eval-windows", type=int, default=512)
    parser.add_argument("--num-generated", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--timesteps", type=int, nargs="+", default=(8, 15, 22))
    parser.add_argument("--joint-noise-std", type=float, default=0.5)
    parser.add_argument("--root-height-range", type=float, nargs=2, default=(0.1, 0.8))
    parser.add_argument("--max-normalized-abs", type=float, default=4.0)
    parser.add_argument("--min-corruption-ratio", type=float, default=1.02)
    parser.add_argument("--min-generated-valid-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 2 when any acceptance criterion fails.",
    )
    return parser.parse_args(argv)


def _load_windows(data_dir: Path, window_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if "windows" not in archive.files:
                continue
            windows = np.asarray(archive["windows"], dtype=np.float32)
        expected = (window_size, GO2_SMP_FEATURE_DIM)
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
            raise ValueError(
                f"{path}: expected [N,{window_size},39], got {windows.shape}"
            )
        if not np.isfinite(windows).all():
            raise ValueError(f"{path}: contains NaN or Inf")
        chunks.append(windows)
    if not chunks:
        raise FileNotFoundError(f"no motion-window NPZ files in {data_dir}")
    return np.concatenate(chunks)


def _normalize(
    windows: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
    return 2.0 * (windows - q_low) / (q_high - q_low + 1.0e-8) - 1.0


def _make_variants(
    clean: torch.Tensor,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    joint_noise_std: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    batch, window_size, feature_dim = clean.shape
    permutation = torch.argsort(
        torch.rand(batch, window_size, generator=generator), dim=1
    )
    gather_index = permutation[..., None].expand(batch, window_size, feature_dim)
    shuffled = torch.gather(clean, 1, gather_index)
    joint_perturbed = clean.clone()
    joint_perturbed[..., 9:21] += joint_noise_std * torch.randn(
        joint_perturbed[..., 9:21].shape, generator=generator
    )
    return {
        "clean": _normalize(clean, q_low.cpu(), q_high.cpu()),
        "temporal_shuffle": _normalize(shuffled, q_low.cpu(), q_high.cpu()),
        "joint_perturbation": _normalize(joint_perturbed, q_low.cpu(), q_high.cpu()),
    }


@torch.no_grad()
def _score_variants(
    variants: dict[str, torch.Tensor],
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    timesteps: tuple[int, ...],
    noise_repeats: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    values: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    count = len(variants["clean"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batches = {
            name: value[start:stop].to(device) for name, value in variants.items()
        }
        batch_errors = {
            name: torch.zeros(stop - start, device=device) for name in variants
        }
        draws = 0
        for timestep in timesteps:
            if not 0 <= timestep < scheduler.num_timesteps:
                raise ValueError(
                    f"timestep {timestep} outside [0,{scheduler.num_timesteps})"
                )
            timestep_batch = torch.full(
                (stop - start,), timestep, dtype=torch.long, device=device
            )
            for _ in range(noise_repeats):
                noise = torch.randn_like(batches["clean"])
                for name, batch in batches.items():
                    noisy = scheduler.add_noise(batch, noise, timestep_batch)
                    prediction = model(noisy, timestep_batch)
                    batch_errors[name] += torch.square(prediction - noise).mean(
                        dim=(-1, -2)
                    )
                draws += 1
        for name in variants:
            values[name].append((batch_errors[name] / draws).cpu())

    merged = {name: torch.cat(chunks) for name, chunks in values.items()}
    summaries = {
        name: {
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "p50": float(value.quantile(0.5)),
            "p95": float(value.quantile(0.95)),
        }
        for name, value in merged.items()
    }
    clean = merged["clean"]
    comparisons = {}
    for name in ("temporal_shuffle", "joint_perturbation"):
        value = merged[name]
        comparisons[name] = {
            "mean_error_ratio": float(value.mean() / clean.mean().clamp_min(1.0e-8)),
            "paired_higher_fraction": float((value > clean).float().mean()),
        }
    return summaries, comparisons


@torch.no_grad()
def _generate(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    count: int,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, count, batch_size):
        chunks.append(
            sample_prior(
                model,
                scheduler,
                q_low,
                q_high,
                min(batch_size, count - start),
            ).cpu()
        )
    return torch.cat(chunks)


def _joint_metadata(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in GO2_SMP_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        raise ValueError("Go2 XML does not contain the expected SMP joints")
    qpos_addresses = model.jnt_qposadr[joint_ids].astype(np.int32)
    return qpos_addresses, model.jnt_range[joint_ids].copy()


def _physical_sample_audit(
    generated: torch.Tensor,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    xml_path: Path,
    root_height_range: tuple[float, float],
    max_normalized_abs: float,
) -> dict[str, Any]:
    finite_per_sample = torch.isfinite(generated).all(dim=(1, 2))
    normalized = _normalize(generated, q_low.cpu(), q_high.cpu())
    support_per_sample = torch.abs(normalized).amax(dim=(1, 2)) <= max_normalized_abs
    parts = split_features(generated)

    root_height = parts["root_pos"][..., 2]
    root_height_valid = (
        (root_height >= root_height_range[0]) & (root_height <= root_height_range[1])
    ).all(dim=1)

    rotation = parts["root_rot_6d"]
    column_0 = rotation[..., :3]
    column_2 = rotation[..., 3:6]
    norm_0 = torch.linalg.vector_norm(column_0, dim=-1)
    norm_2 = torch.linalg.vector_norm(column_2, dim=-1)
    cosine = torch.abs(
        (column_0 * column_2).sum(dim=-1) / (norm_0 * norm_2).clamp_min(1.0e-8)
    )
    rotation_valid = ((norm_0 > 1.0e-3) & (norm_2 > 1.0e-3) & (cosine < 0.999)).all(
        dim=1
    )
    matrices = rotation_6d_to_matrix(rotation)
    identity = torch.eye(3)[None, None]
    orthogonality_error = torch.abs(
        matrices.transpose(-1, -2) @ matrices - identity
    ).amax(dim=(-1, -2))

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos_addresses, joint_limits = _joint_metadata(mj_model)
    joints = parts["joint_pos"].numpy()
    within_joint_limits = (
        (joints >= joint_limits[None, None, :, 0])
        & (joints <= joint_limits[None, None, :, 1])
    ).all(axis=(1, 2))
    joint_valid = torch.from_numpy(within_joint_limits)

    free_joint_ids = np.flatnonzero(mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joint_ids) != 1:
        raise ValueError(f"expected one Go2 free joint, found {len(free_joint_ids)}")
    root_qpos_address = int(mj_model.jnt_qposadr[free_joint_ids[0]])
    final_quaternion = rotation_6d_to_quaternion(rotation[:, -1]).numpy()
    final_root_pos = parts["root_pos"][:, -1].numpy()
    fk_valid = np.zeros(len(generated), dtype=bool)
    contact_counts = np.zeros(len(generated), dtype=np.int32)
    mj_data = mujoco.MjData(mj_model)
    for index in range(len(generated)):
        if not bool(finite_per_sample[index]):
            continue
        try:
            mj_data.qpos[:] = mj_model.qpos0
            mj_data.qpos[root_qpos_address : root_qpos_address + 3] = final_root_pos[
                index
            ]
            mj_data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = (
                final_quaternion[index]
            )
            mj_data.qpos[qpos_addresses] = joints[index, -1]
            mujoco.mj_forward(mj_model, mj_data)
            fk_valid[index] = bool(
                np.isfinite(mj_data.qpos).all()
                and np.isfinite(mj_data.xpos).all()
                and np.isfinite(mj_data.site_xpos).all()
            )
            contact_counts[index] = mj_data.ncon
        except (FloatingPointError, ValueError):
            fk_valid[index] = False

    fk_valid_tensor = torch.from_numpy(fk_valid)
    valid = (
        finite_per_sample
        & support_per_sample
        & root_height_valid
        & rotation_valid
        & joint_valid
        & fk_valid_tensor
    )
    return {
        "count": len(generated),
        "finite_fraction": float(finite_per_sample.float().mean()),
        "within_normalized_bound_fraction": float(support_per_sample.float().mean()),
        "root_height_valid_fraction": float(root_height_valid.float().mean()),
        "rotation_valid_fraction": float(rotation_valid.float().mean()),
        "joint_limit_valid_fraction": float(joint_valid.float().mean()),
        "mujoco_fk_valid_fraction": float(fk_valid_tensor.float().mean()),
        "all_checks_valid_fraction": float(valid.float().mean()),
        "normalized_outside_train_quantiles_fraction": float(
            (torch.abs(normalized) > 1.0).float().mean()
        ),
        "normalized_max_abs": float(torch.abs(normalized).max()),
        "rotation_orthogonality_error_max": float(orthogonality_error.max()),
        "root_height_min": float(root_height.min()),
        "root_height_max": float(root_height.max()),
        "contact_count_p95": float(np.quantile(contact_counts, 0.95)),
        "contact_count_max": int(contact_counts.max()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    positive = (
        args.num_eval_windows,
        args.num_generated,
        args.batch_size,
        args.noise_repeats,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("window, sample, batch, and repeat counts must be positive")
    if args.joint_noise_std <= 0.0:
        raise ValueError("--joint-noise-std must be positive")
    root_height_range = tuple(float(value) for value in args.root_height_range)
    if root_height_range[0] >= root_height_range[1]:
        raise ValueError("--root-height-range must be increasing")

    checkpoint = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint.parent / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model, scheduler, q_low, q_high, feature_dim, window_size = load_prior(
        str(checkpoint), device
    )
    windows = _load_windows(data_dir, window_size)
    rng = np.random.default_rng(args.seed)
    eval_count = min(args.num_eval_windows, len(windows))
    indices = rng.choice(len(windows), size=eval_count, replace=False)
    clean = torch.from_numpy(windows[indices].copy())
    variants = _make_variants(
        clean,
        q_low,
        q_high,
        args.joint_noise_std,
        torch.Generator().manual_seed(args.seed),
    )
    timesteps = tuple(int(value) for value in args.timesteps)
    score_summary, comparisons = _score_variants(
        variants,
        model,
        scheduler,
        timesteps,
        args.noise_repeats,
        args.batch_size,
        device,
    )
    generated = _generate(
        model,
        scheduler,
        q_low,
        q_high,
        args.num_generated,
        args.batch_size,
    )
    generated_audit = _physical_sample_audit(
        generated,
        q_low,
        q_high,
        xml_path,
        (root_height_range[0], root_height_range[1]),
        args.max_normalized_abs,
    )

    criteria = {
        "finite_scores": all(
            np.isfinite(value["mean"]) for value in score_summary.values()
        ),
        "temporal_shuffle_separated": comparisons["temporal_shuffle"][
            "mean_error_ratio"
        ]
        >= args.min_corruption_ratio,
        "joint_perturbation_separated": comparisons["joint_perturbation"][
            "mean_error_ratio"
        ]
        >= args.min_corruption_ratio,
        "generated_states_valid": generated_audit["all_checks_valid_fraction"]
        >= args.min_generated_valid_fraction,
    }
    passed = all(criteria.values())
    report = {
        "format": "nazarite-go2-smp-prior-validation-v1",
        "passed": passed,
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "go2_xml": str(xml_path),
        "device": str(device),
        "dataset_windows": len(windows),
        "evaluated_windows": eval_count,
        "generated_windows": args.num_generated,
        "feature_dim": feature_dim,
        "window_size": window_size,
        "timesteps": list(timesteps),
        "noise_repeats": args.noise_repeats,
        "thresholds": {
            "min_corruption_ratio": args.min_corruption_ratio,
            "min_generated_valid_fraction": args.min_generated_valid_fraction,
            "max_normalized_abs": args.max_normalized_abs,
            "root_height_range": list(root_height_range),
        },
        "score_error": score_summary,
        "corruption_comparison": comparisons,
        "generated_audit": generated_audit,
        "criteria": criteria,
    }
    report_path = output_dir / "prior_validation.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    generated_path = output_dir / "generated_windows.npz"
    np.savez_compressed(
        generated_path,
        windows=generated.numpy().astype(np.float32),
        checkpoint=np.asarray(str(checkpoint)),
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "criteria": criteria,
                "corruption_comparison": comparisons,
                "generated_audit": generated_audit,
            },
            indent=2,
        )
    )
    print(f"Wrote: {report_path}")
    print(f"Wrote: {generated_path}")
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
