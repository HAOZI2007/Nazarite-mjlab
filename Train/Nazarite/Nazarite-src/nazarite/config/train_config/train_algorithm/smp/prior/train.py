"""Train the frozen DDPM motion prior used by Go2 SMP."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, random_split

from .dataset import MotionWindowDataset
from .features import GO2_SMP_FEATURE_DIM, GO2_SMP_WINDOW_SIZE
from .model import DiffusionDenoiser
from .scheduler import DDPMScheduler

_PROJECT_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_SMP_PRIOR_ROOT = _PROJECT_ROOT / "tools/smp_dataset/smp_prior"


def _resolve_prior_output(path: Path, option_name: str) -> Path:
    """Resolve an output path while keeping every prior under its asset root."""
    root = DEFAULT_SMP_PRIOR_ROOT.resolve()
    candidate = path.expanduser()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) and not candidate.is_absolute():
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{option_name} must be inside {root}, got {resolved}")
    return resolved


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        source_state = source.state_dict()
        for name, target in self.model.state_dict().items():
            source_value = source_state[name].detach()
            if target.is_floating_point():
                target.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                target.copy_(source_value)


def diffusion_loss(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    clean: torch.Tensor,
    num_noise_samples: int,
) -> torch.Tensor:
    batch = clean.shape[0]
    expanded = (
        clean[:, None]
        .expand(batch, num_noise_samples, *clean.shape[1:])
        .reshape(batch * num_noise_samples, *clean.shape[1:])
    )
    timesteps = scheduler.sample_timesteps(len(expanded), clean.device)
    noise = torch.randn_like(expanded)
    noisy = scheduler.add_noise(expanded, noise, timesteps)
    return functional.l1_loss(model(noisy, timesteps), noise)


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: DiffusionDenoiser,
    ema: ExponentialMovingAverage | None,
    optimizer: torch.optim.Optimizer | None,
    dataset: MotionWindowDataset,
    config: dict[str, Any],
) -> None:
    checkpoint: dict[str, Any] = {
        "format": "nazarite-go2-smp-prior-v1",
        "epoch": epoch,
        "model": model.state_dict(),
        "q_low": dataset.q_low,
        "q_high": dataset.q_high,
        "cfg": {
            **config,
            "feature_dim": dataset.feature_dim,
            "window_size": dataset.window_size,
        },
    }
    if ema is not None:
        checkpoint["model_ema"] = ema.model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    num_noise_samples: int,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        losses.append(
            diffusion_loss(model, scheduler, batch.to(device), num_noise_samples).item()
        )
    return float(np.mean(losses)) if losses else float("nan")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_SMP_PRIOR_ROOT)
    parser.add_argument("--name", default="go2_3ddogs_1x")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--num-noise-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--export-path",
        type=Path,
        help=(
            "Also write the final inference checkpoint to this stable path. "
            "Defaults to <smp_prior>/<name>/pretrained.pt; custom paths must "
            "remain inside the SMP prior root."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not 0.0 < args.train_split < 1.0:
        raise ValueError("--train-split must be in (0,1)")
    positive = (
        args.d_model,
        args.nhead,
        args.num_layers,
        args.num_timesteps,
        args.num_noise_samples,
        args.batch_size,
        args.num_epochs,
        args.log_interval,
        args.save_interval,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("model and training counts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    log_dir = _resolve_prior_output(args.log_dir, "--log-dir")
    export_path = _resolve_prior_output(
        args.export_path
        if args.export_path is not None
        else DEFAULT_SMP_PRIOR_ROOT / args.name / "pretrained.pt",
        "--export-path",
    )
    dataset = MotionWindowDataset(args.data_dir, args.norm_stats)
    if (
        dataset.feature_dim != GO2_SMP_FEATURE_DIM
        or dataset.window_size != GO2_SMP_WINDOW_SIZE
    ):
        raise ValueError(
            f"expected Go2 SMP [*,{GO2_SMP_WINDOW_SIZE},{GO2_SMP_FEATURE_DIM}], "
            f"got [*,{dataset.window_size},{dataset.feature_dim}]"
        )
    train_count = max(1, int(len(dataset) * args.train_split))
    validation_count = len(dataset) - train_count
    if validation_count == 0:
        train_count -= 1
        validation_count = 1
    generator = torch.Generator().manual_seed(args.seed)
    train_set, validation_set = random_split(
        dataset, (train_count, validation_count), generator=generator
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = DiffusionDenoiser(
        dataset.feature_dim,
        dataset.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    scheduler = DDPMScheduler(args.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    ema = ExponentialMovingAverage(model, args.ema_decay) if args.use_ema else None
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_dir / args.name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["resolved_log_dir"] = str(log_dir)
    config["resolved_export_path"] = str(export_path)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"dataset={len(dataset)} train={train_count} validation={validation_count} "
        f"shape=[{dataset.window_size},{dataset.feature_dim}] parameters={parameter_count:,} "
        f"device={device}"
    )
    history = []
    for epoch in range(args.num_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            clean = batch.to(device, non_blocking=device.type == "cuda")
            loss = diffusion_loss(model, scheduler, clean, args.num_noise_samples)
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        if epoch % args.log_interval == 0 or epoch == args.num_epochs - 1:
            evaluation_model = ema.model if ema is not None else model
            validation_loss = _validate(
                evaluation_model,
                scheduler,
                validation_loader,
                device,
                args.num_noise_samples,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"epoch={epoch:05d} train={train_loss:.6f} validation={validation_loss:.6f}"
            )
        if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
            _save_checkpoint(
                run_dir / f"checkpoint_{epoch:05d}.pt",
                epoch,
                model,
                ema,
                optimizer,
                dataset,
                config,
            )
    final_path = run_dir / "pretrained.pt"
    _save_checkpoint(final_path, args.num_epochs, model, ema, None, dataset, config)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(export_path, args.num_epochs, model, ema, None, dataset, config)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
        stream.write("\n")
    print(f"Wrote: {final_path}")
    print(f"Exported: {export_path}")


if __name__ == "__main__":
    main()
