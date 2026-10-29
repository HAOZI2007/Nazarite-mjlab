"""Offline XZ1 WMP World Model training with held-out terrain seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg
from nazarite.rl.wmp.xz1_dataset import CONTRACT, XZ1SequenceDataset


def sample_batches(dataset, batch_size: int, batches: int, seed: int):
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=batch_size * batches,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def posterior_features(model, batch):
    image = batch["image"].float() / 255 - 0.5
    embed = model.encode(image, batch["prop"])
    post, _ = model.dynamics.observe(
        embed, batch["action"].clone(), batch["is_first"].float()
    )
    feature = post["deter"]
    if not torch.isfinite(feature).all():
        raise RuntimeError("Non-finite held-out World Model features")
    return image, post, feature


@torch.no_grad()
def validate_world_model(model, train, validation, batch_size=4, batches=4, seed=42):
    # The upstream RSSM samples even in eval mode. Keep that behavior, but use
    # fixed validation draws without consuming the training RNG stream.
    device = next(model.parameters()).device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        return _validate_world_model(
            model, train, validation, batch_size, batches, seed
        )


def _validate_world_model(model, train, validation, batch_size=4, batches=4, seed=42):
    """Evaluate future images and a diagnostic ridge probe, never update WM."""
    was_training = model.training
    model.eval()
    losses: dict[str, list[float]] = {}
    train_features, train_labels, val_features, val_labels = [], [], [], []
    try:
        for split, dataset in (("train", train), ("validation", validation)):
            for batch in sample_batches(dataset, batch_size, batches, seed):
                batch = on_device(batch, next(model.parameters()).device)
                image, post, feature = posterior_features(model, batch)
                if split == "train":
                    train_features.append(feature.flatten(0, 1).cpu().double())
                    train_labels.append(
                        batch["teacher_latent"].flatten(0, 1).cpu().double()
                    )
                    continue
                val_features.append(feature.flatten(0, 1).cpu().double())
                val_labels.append(batch["teacher_latent"].flatten(0, 1).cpu().double())
                _, metrics = model.compute_loss(batch)
                for key, value in metrics.items():
                    losses.setdefault(key, []).append(float(value))
                # Posterior conditions only on the context. Future propagation
                # sees actions, never future depth or proprioceptive encodings.
                context = min(5, image.shape[1] - 1)
                horizon = min(5, image.shape[1] - context)
                state = {key: value[:, context - 1] for key, value in post.items()}
                predictions = []
                for step in range(context, context + horizon):
                    state = model.dynamics.img_step(state, batch["action"][:, step])
                    decoded = model.decode_image(state)
                    predictions.append(decoded)
                prediction = torch.stack(predictions, dim=1)
                target = image[:, context : context + horizon]
                baseline = image[:, context - 1 : context].expand_as(target)
                future_per_episode = (prediction - target).square().flatten(1).mean(1)
                baseline_per_episode = (baseline - target).square().flatten(1).mean(1)
                for terrain_id in batch["terrain_id"].unique().tolist():
                    mask = batch["terrain_id"] == terrain_id
                    name = validation.terrain_names[terrain_id]
                    losses.setdefault(f"terrain/{name}/future_image_mse", []).extend(
                        future_per_episode[mask].cpu().tolist()
                    )
                    losses.setdefault(f"terrain/{name}/persistence_mse", []).extend(
                        baseline_per_episode[mask].cpu().tolist()
                    )
                losses.setdefault("future_image_mse", []).append(
                    float(F.mse_loss(prediction, target))
                )
                losses.setdefault("persistence_mse", []).append(
                    float(F.mse_loss(baseline, target))
                )
                losses.setdefault("feature_std", []).append(
                    float(feature.flatten(0, 1).std(0).mean())
                )
        # This probe is an evaluation instrument, not part of the student or WM
        # objective. Fit on train seeds, evaluate on held-out seeds, with a bias.
        x, y = torch.cat(train_features), torch.cat(train_labels)
        vx, vy = torch.cat(val_features), torch.cat(val_labels)
        xmean, xstd = x.mean(0), x.std(0).clamp_min(1e-3)
        x = torch.cat(((x - xmean) / xstd, torch.ones(len(x), 1)), dim=1)
        vx = torch.cat(((vx - xmean) / xstd, torch.ones(len(vx), 1)), dim=1)
        gram = x.T @ x
        regularizer = torch.eye(gram.shape[0], dtype=gram.dtype) * 1.0
        regularizer[-1, -1] = 0
        weights = torch.linalg.solve(gram + regularizer, x.T @ y)
        losses["terrain_probe_mse"] = [float(F.mse_loss(vx @ weights, vy))]
        losses["terrain_mean_baseline_mse"] = [
            float(F.mse_loss(y.mean(0).expand_as(vy), vy))
        ]
        result = {key: sum(values) / len(values) for key, values in losses.items()}
        result["future_to_persistence_ratio"] = result["future_image_mse"] / max(
            result["persistence_mse"], 1e-12
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in result.values()):
            raise RuntimeError("Non-finite validation metrics")
        return result
    finally:
        model.train(was_training)


def run_pretraining(args, cfg: WMPWorldModelCfg | None = None):
    torch.manual_seed(args.seed)
    cfg = cfg or WMPWorldModelCfg(action_chunk=5, reward_loss_scale=0.0)
    if cfg.action_chunk != 5 or cfg.prop_dim != 33 or cfg.action_dim != 12:
        raise ValueError("Model configuration does not match the XZ1 data contract")
    train = XZ1SequenceDataset(args.dataset, "train", args.batch_length)
    validation = XZ1SequenceDataset(args.dataset, "validation", args.batch_length)
    manifest_hash = hashlib.sha256(
        (args.dataset / "manifest.json").read_bytes()
    ).hexdigest()
    model = WMPWorldModel(cfg, device=args.device)
    start = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=args.device, weights_only=True
        )
        if checkpoint.get("world_model_implementation") != model.implementation:
            raise ValueError(
                "Resume requires a reference Dreamer WM checkpoint; the former "
                "compact World Model is incompatible. Reuse the dataset and "
                "pretrain a new WM. The teacher checkpoint is unaffected."
            )
        if checkpoint["model_cfg"] != asdict(cfg) or checkpoint["contract"] != CONTRACT:
            raise ValueError("Resume model/data contract mismatch")
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume requires the original frozen dataset manifest")
        model.load_state_dict(checkpoint["world_model"], strict=True)
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
    if args.updates <= start:
        raise ValueError("--updates is the total update target, above resumed step")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "model_cfg": asdict(cfg),
        "world_model_implementation": model.implementation,
        "contract": CONTRACT,
        "dataset": str(args.dataset.resolve()),
        "manifest_sha256": manifest_hash,
        "batch_size": args.batch_size,
        "batch_length": args.batch_length,
        "sequence_stats": {"train": train.stats, "validation": validation.stats},
        "seed": args.seed,
        "resume": str(args.resume) if args.resume else None,
        "reference": "wmp_go1-main/dreamer/{models.py,networks.py,tools.py,configs.yaml}",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print("Sequence availability: " + json.dumps(config["sequence_stats"]), flush=True)
    writer = SummaryWriter(str(args.output))
    try:
        # Establish a before-training baseline on the identical held-out windows.
        metrics = validate_world_model(
            model,
            train,
            validation,
            args.batch_size,
            args.validation_batches,
            args.seed,
        )
        for key, value in metrics.items():
            writer.add_scalar(f"Validation/{key}", value, start)
        history = [{"step": start, "validation": metrics}]
        batches = sample_batches(
            train, args.batch_size, args.updates - start, args.seed
        )
        for step, batch in enumerate(batches, start=start + 1):
            model.train()
            metrics = model.train_batch(on_device(batch, args.device))
            for key, value in metrics.items():
                writer.add_scalar(f"Train/{key}", value, step)
            if step % args.eval_interval == 0 or step == args.updates:
                validation_metrics = validate_world_model(
                    model,
                    train,
                    validation,
                    args.batch_size,
                    args.validation_batches,
                    args.seed,
                )
                for key, value in validation_metrics.items():
                    writer.add_scalar(f"Validation/{key}", value, step)
                history.append(
                    {"step": step, "train": metrics, "validation": validation_metrics}
                )
                (args.output / "metrics.json").write_text(json.dumps(history, indent=2))
                temporary = args.output / f"wm_{step}.pt.tmp"
                torch.save(
                    {
                        **config,
                        "step": step,
                        "world_model": model.state_dict(),
                        "optimizer": model.optimizer.state_dict(),
                        "validation": validation_metrics,
                    },
                    temporary,
                )
                temporary.replace(args.output / f"wm_{step}.pt")
                print(
                    f"WM step {step}: train={metrics['loss']:.5f}, "
                    f"future/persistence="
                    f"{validation_metrics['future_to_persistence_ratio']:.3f}",
                    flush=True,
                )
        return history
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        min(args.updates, args.batch_size, args.eval_interval, args.validation_batches)
        < 1
    ):
        parser.error("Update, batch and evaluation counts must be positive")
    if args.batch_length < 2 or args.output.exists():
        parser.error("Need sequence length >=2 and a new output directory")
    run_pretraining(args)


if __name__ == "__main__":
    main()
