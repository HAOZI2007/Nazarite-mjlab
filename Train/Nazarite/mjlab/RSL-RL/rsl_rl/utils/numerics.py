"""Coordinated numerical checks for HIM training."""

from collections import deque
from pathlib import Path
import time
import warnings

import torch


def globally_bad(bad: torch.Tensor) -> bool:
    """Return whether any distributed worker reports a numerical fault."""
    flag = bad.any().to(dtype=torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def tensor_items(value, prefix=""):
    """Yield tensors nested in dictionaries, sequences, and optimizer states."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif hasattr(value, "items"):
        for index, (name, item) in enumerate(value.items()):
            label = name if isinstance(name, (str, int)) else index
            yield from tensor_items(item, f"{prefix}/{label}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f"{prefix}/{index}")


class TrainingNumericsGuard:
    """Stop every worker before corrupt data reaches an optimizer step."""

    def __init__(self, device, log_dir=None, **cfg):
        self.device = device
        self.log_dir = Path(log_dir or ".") / "numerics"
        self.cfg = cfg
        self.recent = deque(maxlen=cfg.get("history_steps", 8))
        self.iteration = -1
        self.step = -1
        self._recent_key = None

    def remember(self, **values):
        key = (self.iteration, self.step)
        if not self.recent or key != self._recent_key:
            self.recent.append({})
            self._recent_key = key
        self.recent[-1].update({name: value.detach().clone() for name, value in tensor_items(values)})

    def check(self, stage, values, *, limit=None, extra_bad=None):
        items = list(tensor_items(values))
        bad = torch.zeros((), dtype=torch.bool, device=self.device)
        for _, value in items:
            bad = bad | (~torch.isfinite(value)).any()
            if limit is not None:
                bad = bad | (value.abs() > limit).any()
        if extra_bad is not None:
            bad = bad | extra_bad.any()
        if globally_bad(bad):
            self.fail(stage, values, limit=limit, local_bad=bool(bad.item()))

    def fail(self, stage, values, **details):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = self.log_dir / f"incident_{time.time_ns()}_rank{rank}.pt"

        def snapshot(value):
            result = {}
            for name, tensor in tensor_items(value):
                flat = tensor.detach().reshape(-1)
                mask = ~torch.isfinite(flat)
                limit = details.get("limit")
                if limit is not None:
                    mask |= flat.abs() > limit
                indices = mask.nonzero().flatten()[:256]
                result[name] = {
                    "shape": tuple(tensor.shape),
                    "sample": flat[:65536].cpu(),
                    "bad_indices": indices.cpu(),
                    "bad_values": flat[indices].cpu(),
                }
            return result

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "stage": stage,
                "rank": rank,
                "iteration": self.iteration,
                "step": self.step,
                "details": details,
                "config": self.cfg,
                "current": snapshot(values),
                "recent": [{key: value.cpu() for key, value in frame.items()} for frame in self.recent],
            }, path)
        except Exception as exc:
            warnings.warn(f"Could not write numerical incident {path}: {exc}")
        raise FloatingPointError(f"HIM numerical fault at {stage}; rank {rank}; diagnostics: {path}")

    def gradients(self, stage, parameters):
        parameters = list(parameters)
        self.check(stage, {str(index): parameter.grad for index, parameter in enumerate(parameters) if parameter.grad is not None})
