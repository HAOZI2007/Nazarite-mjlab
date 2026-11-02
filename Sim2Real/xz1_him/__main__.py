"""Run XZ1 HIM inference tests: python -m Sim2Real.xz1_him."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .runtime import CONFIG, ONNXPolicy, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="XZ1 HIM zero-input ONNX benchmark")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--policy", type=Path, help="Override config ONNX path")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--hz",
        type=float,
        default=0,
        help="Inference frequency; 0 runs continuously without sleeping",
    )
    parser.add_argument("--output", type=Path, help="Save benchmark JSON")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")
    if not np.isfinite(args.hz) or args.hz < 0:
        parser.error("--hz must be finite and >= 0")
    config = load_config(args.config)
    path = (args.policy or Path(config["policy"])).expanduser().resolve()
    start = time.perf_counter_ns()
    policy = ONNXPolicy(path, config)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    observation = np.zeros((1, 282), dtype=np.float32)

    def infer():
        return policy.session.run(
            [policy.output_name], {policy.input_name: observation}
        )[0]

    start = time.perf_counter_ns()
    first_action = infer()
    first_ms = (time.perf_counter_ns() - start) / 1e6
    for _ in range(args.warmup):
        infer()
    durations = np.empty(args.iterations, dtype=np.float64)
    period = 1 / args.hz if args.hz else 0
    logical_cpus = os.cpu_count() or 1
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    late_cycles = 0
    for index in range(args.iterations):
        start = time.perf_counter_ns()
        action = infer()
        durations[index] = (time.perf_counter_ns() - start) / 1e6
        if action.shape != (1, 12) or not np.isfinite(action).all():
            raise ValueError("Invalid ONNX output")
        if period:
            remaining = wall_start + (index + 1) * period - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                late_cycles += 1
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    cpu_percent = cpu_seconds / wall_seconds * 100
    report = {
        "policy": str(path),
        "providers": policy.session.get_providers(),
        "input_shape": [1, 282],
        "input": "all zeros (including gravity/history)",
        "output_shape": list(first_action.shape),
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "requested_hz": args.hz,
        "actual_hz": args.iterations / wall_seconds,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent_one_core": cpu_percent,
        "process_cpu_percent_whole_machine": cpu_percent / logical_cpus,
        "logical_cpus": logical_cpus,
        "late_cycles_before_sleep": late_cycles,
        "model_load_ms": load_ms,
        "first_inference_ms": first_ms,
        "mean_ms": float(durations.mean()),
        "min_ms": float(durations.min()),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
        "max_ms": float(durations.max()),
        "control_budget_ms": config["control_dt"] * 1000,
        "over_budget_count": int(np.sum(durations > config["control_dt"] * 1000)),
        "raw_action": first_action[0].tolist(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
