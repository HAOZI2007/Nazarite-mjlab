"""Minimal CPU-only ONNX actor smoke test.

This script deliberately does not drive a robot.  It checks that an exported
actor can be loaded on the target machine and reports inference latency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    session = ort.InferenceSession(
        str(args.policy),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"expected one input and one output, got {len(inputs)} and {len(outputs)}"
        )

    input_node = inputs[0]
    output_node = outputs[0]
    shape = input_node.shape
    if len(shape) != 2 or not isinstance(shape[1], int):
        raise RuntimeError(f"expected [batch, obs_dim] input, got {shape}")
    obs_dim = int(shape[1])
    obs = np.zeros((1, obs_dim), dtype=np.float32)

    for _ in range(max(0, args.warmup)):
        session.run([output_node.name], {input_node.name: obs})

    start = time.perf_counter()
    result = None
    for _ in range(args.iterations):
        result = session.run([output_node.name], {input_node.name: obs})[0]
    elapsed = time.perf_counter() - start

    if result is None:
        raise RuntimeError("no inference was executed")
    result = np.asarray(result)
    print(f"providers: {session.get_providers()}")
    print(f"input:     {input_node.name} {shape} {input_node.type}")
    print(f"output:    {output_node.name} {output_node.shape} {output_node.type}")
    print(f"metadata:  {session.get_modelmeta().custom_metadata_map}")
    print(f"obs_dim:   {obs_dim}")
    print(f"result:    shape={result.shape}, first={result.reshape(-1)[:12]}")
    print(f"latency:   {elapsed / args.iterations * 1000:.3f} ms/inference")
    print(f"rate:      {args.iterations / elapsed:.1f} inferences/s")


if __name__ == "__main__":
    main()
