#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv_bin="${UV_BIN:-/home/haozi/miniforge3/envs/mjlab_env/bin/uv}"
exec "$uv_bin" run --no-sync python -m sim2sim.xz1_teacher --gamepad "$@"
