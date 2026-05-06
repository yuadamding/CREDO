#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/run_hnscc_h100_heavy_c_joint_4cv_2gpu.sh "$@"
