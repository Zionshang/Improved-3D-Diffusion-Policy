#!/usr/bin/env bash

set -e

# Usage: bash scripts/deploy_policy.sh <alg_name> <task_name> [addition_info]
alg_name=${1:-}
task_name=${2:-}
addition_info=${3:-run}

if [[ -z "$alg_name" || -z "$task_name" ]]; then
    echo "Usage: $0 <alg_name> <task_name> [addition_info]"
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDP_DIR="$ROOT/Improved-3D-Diffusion-Policy"
SDK_DIR="$ROOT/arx5-sdk"

# Minimal env for ARX5 SDK
ARCH=$(uname -m); [[ "$ARCH" == "x86_64" ]] && LIB_ARCH=x86_64 || LIB_ARCH=aarch64
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SDK_DIR/python"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SDK_DIR/lib/$LIB_ARCH"

# If AMENT_PREFIX_PATH is empty, fall back to CONDA_PREFIX (common with ament on conda)
: "${AMENT_PREFIX_PATH:=${CONDA_PREFIX:-}}"; [[ -n "$AMENT_PREFIX_PATH" ]] && export AMENT_PREFIX_PATH

run_dir="data/outputs/${task_name}-${alg_name}-${addition_info}_seed0"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

cd "$IDP_DIR"
python deploy_arx.py --config-name="${alg_name}.yaml" \
    task="${task_name}" \
    hydra.run.dir="${run_dir}"



                                