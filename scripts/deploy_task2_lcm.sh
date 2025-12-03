#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDP_DIR="$ROOT/Improved-3D-Diffusion-Policy"
SDK_DIR="$ROOT/arx5-sdk"

# Minimal env for ARX5 SDK
ARCH=$(uname -m); [[ "$ARCH" == "x86_64" ]] && LIB_ARCH=x86_64 || LIB_ARCH=aarch64
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SDK_DIR/python"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SDK_DIR/lib/$LIB_ARCH"

# If AMENT_PREFIX_PATH is empty, fall back to CONDA_PREFIX (common with ament on conda)
: "${AMENT_PREFIX_PATH:=${CONDA_PREFIX:-}}"; [[ -n "$AMENT_PREFIX_PATH" ]] && export AMENT_PREFIX_PATH

echo "Environment configured."
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "AMENT_PREFIX_PATH=$AMENT_PREFIX_PATH"

cd "$IDP_DIR"
python deploy_task2_lcm.py
