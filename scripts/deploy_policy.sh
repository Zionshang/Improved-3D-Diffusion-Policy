#!/usr/bin/env bash

# Minimal deploy script for running inference.
# Usage examples:
#   bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/deploy_policy.sh dp_224x224_r3m gr1_dex-image 0913_example
# Notes:
#   - This script only passes parameters required for deployment.
#   - deploy.py will load model from: ${hydra.run.dir}/checkpoints/latest.ckpt
#     Make sure hydra.run.dir points to the training run output directory.

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}

# Build run_dir following the training output pattern.
# If your training run uses a different path, set this variable directly.
run_dir="data/outputs/${task_name}-${alg_name}-${addition_info}_seed0"

# Optional: select GPU visibility (deploy.py currently uses CPU for env tensors)
gpu_id=0
echo -e "\033[33mGPU to expose (optional): ${gpu_id}\033[0m"

cd Improved-3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python deploy_arx.py --config-name=${config_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir}



                                