#!/bin/bash

#SBATCH --job-name=flowedit # create a short name for your job
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod05
#SBATCH --ntasks-per-node=1 # total number of tasks per node
#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4 # number of gpus per node
#SBATCH --time=07:00:00 # total run time limit (HH:MM:SS)
#SBATCH --output=fllm.out
#SBATCH --error=fllm.err
#SBATCH --mail-type=begin # send email when job begins
#SBATCH --mail-type=end # send email when job ends
#SBATCH --mail-user=sunny17@kaist.ac.kr

# 분산 처리 비활성화 (단일 GPU 모드)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DISABLE=1

# CUDA 메모리 최적화 - 강화된 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# 기존 분산 처리 환경 변수들 (주석 처리)
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$(($SLURM_NNODES * SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

echo "Starting Simple GRPO Training (Single GPU mode)..."
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

#conda activate torch-env
python main.py 