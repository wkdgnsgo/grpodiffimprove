#!/bin/bash

#SBATCH --job-name=pure-grpo-vlm # 순수 GRPO VLM 학습
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod16
#SBATCH --ntasks-per-node=1 # total number of tasks per node
#SBATCH --cpus-per-task=4 # cpu-cores per task (멀티 GPU 처리용)
#SBATCH --gres=gpu:3 # 최소 3개 GPU 필요 (QWEN, SD3, CLIP)
#SBATCH --time=24:00:00 # total run time limit (HH:MM:SS)
#SBATCH --output=grpo_training.out
#SBATCH --error=grpo_training.err
#SBATCH --mail-type=begin # send email when job begins
#SBATCH --mail-type=end # send email when job ends
#SBATCH --mail-user=sunny17@kaist.ac.kr

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# 환경 활성화 (필요시)
#conda activate torch-env

# GPU 정보 출력
echo "=== GPU 정보 ==="
nvidia-smi
echo ""

# GPU 배치 정보 출력
echo "=== GPU 배치 계획 ==="
echo "GPU 0: QWEN VL 모델 (프롬프트 향상)"
echo "GPU 1: Stable Diffusion 3 (이미지 생성)"
echo "GPU 2: CLIP 리워드 모델 (리워드 계산)"
echo ""

# 순수 GRPO VLM 학습 실행
echo "=== 순수 GRPO VLM 학습 시작 ==="
python main.py
