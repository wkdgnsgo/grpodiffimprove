#!/bin/bash

# Multi-GPU GRPO Training Script
# ===============================
# GPU 1, 2, 3번을 사용하여 QWEN 모델을 GRPO로 학습합니다.

echo "🚀 Starting Multi-GPU GRPO Training"
echo "=================================="

# GPU 환경 변수 설정
export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

echo "🔧 GPU Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

# GPU 상태 확인
echo ""
echo "📱 GPU Status Check:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep -E "^[1-3],"

echo ""
echo "🎯 Model GPU Assignment:"
echo "  GPU 1 (cuda:0): QWEN VL Model (Policy Network)"
echo "  GPU 2 (cuda:1): Stable Diffusion 3 (Environment)"  
echo "  GPU 3 (cuda:2): CLIP Model (Reward Calculator)"

echo ""
echo "⚡ Starting GRPO Training..."
echo "=================================="

# Python 학습 실행
python train_grpo.py

# 학습 완료 후 GPU 메모리 상태 확인
echo ""
echo "📊 Post-Training GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep -E "^[1-3],"

echo ""
echo "🎉 Multi-GPU GRPO Training Completed!" 