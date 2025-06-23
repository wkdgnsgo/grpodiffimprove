#!/bin/bash
# Enhanced VLM GRPO Full Training Script
# ======================================
# 
# MS Swift CoZ GRPO를 참조한 전체 파라미터 학습 스크립트입니다.
# 고성능 학습을 위해 전체 모델을 학습합니다.
#
# 사용법:
#   bash scripts/run_full_training.sh
#
# 요구사항:
#   - 충분한 GPU 메모리 (16GB+)
#   - DeepSpeed 설치 권장
#
# Author: AI Assistant (Based on MS Swift CoZ GRPO)
# Date: 2025-01-22

echo "🚀 Enhanced VLM GRPO Full Training"
echo "=================================="
echo "📋 Full Training 설정:"
echo "   - Train Type: Full Parameter"
echo "   - DeepSpeed: zero2"
echo "   - High Performance: ✅"
echo "   - Memory Requirement: High"
echo ""

# 전체 학습 실행
python vlm_grpo_system/run_enhanced_training.py \
    --train_type full \
    --model microsoft/DialoGPT-medium \
    --torch_dtype bfloat16 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --group_size 4 \
    --num_generations 7 \
    --temperature 0.9 \
    --top_p 0.9 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --validation_interval 3 \
    --checkpoint_interval 5 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 2 \
    --output_dir vlm_grpo_results_full \
    --log_completions \
    --use_wandb \
    --wandb_project vlm-grpo-full \
    --device auto \
    --use_flash_attention \
    --gradient_checkpointing \
    --use_deepspeed \
    --deepspeed zero2

echo ""
echo "🎉 Full Training 완료!"
echo "📊 결과 확인: vlm_grpo_results_full/" 