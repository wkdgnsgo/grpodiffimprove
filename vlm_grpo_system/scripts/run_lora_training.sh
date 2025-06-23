#!/bin/bash
# Enhanced VLM GRPO LoRA Training Script
# =====================================
# 
# MS Swift CoZ GRPO를 참조한 LoRA 학습 스크립트입니다.
# 메모리 효율적인 학습을 위해 LoRA를 사용합니다.
#
# 사용법:
#   bash scripts/run_lora_training.sh
#
# Author: AI Assistant (Based on MS Swift CoZ GRPO)
# Date: 2025-01-22

echo "🚀 Enhanced VLM GRPO LoRA Training"
echo "=================================="
echo "📋 LoRA 설정:"
echo "   - Rank: 8"
echo "   - Alpha: 32"
echo "   - Target Modules: all-linear"
echo "   - Memory Efficient: ✅"
echo ""

# LoRA 학습 실행
python vlm_grpo_system/run_enhanced_training.py \
    --train_type lora \
    --model microsoft/DialoGPT-medium \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 20 \
    --group_size 4 \
    --num_generations 4 \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --validation_interval 5 \
    --checkpoint_interval 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --output_dir vlm_grpo_results_lora \
    --log_completions \
    --use_wandb \
    --wandb_project vlm-grpo-lora \
    --device auto \
    --use_flash_attention \
    --gradient_checkpointing

echo ""
echo "🎉 LoRA 학습 완료!"
echo "📊 결과 확인: vlm_grpo_results_lora/" 