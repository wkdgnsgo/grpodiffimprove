#!/bin/bash
# Enhanced VLM GRPO Test Training Script
# ======================================
# 
# 빠른 테스트를 위한 스크립트입니다.
# 적은 반복으로 시스템이 정상 작동하는지 확인합니다.
#
# 사용법:
#   bash scripts/run_test_training.sh
#
# Author: AI Assistant (Based on MS Swift CoZ GRPO)
# Date: 2025-01-22

echo "🧪 Enhanced VLM GRPO Test Training"
echo "=================================="
echo "📋 Test 설정:"
echo "   - Train Type: LoRA (빠른 테스트)"
echo "   - Iterations: 3 (매우 적음)"
echo "   - Batch Size: 2 (작음)"
echo "   - Purpose: 시스템 검증"
echo ""

# 테스트 학습 실행
python vlm_grpo_system/run_enhanced_training.py \
    --train_type lora \
    --model microsoft/DialoGPT-medium \
    --lora_rank 4 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --group_size 2 \
    --num_generations 2 \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_length 512 \
    --max_completion_length 256 \
    --validation_interval 2 \
    --checkpoint_interval 2 \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 1 \
    --output_dir vlm_grpo_results_test \
    --log_completions \
    --device auto \
    --test \
    --debug

echo ""
echo "🎉 테스트 완료!"
echo "📊 결과 확인: vlm_grpo_results_test/"
echo "💡 정상 작동 확인 후 본격적인 학습을 시작하세요." 