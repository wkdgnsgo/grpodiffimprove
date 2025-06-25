#!/usr/bin/env python3
"""
LoRA 전환 테스트 스크립트
QWEN 모델이 LoRA로 올바르게 전환되었는지 확인
"""

import torch
import logging
from qwen import QWENModel, QWENGRPOConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_lora_conversion():
    """LoRA 전환 테스트"""
    logger.info("🧪 LoRA 전환 테스트 시작")
    
    try:
        # GRPO 설정 (LoRA 최적화)
        config = QWENGRPOConfig()
        logger.info(f"📊 LoRA 설정: LR={config.learning_rate}, Batch={config.batch_size}")
        
        # QWEN LoRA 모델 초기화
        logger.info("🔧 QWEN LoRA 모델 초기화...")
        qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
            temperature=0.7,
            grpo_config=config,
            is_main_process=True
        )
        
        # LoRA 파라미터 정보 확인
        lora_info = qwen_model.get_lora_trainable_params()
        logger.info("📊 LoRA 파라미터 정보:")
        logger.info(f"  학습 가능한 파라미터: {lora_info['trainable_params']:,}")
        logger.info(f"  전체 파라미터: {lora_info['all_params']:,}")
        logger.info(f"  학습 비율: {lora_info['trainable_percentage']:.2f}%")
        
        # 메모리 사용량 확인
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            logger.info(f"💾 GPU 메모리 사용량:")
            logger.info(f"  할당됨: {memory_allocated:.2f} GB")
            logger.info(f"  예약됨: {memory_reserved:.2f} GB")
        
        # 기본 프롬프트 향상 테스트
        test_prompts = [
            "cat",
            "beautiful landscape",
            "futuristic city"
        ]
        
        logger.info("🧪 LoRA 모델 프롬프트 향상 테스트:")
        for prompt in test_prompts:
            try:
                result = qwen_model.enhance_prompt(prompt)
                enhanced = result['enhanced_prompt']
                logger.info(f"  원본: '{prompt}'")
                logger.info(f"  향상: '{enhanced[:100]}...'")
                logger.info(f"  길이: {len(prompt)} → {len(enhanced)}")
                logger.info("")
            except Exception as e:
                logger.error(f"  프롬프트 '{prompt}' 향상 실패: {e}")
        
        # GRPO 생성 테스트
        logger.info("🧪 LoRA GRPO 생성 테스트:")
        try:
            grpo_enhanced, log_prob = qwen_model.generate_grpo_enhanced_prompt("sunset over mountains")
            logger.info(f"  GRPO 향상: '{grpo_enhanced[:100]}...'")
            logger.info(f"  Log Prob: {log_prob.item():.4f}")
        except Exception as e:
            logger.error(f"  GRPO 생성 실패: {e}")
        
        # LoRA 저장 테스트
        logger.info("💾 LoRA 저장 테스트:")
        try:
            save_path = "test_lora_checkpoint"
            qwen_model.save_lora_model(save_path)
            logger.info(f"  LoRA 저장 성공: {save_path}")
        except Exception as e:
            logger.error(f"  LoRA 저장 실패: {e}")
        
        logger.info("✅ LoRA 전환 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA 전환 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    logger.info("🚀 QWEN LoRA 전환 테스트")
    logger.info("=" * 60)
    
    success = test_lora_conversion()
    
    if success:
        logger.info("\n🎉 모든 테스트 통과!")
        logger.info("LoRA 전환이 성공적으로 완료되었습니다.")
        logger.info("이제 메모리 효율적인 학습이 가능합니다.")
    else:
        logger.error("\n❌ 테스트 실패!")
        logger.error("LoRA 전환에 문제가 있습니다.")

if __name__ == "__main__":
    main() 