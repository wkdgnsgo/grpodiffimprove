#!/usr/bin/env python3
"""
Accelerate DeviceMesh 수정 테스트 스크립트
"""

import torch
from accelerate import Accelerator
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_accelerate_preparation():
    """Accelerate 모델 준비 테스트"""
    logger.info("🚀 Accelerate 모델 준비 테스트 시작")
    
    # Accelerate 초기화
    accelerator = Accelerator()
    logger.info(f"✅ Accelerate 초기화 완료")
    logger.info(f"  - 프로세스 수: {accelerator.num_processes}")
    logger.info(f"  - 디바이스: {accelerator.device}")
    
    try:
        # QWEN 모델 및 컴포넌트 로드
        from qwen import QWENModel, QWENGRPOConfig
        
        config = QWENGRPOConfig(
            learning_rate=1e-6,
            batch_size=2,  # 테스트용 작은 배치
            num_rollouts=2,
            max_new_tokens=20
        )
        
        logger.info("🧠 QWEN 모델 로딩 중...")
        qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="accelerate",  # Accelerate 전용 모드
            temperature=0.7,
            grpo_config=config
        )
        
        # Accelerate로 모델 준비 (각각 개별적으로)
        logger.info("🔧 Accelerate 모델 준비 중...")
        qwen_model.model = accelerator.prepare(qwen_model.model)
        logger.info("✅ 모델 준비 완료")
        
        qwen_model.grpo_optimizer = accelerator.prepare(qwen_model.grpo_optimizer)
        logger.info("✅ 옵티마이저 준비 완료")
        
        # 간단한 테스트
        logger.info("🧪 간단한 모델 테스트...")
        test_result = qwen_model.enhance_prompt("test cat")
        logger.info(f"📝 테스트 결과: {test_result['enhanced_prompt'][:100]}...")
        
        logger.info("✅ Accelerate 모델 준비 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Accelerate 모델 준비 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_accelerate_preparation()
    if success:
        print("🎉 테스트 성공!")
    else:
        print("😢 테스트 실패") 