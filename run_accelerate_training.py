#!/usr/bin/env python3
"""
Accelerate 멀티 GPU QWEN GRPO 학습 실행 스크립트
GPU 0-3: QWEN RL 학습 (Accelerate 분산)
GPU 4: SD3 + CLIP + Reference 모델 (통합)
"""

import subprocess
import sys
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_accelerate_config():
    """Accelerate 설정 파일 생성"""
    config_content = """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0,1,2,3'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    # accelerate config 디렉토리 생성
    config_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
    os.makedirs(config_dir, exist_ok=True)
    
    # 설정 파일 저장
    config_path = os.path.join(config_dir, "default_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    logger.info(f"✅ Accelerate 설정 파일 생성: {config_path}")
    return config_path

def check_gpu_availability():
    """GPU 사용 가능성 확인"""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("❌ CUDA를 사용할 수 없습니다.")
            return False
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ 사용 가능한 GPU 개수: {gpu_count}")
        
        if gpu_count < 5:
            logger.warning(f"⚠️ GPU가 {gpu_count}개만 있습니다. 최소 5개 권장 (0-3: QWEN, 4: 기타 모델)")
            if gpu_count < 4:
                logger.error("❌ 최소 4개의 GPU가 필요합니다.")
                return False
        
        # GPU 메모리 확인
        for i in range(min(gpu_count, 5)):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f}GB)")
        
        return True
        
    except ImportError:
        logger.error("❌ PyTorch가 설치되지 않았습니다.")
        return False

def run_accelerate_training():
    """Accelerate로 멀티 GPU 학습 실행"""
    logger.info("🚀 Accelerate 멀티 GPU QWEN GRPO 학습 시작")
    logger.info("=" * 80)
    
    # 1. GPU 확인
    if not check_gpu_availability():
        logger.error("❌ GPU 요구사항을 만족하지 않습니다.")
        return False
    
    # 2. Accelerate 설정
    config_path = setup_accelerate_config()
    
    # 3. 환경 변수 설정
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  # 5개 GPU 사용
    env["TOKENIZERS_PARALLELISM"] = "false"    # 토크나이저 병렬 처리 비활성화
    env["WANDB_DISABLED"] = "true"             # WandB 비활성화 (선택사항)
    
    # 4. Accelerate 명령어 구성
    cmd = [
        "accelerate", "launch",
        "--config_file", config_path,
        "--main_process_port", "29500",
        "main.py"
    ]
    
    logger.info("🎯 실행 명령어:")
    logger.info(f"  {' '.join(cmd)}")
    logger.info("\n📋 GPU 배치:")
    logger.info("  GPU 0-3: QWEN RL 학습 (Accelerate 분산)")
    logger.info("  GPU 4: SD3 + CLIP + Reference 모델")
    
    try:
        # 5. 학습 실행
        logger.info("\n🚀 학습 시작...")
        result = subprocess.run(cmd, env=env, check=True)
        
        if result.returncode == 0:
            logger.info("✅ 학습이 성공적으로 완료되었습니다!")
            return True
        else:
            logger.error(f"❌ 학습이 실패했습니다. 종료 코드: {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 학습 실행 중 오류: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("⚠️ 사용자에 의해 학습이 중단되었습니다.")
        return False
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        return False

def main():
    """메인 함수"""
    success = run_accelerate_training()
    
    if success:
        logger.info("\n🎉 프로그램이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        logger.error("\n❌ 프로그램 실행 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 