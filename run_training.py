#!/usr/bin/env python3
"""
VLM GRPO Training System Main Runner
===================================

VLM GRPO 학습 시스템의 메인 실행 스크립트입니다.

사용법:
    python run_training.py                    # 기본 설정으로 실행
    python run_training.py --config my.json  # 커스텀 설정으로 실행
    python run_training.py --test            # 테스트 모드로 실행

Author: AI Assistant
Date: 2025-01-22
"""

import argparse
import sys
import os
import json
from pathlib import Path
import logging

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> dict:
    """설정 파일 로드"""
    if config_path is None:
        config_path = current_dir / "config" / "default_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"⚙️ Config loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise

def main():
    """메인 실행 함수"""
    logger.info("🚀 Starting VLM GRPO Training System")
    
    try:
        # 1. 설정 로드
        config = load_config()
        
        # 2. 출력 디렉토리 생성
        output_dir = config.get("output_settings", {}).get("output_dir", "vlm_grpo_results")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # 3. 메인 트레이너 초기화
        from integration.main_trainer import VLMGRPOSystem
        trainer = VLMGRPOSystem()
        
        # 4. 컴포넌트 초기화
        trainer.initialize_components()
        
        # 5. 학습 실행
        trainer.run_training()
        
        logger.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def create_sample_data():
    """샘플 데이터 생성 함수"""
    print("📊 샘플 데이터 생성 중...")
    
    try:
        from utils.data_loader import create_sample_data
        create_sample_data(
            train_path="train_prompts.jsonl",
            val_path="val_prompts.jsonl"
        )
        print("✅ 샘플 데이터가 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 샘플 데이터 생성 실패: {e}")


def show_system_info():
    """시스템 정보 출력"""
    print("💻 시스템 정보:")
    print(f"  - Python: {sys.version}")
    print(f"  - 작업 디렉토리: {os.getcwd()}")
    
    try:
        import torch
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"  - MPS 사용 가능: {torch.backends.mps.is_available()}")
    except ImportError:
        print("  - PyTorch: 설치되지 않음")
    
    try:
        import transformers
        print(f"  - Transformers: {transformers.__version__}")
    except ImportError:
        print("  - Transformers: 설치되지 않음")


if __name__ == "__main__":
    # 시스템 정보 출력
    show_system_info()
    print()
    
    # 샘플 데이터가 없으면 생성
    if not Path("train_prompts.jsonl").exists():
        create_sample_data()
        print()
    
    # 메인 실행
    main() 