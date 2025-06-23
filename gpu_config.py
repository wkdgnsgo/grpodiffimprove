"""
Multi-GPU Configuration for GRPO Training
=========================================

GPU 3개 (1, 2, 3번)를 효율적으로 활용하기 위한 설정 모듈입니다.

GPU 분배 전략:
- GPU 1: QWEN VL 모델 (정책 네트워크)
- GPU 2: Stable Diffusion 3 (이미지 생성)
- GPU 3: CLIP 모델 (보상 계산)

Author: AI Assistant
Date: 2025-01-22
"""

import os
import torch
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MultiGPUConfig:
    """멀티 GPU 설정 및 관리 클래스"""
    
    def __init__(self, gpu_ids: List[int] = [1, 2, 3]):
        """
        Multi-GPU 설정 초기화
        
        Args:
            gpu_ids: 사용할 GPU ID 리스트
        """
        self.gpu_ids = gpu_ids
        self.setup_environment()
        self.device_map = self.create_device_map()
        
    def setup_environment(self):
        """환경 변수 및 CUDA 설정"""
        # GPU 환경 변수 설정
        gpu_env_vars = {
            'CUDA_VISIBLE_DEVICES': ','.join(map(str, self.gpu_ids)),
            'CUDA_LAUNCH_BLOCKING': '0',  # 비동기 실행 활성화
            'TORCH_CUDA_ARCH_LIST': '8.0;8.6;8.9;9.0',  # 최신 아키텍처 지원
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',  # 메모리 관리 최적화
            'NCCL_DEBUG': 'INFO',  # Multi-GPU 통신 디버깅
            'NCCL_IB_DISABLE': '1',  # InfiniBand 비활성화 (필요시)
        }
        
        # 환경 변수 적용
        for key, value in gpu_env_vars.items():
            os.environ[key] = value
            logger.info(f"🔧 {key} = {value}")
        
        logger.info("🚀 Multi-GPU environment configured")
        
    def create_device_map(self) -> Dict[str, str]:
        """모델별 GPU 디바이스 맵 생성"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            return {
                'qwen': 'cpu',
                'sd3': 'cpu', 
                'clip': 'cpu'
            }
        
        available_gpus = torch.cuda.device_count()
        logger.info(f"📱 Available GPUs: {available_gpus}")
        
        if available_gpus >= 3:
            device_map = {
                'qwen': f'cuda:0',  # 실제로는 GPU 1
                'sd3': f'cuda:1',   # 실제로는 GPU 2
                'clip': f'cuda:2'   # 실제로는 GPU 3
            }
        elif available_gpus == 2:
            device_map = {
                'qwen': f'cuda:0',
                'sd3': f'cuda:1', 
                'clip': f'cuda:0'  # QWEN과 공유
            }
        else:
            device_map = {
                'qwen': f'cuda:0',
                'sd3': f'cuda:0',
                'clip': f'cuda:0'
            }
        
        logger.info(f"🗺️ Device mapping: {device_map}")
        return device_map
    
    def get_device(self, model_name: str) -> str:
        """모델별 디바이스 반환"""
        return self.device_map.get(model_name, 'cuda:0')
    
    def print_gpu_status(self):
        """GPU 상태 출력"""
        if not torch.cuda.is_available():
            logger.info("❌ CUDA not available")
            return
            
        logger.info(f"🔍 GPU Status:")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            
            logger.info(f"  GPU {i} ({gpu_name}):")
            logger.info(f"    Total Memory: {memory_total:.1f} GB")
            logger.info(f"    Allocated: {memory_allocated:.1f} GB")
            logger.info(f"    Cached: {memory_cached:.1f} GB")
            logger.info(f"    Free: {memory_total - memory_cached:.1f} GB")
    
    def clear_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            logger.info("🧹 GPU memory cleared")
    
    def set_memory_fraction(self, fraction: float = 0.9):
        """GPU 메모리 사용량 제한"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(fraction, device=i)
            logger.info(f"📊 GPU memory fraction set to {fraction}")

# 전역 설정 인스턴스
gpu_config = MultiGPUConfig([1, 2, 3])

def setup_multi_gpu():
    """멀티 GPU 설정 함수"""
    global gpu_config
    gpu_config.clear_gpu_memory()
    gpu_config.set_memory_fraction(0.85)  # 85% 메모리 사용
    gpu_config.print_gpu_status()
    return gpu_config

def get_device_for_model(model_name: str) -> str:
    """모델별 디바이스 반환 함수"""
    return gpu_config.get_device(model_name) 