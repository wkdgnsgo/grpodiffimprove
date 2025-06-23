"""
Wandb Logger
============

Wandb를 사용한 실험 추적 및 로깅 모듈입니다.

주요 기능:
1. 학습 메트릭 실시간 로깅
2. 검증 결과 추적
3. 이미지 샘플 업로드
4. 하이퍼파라미터 추적

Author: AI Assistant
Date: 2025-01-22
"""

import logging
from typing import Dict, List, Any, Optional
import os

logger = logging.getLogger(__name__)

class WandbLogger:
    """
    Wandb를 사용한 실험 추적 클래스
    
    이 클래스는:
    1. 실험 초기화 및 설정
    2. 학습/검증 메트릭 로깅
    3. 이미지 및 아티팩트 업로드
    4. 실험 종료 및 정리
    """
    
    def __init__(self, 
                 project: str = "vlm-grpo-training",
                 entity: Optional[str] = None,
                 config: Optional[Dict] = None,
                 tags: Optional[List[str]] = None):
        """
        Wandb Logger 초기화
        
        Args:
            project (str): Wandb 프로젝트 이름
            entity (str, optional): Wandb 엔티티 (팀/사용자)
            config (Dict, optional): 실험 설정
            tags (List[str], optional): 실험 태그
        """
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags or ["vlm", "grpo"]
        
        # Wandb 사용 가능 여부 확인
        self.wandb_available = self._check_wandb_availability()
        
        if self.wandb_available:
            self._initialize_wandb()
        else:
            logger.warning("⚠️ Wandb not available, logging will be skipped")
    
    def _check_wandb_availability(self) -> bool:
        """Wandb 사용 가능 여부 확인"""
        try:
            import wandb
            self.wandb = wandb
            return True
        except ImportError:
            return False
    
    def _initialize_wandb(self):
        """Wandb 초기화"""
        try:
            # Wandb 실행 초기화
            self.run = self.wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                tags=self.tags,
                notes="VLM GRPO training experiment"
            )
            
            logger.info(f"📈 Wandb initialized: {self.run.url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Wandb: {e}")
            self.wandb_available = False
    
    def log_training_metrics(self, metrics: Dict[str, Any]):
        """
        학습 메트릭 로깅
        
        Args:
            metrics (Dict[str, Any]): 로깅할 메트릭들
        """
        if not self.wandb_available:
            return
        
        try:
            # 학습 메트릭 로깅
            log_dict = {
                "train/policy_loss": metrics.get("policy_loss", 0),
                "train/kl_divergence": metrics.get("kl_divergence", 0),
                "train/entropy": metrics.get("entropy", 0),
                "train/average_reward": metrics.get("average_reward", 0),
                "train/iteration": metrics.get("iteration", 0),
                "train/iteration_time": metrics.get("iteration_time", 0),
                "train/total_time": metrics.get("total_time", 0)
            }
            
            self.wandb.log(log_dict)
            logger.debug(f"📊 Training metrics logged: {list(log_dict.keys())}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log training metrics: {e}")
    
    def log_validation_results(self, results: Dict[str, Any]):
        """
        검증 결과 로깅
        
        Args:
            results (Dict[str, Any]): 검증 결과
        """
        if not self.wandb_available:
            return
        
        try:
            # 검증 메트릭 로깅
            log_dict = {
                "val/success_rate": results.get("success_rate", 0),
                "val/avg_clip_score": results.get("avg_clip_score", 0),
                "val/quality_score": results.get("quality_score", 0),
                "val/processing_time": results.get("processing_time", 0),
                "val/total_samples": results.get("total_samples", 0)
            }
            
            # 카테고리별 성능
            for category, stats in results.get("category_results", {}).items():
                log_dict[f"val/category_{category}_success_rate"] = stats.get("success_rate", 0)
            
            # 난이도별 성능
            for difficulty, stats in results.get("difficulty_results", {}).items():
                log_dict[f"val/difficulty_{difficulty}_success_rate"] = stats.get("success_rate", 0)
            
            self.wandb.log(log_dict)
            logger.debug(f"✅ Validation results logged: {list(log_dict.keys())}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log validation results: {e}")
    
    def log_images(self, 
                   images: List[Any], 
                   captions: List[str],
                   step: Optional[int] = None):
        """
        이미지 샘플 로깅
        
        Args:
            images (List[Any]): 로깅할 이미지들
            captions (List[str]): 이미지 캡션들
            step (int, optional): 스텝 번호
        """
        if not self.wandb_available:
            return
        
        try:
            wandb_images = []
            
            for image, caption in zip(images, captions):
                if image is not None:
                    # PIL Image를 Wandb Image로 변환
                    wandb_image = self.wandb.Image(image, caption=caption)
                    wandb_images.append(wandb_image)
            
            if wandb_images:
                log_dict = {"generated_images": wandb_images}
                if step is not None:
                    log_dict["step"] = step
                
                self.wandb.log(log_dict)
                logger.debug(f"🖼️ Logged {len(wandb_images)} images")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log images: {e}")
    
    def log_system_info(self, system_info: Dict[str, Any]):
        """
        시스템 정보 로깅
        
        Args:
            system_info (Dict[str, Any]): 시스템 정보
        """
        if not self.wandb_available:
            return
        
        try:
            # 시스템 정보를 config에 추가
            self.wandb.config.update({
                "system": system_info
            })
            
            logger.debug("💻 System info logged to config")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log system info: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        하이퍼파라미터 로깅
        
        Args:
            hyperparams (Dict[str, Any]): 하이퍼파라미터
        """
        if not self.wandb_available:
            return
        
        try:
            self.wandb.config.update(hyperparams)
            logger.debug(f"⚙️ Hyperparameters logged: {list(hyperparams.keys())}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log hyperparameters: {e}")
    
    def save_artifact(self, 
                     file_path: str, 
                     artifact_name: str,
                     artifact_type: str = "model"):
        """
        아티팩트 저장 (모델, 데이터 등)
        
        Args:
            file_path (str): 저장할 파일 경로
            artifact_name (str): 아티팩트 이름
            artifact_type (str): 아티팩트 타입
        """
        if not self.wandb_available:
            return
        
        try:
            artifact = self.wandb.Artifact(
                name=artifact_name,
                type=artifact_type
            )
            
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            
            logger.info(f"📦 Artifact saved: {artifact_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save artifact: {e}")
    
    def log_custom_metric(self, 
                         metric_name: str, 
                         value: float,
                         step: Optional[int] = None):
        """
        커스텀 메트릭 로깅
        
        Args:
            metric_name (str): 메트릭 이름
            value (float): 메트릭 값
            step (int, optional): 스텝 번호
        """
        if not self.wandb_available:
            return
        
        try:
            log_dict = {metric_name: value}
            if step is not None:
                log_dict["step"] = step
            
            self.wandb.log(log_dict)
            logger.debug(f"📈 Custom metric logged: {metric_name}={value}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log custom metric: {e}")
    
    def log_table(self, 
                  table_name: str, 
                  data: List[List[Any]], 
                  columns: List[str]):
        """
        테이블 데이터 로깅
        
        Args:
            table_name (str): 테이블 이름
            data (List[List[Any]]): 테이블 데이터
            columns (List[str]): 컬럼 이름들
        """
        if not self.wandb_available:
            return
        
        try:
            table = self.wandb.Table(data=data, columns=columns)
            self.wandb.log({table_name: table})
            
            logger.debug(f"📋 Table logged: {table_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log table: {e}")
    
    def finish(self):
        """Wandb 세션 종료"""
        if not self.wandb_available:
            return
        
        try:
            self.wandb.finish()
            logger.info("🏁 Wandb session finished")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to finish Wandb session: {e}")
    
    def get_run_url(self) -> Optional[str]:
        """실행 URL 반환"""
        if self.wandb_available and hasattr(self, 'run'):
            return self.run.url
        return None
    
    def is_available(self) -> bool:
        """Wandb 사용 가능 여부 반환"""
        return self.wandb_available


# 간편한 사용을 위한 헬퍼 함수들
def create_wandb_logger(project: str = "vlm-grpo-training",
                       config: Optional[Dict] = None) -> WandbLogger:
    """
    Wandb Logger 생성 헬퍼 함수
    
    Args:
        project (str): 프로젝트 이름
        config (Dict, optional): 설정
        
    Returns:
        WandbLogger: 생성된 로거
    """
    return WandbLogger(
        project=project,
        config=config,
        tags=["vlm", "grpo", "text-to-image"]
    )


def log_experiment_summary(logger: WandbLogger, 
                          summary: Dict[str, Any]):
    """
    실험 요약 로깅 헬퍼 함수
    
    Args:
        logger (WandbLogger): Wandb 로거
        summary (Dict[str, Any]): 실험 요약
    """
    if logger.is_available():
        logger.log_custom_metric("final/best_reward", summary.get("best_reward", 0))
        logger.log_custom_metric("final/total_time", summary.get("total_time", 0))
        logger.log_custom_metric("final/total_iterations", summary.get("total_iterations", 0))


if __name__ == "__main__":
    # Wandb Logger 테스트 코드
    print("🧪 Wandb Logger Test")
    print("=" * 25)
    
    try:
        # 로거 초기화
        logger_instance = WandbLogger(
            project="vlm-grpo-test",
            config={"test": True},
            tags=["test"]
        )
        
        print(f"✅ Wandb Logger initialized: available={logger_instance.is_available()}")
        
        if logger_instance.is_available():
            print(f"🌐 Run URL: {logger_instance.get_run_url()}")
            
            # 테스트 메트릭 로깅
            print("\n🔄 Testing metric logging:")
            
            # 학습 메트릭
            training_metrics = {
                "policy_loss": 0.5,
                "kl_divergence": 0.01,
                "average_reward": 0.75,
                "iteration": 1
            }
            logger_instance.log_training_metrics(training_metrics)
            print("  ✅ Training metrics logged")
            
            # 검증 결과
            validation_results = {
                "success_rate": 0.8,
                "avg_clip_score": 0.65,
                "category_results": {"basic": {"success_rate": 0.9}}
            }
            logger_instance.log_validation_results(validation_results)
            print("  ✅ Validation results logged")
            
            # 커스텀 메트릭
            logger_instance.log_custom_metric("test_metric", 42.0)
            print("  ✅ Custom metric logged")
            
            # 세션 종료
            logger_instance.finish()
            print("  ✅ Session finished")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\nUsage:")
    print("from integration.wandb_logger import WandbLogger")
    print("logger = WandbLogger(project='my-project')")
    print("logger.log_training_metrics(metrics)") 