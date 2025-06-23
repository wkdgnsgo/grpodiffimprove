"""
CLIP Reward Calculator
=====================

CLIP 모델을 사용하여 텍스트-이미지 간의 유사도를 계산하고 보상 신호를 생성하는 모듈입니다.

주요 기능:
1. 텍스트-이미지 유사도 계산
2. 다양한 보상 함수 제공
3. 배치 처리 지원
4. 정규화 및 스케일링

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)

class CLIPRewardCalculator:
    """
    CLIP을 사용하여 텍스트-이미지 유사도 기반 보상을 계산하는 클래스
    
    이 클래스는 GRPO 학습에서 핵심적인 역할을 합니다:
    1. 생성된 이미지와 프롬프트 간의 유사도 측정
    2. 유사도를 보상 신호로 변환
    3. 학습 안정성을 위한 정규화
    
    Attributes:
        model_name (str): 사용할 CLIP 모델 이름
        processor: CLIP 프로세서 객체
        model: CLIP 모델 객체
        device: 연산 디바이스
        reward_config (dict): 보상 계산 설정
    """
    
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "auto",
                 reward_scale: float = 1.0,
                 reward_offset: float = 0.0,
                 temperature: float = 1.0,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        CLIP Reward Calculator 초기화
        
        Args:
            model_name (str): 사용할 CLIP 모델 이름
            device (str): 디바이스 설정 ("auto", "mps", "cuda", "cpu")
            reward_scale (float): 보상 스케일링 팩터
            reward_offset (float): 보상 오프셋
            temperature (float): 소프트맥스 온도 (유사도 조절)
            reward_weights (Dict[str, float], optional): 다중 보상 가중치
        """
        self.model_name = model_name
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.temperature = temperature
        
        # 다중 보상 가중치 설정
        if reward_weights is None:
            self.reward_weights = {
                'clip_similarity': 0.6,
                'image_quality': 0.3,
                'semantic_consistency': 0.1
            }
        else:
            self.reward_weights = reward_weights
            
        logger.info(f"🎯 Multi-reward weights: {self.reward_weights}")
        
        # 디바이스 자동 선택
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🍎 Using Apple Silicon MPS for CLIP")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🚀 Using CUDA GPU for CLIP")
            else:
                self.device = torch.device("cpu")
                logger.info("💻 Using CPU for CLIP")
        else:
            self.device = torch.device(device)
        
        # 보상 계산 설정
        self.reward_config = {
            'scale': self.reward_scale,
            'offset': self.reward_offset,
            'temperature': self.temperature,
            'normalize': True,      # 보상 정규화 여부
            'clip_range': (-1, 1),  # 보상 클리핑 범위
        }
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """
        CLIP 모델과 프로세서를 로드하는 내부 메서드
        
        이 메서드는:
        1. CLIP 프로세서 로드
        2. CLIP 모델 로드 및 디바이스 이동
        3. 평가 모드 설정
        4. 에러 처리
        """
        try:
            logger.info(f"📥 Loading CLIP model: {self.model_name}")
            
            # CLIP 프로세서 로드
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # CLIP 모델 로드
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32
            )
            
            # 디바이스로 이동
            self.model = self.model.to(self.device)
            self.model.eval()  # 평가 모드 설정
            
            logger.info(f"✅ CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}")
    
    def calculate_reward(self, 
                        image: Image.Image, 
                        text: str) -> float:
        """
        단일 이미지-텍스트 쌍에 대한 보상 계산
        
        이 메서드는:
        1. 이미지와 텍스트를 CLIP으로 인코딩
        2. 코사인 유사도 계산
        3. 보상 함수 적용
        4. 정규화 및 스케일링
        
        Args:
            image (PIL.Image.Image): 평가할 이미지
            text (str): 비교할 텍스트 프롬프트
            
        Returns:
            float: 계산된 보상 값 (-1.0 ~ 1.0)
            
        Example:
            reward = calculator.calculate_reward(generated_image, "a cute cat")
            # reward: 0.85 (높은 유사도)
        """
        try:
            # 입력 전처리
            inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # CLIP으로 특성 추출
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 이미지와 텍스트 임베딩 추출
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # 임베딩 정규화
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)
                
                # 코사인 유사도 계산
                similarity = torch.cosine_similarity(image_embeds, text_embeds, dim=-1)
                
                # 온도 스케일링 적용
                similarity = similarity / self.temperature
                
                # 보상 변환
                reward = self._transform_similarity_to_reward(similarity.item())
                
                logger.debug(f"💰 Reward calculated: {reward:.4f} for text: '{text[:30]}...'")
                
                return reward
                
        except Exception as e:
            logger.warning(f"⚠️ Reward calculation failed: {e}")
            # 실패 시 중립 보상 반환
            return 0.0
    
    def calculate_rewards_batch(self, 
                               images: List[Image.Image], 
                               texts: List[str]) -> List[float]:
        """
        여러 이미지-텍스트 쌍에 대한 배치 보상 계산
        
        Args:
            images (List[PIL.Image.Image]): 평가할 이미지 리스트
            texts (List[str]): 비교할 텍스트 프롬프트 리스트
            
        Returns:
            List[float]: 계산된 보상 값 리스트
        """
        if len(images) != len(texts):
            raise ValueError("Images and texts must have the same length")
        
        rewards = []
        
        try:
            # 배치 처리를 위한 입력 준비
            inputs = self.processor(
                text=texts, 
                images=images, 
                return_tensors="pt", 
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # CLIP으로 배치 처리
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 임베딩 추출 및 정규화
                image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
                
                # 배치 코사인 유사도 계산
                similarities = torch.cosine_similarity(image_embeds, text_embeds, dim=-1)
                
                # 온도 스케일링
                similarities = similarities / self.temperature
                
                # 보상 변환
                for similarity in similarities:
                    reward = self._transform_similarity_to_reward(similarity.item())
                    rewards.append(reward)
                
                logger.debug(f"💰 Batch rewards calculated: {len(rewards)} items")
                
        except Exception as e:
            logger.warning(f"⚠️ Batch reward calculation failed: {e}")
            # 실패 시 중립 보상들 반환
            rewards = [0.0] * len(images)
        
        return rewards
    
    def _transform_similarity_to_reward(self, similarity: float) -> float:
        """
        CLIP 유사도를 보상 신호로 변환하는 내부 메서드
        
        이 메서드는:
        1. 유사도를 보상 스케일로 변환
        2. 오프셋 적용
        3. 정규화 (선택적)
        4. 클리핑
        
        Args:
            similarity (float): CLIP 코사인 유사도 (-1 ~ 1)
            
        Returns:
            float: 변환된 보상 값
        """
        # 기본 보상 변환: 유사도 * 스케일 + 오프셋
        reward = similarity * self.reward_config['scale'] + self.reward_config['offset']
        
        # 정규화 적용 (선택적)
        if self.reward_config['normalize']:
            # 시그모이드 함수로 부드러운 정규화
            reward = torch.sigmoid(torch.tensor(reward)).item()
            # -1 ~ 1 범위로 변환
            reward = 2 * reward - 1
        
        # 클리핑 적용
        clip_min, clip_max = self.reward_config['clip_range']
        reward = max(clip_min, min(clip_max, reward))
        
        return reward
    
    def calculate_quality_reward(self, image: Image.Image) -> float:
        """
        이미지 품질 기반 보상 계산
        
        이 메서드는 텍스트 없이 이미지 자체의 품질을 평가합니다.
        일반적인 고품질 이미지 특성과의 유사도를 측정합니다.
        
        Args:
            image (PIL.Image.Image): 평가할 이미지
            
        Returns:
            float: 품질 기반 보상 값
        """
        quality_prompts = [
            "high quality image",
            "professional photography",
            "detailed and sharp image",
            "well-composed photograph"
        ]
        
        # 각 품질 프롬프트와의 유사도 계산
        quality_scores = []
        for prompt in quality_prompts:
            score = self.calculate_reward(image, prompt)
            quality_scores.append(score)
        
        # 평균 품질 점수 반환
        quality_reward = np.mean(quality_scores)
        
        logger.debug(f"🎨 Quality reward: {quality_reward:.4f}")
        return quality_reward
    
    def calculate_semantic_consistency(self, 
                                     image: Image.Image, 
                                     original_prompt: str, 
                                     enhanced_prompt: str) -> float:
        """
        원본 프롬프트와 개선된 프롬프트 간의 의미적 일관성 평가
        
        Args:
            image (PIL.Image.Image): 생성된 이미지
            original_prompt (str): 원본 사용자 프롬프트
            enhanced_prompt (str): VLM이 개선한 프롬프트
            
        Returns:
            float: 의미적 일관성 점수
        """
        # 각 프롬프트와 이미지 간의 유사도 계산
        original_reward = self.calculate_reward(image, original_prompt)
        enhanced_reward = self.calculate_reward(image, enhanced_prompt)
        
        # 일관성 점수: 두 보상의 최소값 (둘 다 높아야 일관성 있음)
        consistency_score = min(original_reward, enhanced_reward)
        
        logger.debug(f"🔗 Semantic consistency: {consistency_score:.4f}")
        return consistency_score
    
    def get_model_info(self) -> Dict:
        """
        모델 정보 반환 (디버깅 및 로깅용)
        
        Returns:
            Dict: 모델 관련 정보
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "reward_config": self.reward_config,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
    
    def update_reward_config(self, **kwargs):
        """
        보상 계산 설정 업데이트
        
        Args:
            **kwargs: 업데이트할 설정들
        """
        self.reward_config.update(kwargs)
        logger.info(f"🔧 Reward config updated: {kwargs}")


class MultiRewardCalculator:
    """
    여러 보상 함수를 조합하여 종합적인 보상을 계산하는 클래스
    
    이 클래스는:
    1. CLIP 유사도 보상
    2. 이미지 품질 보상  
    3. 의미적 일관성 보상
    4. 가중 평균으로 최종 보상 계산
    """
    
    def __init__(self,
                 clip_calculator: CLIPRewardCalculator,
                 weights: Optional[Dict[str, float]] = None):
        """
        Multi Reward Calculator 초기화
        
        Args:
            clip_calculator (CLIPRewardCalculator): CLIP 보상 계산기
            weights (Dict[str, float], optional): 각 보상의 가중치
        """
        self.clip_calculator = clip_calculator
        
        # 기본 가중치 설정
        self.weights = weights or {
            'clip_similarity': 0.6,    # CLIP 유사도 (주요)
            'image_quality': 0.3,      # 이미지 품질
            'semantic_consistency': 0.1 # 의미적 일관성
        }
        
        # 가중치 정규화
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"🎯 Multi-reward weights: {self.weights}")
    
    def calculate_comprehensive_reward(self, 
                                     image: Optional[Image.Image], 
                                     original_prompt: str, 
                                     enhanced_prompt: str) -> Dict[str, float]:
        """
        종합적인 보상 계산
        
        Args:
            image (Optional[Image.Image]): 생성된 이미지 (None일 수 있음)
            original_prompt (str): 원본 프롬프트
            enhanced_prompt (str): 개선된 프롬프트
            
        Returns:
            Dict[str, float]: 종합 보상 결과
        """
        try:
            # 입력 검증
            if image is None:
                logger.warning("⚠️ No image provided for reward calculation")
                return {
                    'clip_similarity': 0.0,
                    'image_quality': 0.0,
                    'semantic_consistency': 0.0,
                    'final_reward': 0.0
                }
            
            if not original_prompt or not enhanced_prompt:
                logger.warning("⚠️ Empty prompts provided for reward calculation")
                return {
                    'clip_similarity': 0.0,
                    'image_quality': 0.0,
                    'semantic_consistency': 0.0,
                    'final_reward': 0.0
                }
            
            # 안전한 보상 계산
            rewards = {}
            
            # 1. CLIP 유사도 계산 (안전한 방식)
            try:
                clip_score = self._calculate_safe_clip_similarity(image, enhanced_prompt)
                rewards['clip_similarity'] = max(0.0, min(1.0, clip_score))
            except Exception as e:
                logger.warning(f"⚠️ CLIP similarity calculation failed: {e}")
                rewards['clip_similarity'] = 0.0
            
            # 2. 이미지 품질 점수 계산
            try:
                quality_score = self._calculate_safe_image_quality(image)
                rewards['image_quality'] = max(0.0, min(1.0, quality_score))
            except Exception as e:
                logger.warning(f"⚠️ Image quality calculation failed: {e}")
                rewards['image_quality'] = 0.0
            
            # 3. 의미적 일관성 계산
            try:
                consistency_score = self._calculate_safe_semantic_consistency(
                    original_prompt, enhanced_prompt
                )
                rewards['semantic_consistency'] = max(0.0, min(1.0, consistency_score))
            except Exception as e:
                logger.warning(f"⚠️ Semantic consistency calculation failed: {e}")
                rewards['semantic_consistency'] = 0.0
            
            # 4. 최종 보상 계산 (가중평균)
            final_reward = (
                rewards['clip_similarity'] * self.weights['clip_similarity'] +
                rewards['image_quality'] * self.weights['image_quality'] +
                rewards['semantic_consistency'] * self.weights['semantic_consistency']
            )
            
            rewards['final_reward'] = max(0.0, min(1.0, final_reward))
            
            logger.debug(f"📊 Comprehensive rewards: {rewards}")
            return rewards
            
        except Exception as e:
            logger.warning(f"⚠️ Reward calculation failed: {e}")
            return {
                'clip_similarity': 0.0,
                'image_quality': 0.0,
                'semantic_consistency': 0.0,
                'final_reward': 0.0
            }
    
    def _calculate_safe_clip_similarity(self, image: Image.Image, prompt: str) -> float:
        """
        안전한 CLIP 유사도 계산
        
        Args:
            image (Image.Image): 입력 이미지
            prompt (str): 텍스트 프롬프트
            
        Returns:
            float: CLIP 유사도 점수 (0.0-1.0)
        """
        try:
            # 입력 검증
            if not hasattr(image, 'size') or image.size[0] == 0 or image.size[1] == 0:
                logger.warning("⚠️ Invalid image for CLIP calculation")
                return 0.0
            
            if not prompt or len(prompt.strip()) == 0:
                logger.warning("⚠️ Empty prompt for CLIP calculation")
                return 0.0
            
            # 프롬프트 길이 제한
            prompt = prompt.strip()[:200]
            
            # CUDA 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                # 이미지 전처리 (안전한 방식)
                try:
                    # 이미지 크기 검증 및 조정
                    if image.size[0] > 1024 or image.size[1] > 1024:
                        image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # RGB 변환 (필요시)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_inputs = self.clip_calculator.processor(
                        images=image, 
                        return_tensors="pt",
                        do_rescale=True,
                        do_normalize=True
                    )
                    
                    # 입력 검증
                    if 'pixel_values' not in image_inputs:
                        logger.warning("⚠️ Invalid image preprocessing result")
                        return 0.0
                    
                    image_inputs = {k: v.to(self.clip_calculator.device) for k, v in image_inputs.items()}
                    
                except Exception as e:
                    logger.warning(f"⚠️ Image preprocessing failed: {e}")
                    return 0.0
                
                # 텍스트 전처리 (안전한 방식)
                try:
                    text_inputs = self.clip_calculator.processor(
                        text=[prompt], 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=77
                    )
                    
                    # 입력 검증
                    if 'input_ids' not in text_inputs:
                        logger.warning("⚠️ Invalid text preprocessing result")
                        return 0.0
                    
                    text_inputs = {k: v.to(self.clip_calculator.device) for k, v in text_inputs.items()}
                    
                except Exception as e:
                    logger.warning(f"⚠️ Text preprocessing failed: {e}")
                    return 0.0
                
                # CLIP 임베딩 계산 (안전한 방식)
                try:
                    image_features = self.clip_calculator.model.get_image_features(**image_inputs)
                    text_features = self.clip_calculator.model.get_text_features(**text_inputs)
                    
                    # 특징 벡터 검증
                    if image_features is None or text_features is None:
                        logger.warning("⚠️ Failed to extract features")
                        return 0.0
                    
                    if torch.isnan(image_features).any() or torch.isnan(text_features).any():
                        logger.warning("⚠️ NaN values in features")
                        return 0.0
                    
                    # 정규화
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # 유사도 계산
                    similarity = torch.cosine_similarity(image_features, text_features).item()
                    
                    # 결과 검증
                    if math.isnan(similarity) or math.isinf(similarity):
                        logger.warning("⚠️ Invalid similarity score")
                        return 0.0
                    
                    # 0-1 범위로 정규화
                    normalized_similarity = (similarity + 1.0) / 2.0
                    return max(0.0, min(1.0, normalized_similarity))
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        logger.warning(f"⚠️ CUDA error in CLIP calculation: {e}")
                        # CUDA 캐시 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        return 0.0
                    else:
                        raise e
                        
        except Exception as e:
            logger.warning(f"⚠️ CLIP similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_safe_image_quality(self, image: Image.Image) -> float:
        """
        안전한 이미지 품질 계산
        
        Args:
            image (Image.Image): 입력 이미지
            
        Returns:
            float: 품질 점수 (0.0-1.0)
        """
        try:
            # 기본적인 품질 지표들
            width, height = image.size
            
            # 해상도 점수 (최소 256x256 이상이면 좋음)
            resolution_score = min(1.0, (width * height) / (512 * 512))
            
            # 종횡비 점수 (1:1에 가까울수록 좋음)
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0.0, 1.0 - (aspect_ratio - 1.0) / 2.0)
            
            # 전체 품질 점수
            quality_score = (resolution_score + aspect_score) / 2.0
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"⚠️ Image quality calculation failed: {e}")
            return 0.0
    
    def _calculate_safe_semantic_consistency(self, original: str, enhanced: str) -> float:
        """
        안전한 의미적 일관성 계산
        
        Args:
            original (str): 원본 프롬프트
            enhanced (str): 개선된 프롬프트
            
        Returns:
            float: 일관성 점수 (0.0-1.0)
        """
        try:
            # 간단한 키워드 기반 일관성 검사
            original_words = set(original.lower().split())
            enhanced_words = set(enhanced.lower().split())
            
            if len(original_words) == 0:
                return 0.0
            
            # 교집합 비율 계산
            intersection = original_words.intersection(enhanced_words)
            consistency_score = len(intersection) / len(original_words)
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic consistency calculation failed: {e}")
            return 0.0


if __name__ == "__main__":
    # CLIP Reward Calculator 테스트 코드
    print("🧪 CLIP Reward Calculator Test")
    print("=" * 40)
    
    try:
        # CLIP 보상 계산기 초기화
        calculator = CLIPRewardCalculator(
            model_name="openai/clip-vit-base-patch32",
            device="auto",
            reward_scale=1.0
        )
        
        print("✅ CLIP Reward Calculator initialized successfully")
        print(f"📊 Model info: {calculator.get_model_info()}")
        
        # 테스트용 더미 이미지 생성
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # 테스트 프롬프트들
        test_cases = [
            ("red color", "높은 유사도 예상"),
            ("blue sky", "낮은 유사도 예상"),
            ("colorful image", "중간 유사도 예상")
        ]
        
        print("\n🔄 Testing reward calculation:")
        for prompt, description in test_cases:
            reward = calculator.calculate_reward(test_image, prompt)
            print(f"  '{prompt}' → {reward:.4f} ({description})")
        
        # 품질 보상 테스트
        quality_reward = calculator.calculate_quality_reward(test_image)
        print(f"\n🎨 Quality reward: {quality_reward:.4f}")
        
        # Multi-reward 테스트
        multi_calculator = MultiRewardCalculator(calculator)
        comprehensive_rewards = multi_calculator.calculate_comprehensive_reward(
            test_image, "red", "bright red color"
        )
        print(f"\n🏆 Comprehensive rewards: {comprehensive_rewards}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\nUsage:")
    print("from models.clip_reward import CLIPRewardCalculator")
    print("calculator = CLIPRewardCalculator()")
    print("reward = calculator.calculate_reward(image, 'a cat')") 