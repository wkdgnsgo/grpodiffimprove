"""
Data Loader Utilities
====================

VLM GRPO 학습을 위한 데이터 로딩 및 관리 유틸리티입니다.

주요 기능:
1. 학습/검증 프롬프트 로딩
2. 데이터 전처리
3. 배치 생성
4. 데이터 증강

Author: AI Assistant  
Date: 2025-01-22
"""

import json
import random
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptDataLoader:
    """
    프롬프트 데이터 로딩 및 관리 클래스
    
    이 클래스는:
    1. JSONL 형식의 프롬프트 데이터 로딩
    2. 카테고리 및 난이도별 분류
    3. 배치 생성 및 셔플링
    4. 데이터 통계 제공
    """
    
    def __init__(self, 
                 train_data_path: str = "train_prompts.jsonl",
                 val_data_path: str = "val_prompts.jsonl",
                 batch_shuffle: bool = True):
        """
        Data Loader 초기화
        
        Args:
            train_data_path (str): 학습 데이터 경로
            val_data_path (str): 검증 데이터 경로
            batch_shuffle (bool): 배치 생성 시 셔플 여부
        """
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_shuffle = batch_shuffle
        
        # 데이터 저장소
        self.train_data = []
        self.val_data = []
        
        # 데이터 로딩
        self._load_data()
        
        # 통계 계산
        self._calculate_statistics()
    
    def _load_data(self):
        """데이터 파일들을 로딩하는 내부 메서드"""
        try:
            # 학습 데이터 로딩
            if Path(self.train_data_path).exists():
                self.train_data = self._load_jsonl(self.train_data_path)
                logger.info(f"📥 Loaded {len(self.train_data)} training prompts")
            else:
                logger.warning(f"⚠️ Training data not found: {self.train_data_path}")
            
            # 검증 데이터 로딩  
            if Path(self.val_data_path).exists():
                self.val_data = self._load_jsonl(self.val_data_path)
                logger.info(f"📥 Loaded {len(self.val_data)} validation prompts")
            else:
                logger.warning(f"⚠️ Validation data not found: {self.val_data_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """
        JSONL 파일 로딩
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            List[Dict]: 로딩된 데이터
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        data.append(item)
            
            logger.debug(f"✅ Loaded {len(data)} items from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            return []
    
    def _calculate_statistics(self):
        """데이터 통계 계산"""
        self.stats = {
            'train_count': len(self.train_data),
            'val_count': len(self.val_data),
            'categories': {},
            'difficulties': {},
            'avg_prompt_length': 0
        }
        
        # 전체 데이터 분석
        all_data = self.train_data + self.val_data
        
        if all_data:
            # 카테고리별 통계
            for item in all_data:
                category = item.get('category', 'unknown')
                difficulty = item.get('difficulty', 'unknown')
                
                self.stats['categories'][category] = self.stats['categories'].get(category, 0) + 1
                self.stats['difficulties'][difficulty] = self.stats['difficulties'].get(difficulty, 0) + 1
            
            # 평균 프롬프트 길이
            prompt_lengths = [len(item.get('user_prompt', '')) for item in all_data]
            self.stats['avg_prompt_length'] = sum(prompt_lengths) / len(prompt_lengths)
        
        logger.info(f"📊 Data statistics: {self.stats}")
    
    def get_training_batch(self, batch_size: int, shuffle: Optional[bool] = None) -> List[str]:
        """
        학습용 배치 생성
        
        Args:
            batch_size (int): 배치 크기
            shuffle (bool, optional): 셔플 여부 (None이면 기본 설정 사용)
            
        Returns:
            List[str]: 프롬프트 배치
        """
        if not self.train_data:
            logger.warning("⚠️ No training data available")
            return []
        
        # 셔플 설정 결정
        if shuffle is None:
            shuffle = self.batch_shuffle
        
        # 데이터 복사 및 셔플
        data = self.train_data.copy()
        if shuffle:
            random.shuffle(data)
        
        # 배치 추출
        batch = data[:batch_size]
        prompts = [item.get('user_prompt', '') for item in batch]
        
        logger.debug(f"📦 Generated training batch: {len(prompts)} prompts")
        return prompts
    
    def get_validation_data(self) -> List[Dict]:
        """
        전체 검증 데이터 반환
        
        Returns:
            List[Dict]: 검증 데이터
        """
        return self.val_data.copy()
    
    def get_category_batch(self, category: str, batch_size: int) -> List[str]:
        """
        특정 카테고리의 배치 생성
        
        Args:
            category (str): 카테고리 이름
            batch_size (int): 배치 크기
            
        Returns:
            List[str]: 카테고리별 프롬프트 배치
        """
        # 카테고리 필터링
        category_data = [
            item for item in self.train_data 
            if item.get('category') == category
        ]
        
        if not category_data:
            logger.warning(f"⚠️ No data found for category: {category}")
            return []
        
        # 배치 생성
        random.shuffle(category_data)
        batch = category_data[:batch_size]
        prompts = [item.get('user_prompt', '') for item in batch]
        
        logger.debug(f"📦 Generated {category} batch: {len(prompts)} prompts")
        return prompts
    
    def get_difficulty_batch(self, difficulty: str, batch_size: int) -> List[str]:
        """
        특정 난이도의 배치 생성
        
        Args:
            difficulty (str): 난이도 레벨
            batch_size (int): 배치 크기
            
        Returns:
            List[str]: 난이도별 프롬프트 배치
        """
        # 난이도 필터링
        difficulty_data = [
            item for item in self.train_data 
            if item.get('difficulty') == difficulty
        ]
        
        if not difficulty_data:
            logger.warning(f"⚠️ No data found for difficulty: {difficulty}")
            return []
        
        # 배치 생성
        random.shuffle(difficulty_data)
        batch = difficulty_data[:batch_size]
        prompts = [item.get('user_prompt', '') for item in batch]
        
        logger.debug(f"📦 Generated {difficulty} batch: {len(prompts)} prompts")
        return prompts
    
    def get_balanced_batch(self, batch_size: int) -> List[str]:
        """
        카테고리별로 균형잡힌 배치 생성
        
        Args:
            batch_size (int): 배치 크기
            
        Returns:
            List[str]: 균형잡힌 프롬프트 배치
        """
        if not self.train_data:
            return []
        
        # 카테고리별 데이터 분류
        category_data = {}
        for item in self.train_data:
            category = item.get('category', 'unknown')
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(item)
        
        # 각 카테고리에서 균등하게 선택
        balanced_batch = []
        categories = list(category_data.keys())
        per_category = max(1, batch_size // len(categories))
        
        for category in categories:
            if len(balanced_batch) >= batch_size:
                break
                
            category_items = random.sample(
                category_data[category], 
                min(per_category, len(category_data[category]))
            )
            balanced_batch.extend(category_items)
        
        # 부족한 경우 랜덤하게 추가
        if len(balanced_batch) < batch_size:
            remaining = batch_size - len(balanced_batch)
            additional = random.sample(self.train_data, remaining)
            balanced_batch.extend(additional)
        
        # 배치 크기 맞추기
        balanced_batch = balanced_batch[:batch_size]
        prompts = [item.get('user_prompt', '') for item in balanced_batch]
        
        logger.debug(f"📦 Generated balanced batch: {len(prompts)} prompts")
        return prompts
    
    def get_statistics(self) -> Dict:
        """데이터 통계 반환"""
        return self.stats.copy()
    
    def save_batch_results(self, 
                          prompts: List[str], 
                          enhanced_prompts: List[str],
                          rewards: List[float],
                          save_path: str):
        """
        배치 결과 저장
        
        Args:
            prompts (List[str]): 원본 프롬프트들
            enhanced_prompts (List[str]): 개선된 프롬프트들  
            rewards (List[float]): 보상 값들
            save_path (str): 저장 경로
        """
        try:
            results = []
            for i in range(len(prompts)):
                result = {
                    'user_prompt': prompts[i],
                    'enhanced_prompt': enhanced_prompts[i] if i < len(enhanced_prompts) else '',
                    'reward': rewards[i] if i < len(rewards) else 0.0,
                    'improvement': len(enhanced_prompts[i]) - len(prompts[i]) if i < len(enhanced_prompts) else 0
                }
                results.append(result)
            
            # JSONL 형식으로 저장
            with open(save_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Batch results saved: {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save batch results: {e}")

# 호환성을 위한 별칭 생성
DataLoader = PromptDataLoader

def create_sample_data(train_path: str = "train_prompts.jsonl", 
                      val_path: str = "val_prompts.jsonl"):
    """
    샘플 데이터 생성 함수 (테스트용)
    
    Args:
        train_path (str): 학습 데이터 저장 경로
        val_path (str): 검증 데이터 저장 경로
    """
    # 샘플 학습 데이터
    train_samples = [
        {"user_prompt": "a cat", "category": "basic", "difficulty": "easy"},
        {"user_prompt": "sunset", "category": "basic", "difficulty": "easy"},
        {"user_prompt": "beautiful woman", "category": "complex", "difficulty": "medium"},
        {"user_prompt": "mountain landscape", "category": "photography", "difficulty": "medium"},
        {"user_prompt": "abstract art", "category": "creative", "difficulty": "hard"},
    ]
    
    # 샘플 검증 데이터
    val_samples = [
        {"user_prompt": "dog", "category": "basic", "difficulty": "easy"},
        {"user_prompt": "city skyline", "category": "photography", "difficulty": "medium"},
    ]
    
    # 파일 저장
    try:
        with open(train_path, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Sample data created: {train_path}, {val_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")


if __name__ == "__main__":
    # Data Loader 테스트 코드
    print("🧪 Data Loader Test")
    print("=" * 25)
    
    try:
        # 샘플 데이터 생성
        create_sample_data()
        
        # 데이터 로더 초기화
        loader = PromptDataLoader()
        
        print("✅ Data Loader initialized successfully")
        print(f"📊 Statistics: {loader.get_statistics()}")
        
        # 배치 생성 테스트
        print("\n🔄 Testing batch generation:")
        
        # 일반 배치
        batch = loader.get_training_batch(batch_size=3)
        print(f"  Training batch: {batch}")
        
        # 카테고리별 배치
        category_batch = loader.get_category_batch("basic", batch_size=2)
        print(f"  Basic category batch: {category_batch}")
        
        # 균형잡힌 배치
        balanced_batch = loader.get_balanced_batch(batch_size=4)
        print(f"  Balanced batch: {balanced_batch}")
        
        # 검증 데이터
        val_data = loader.get_validation_data()
        print(f"  Validation data: {len(val_data)} items")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\nUsage:")
    print("from utils.data_loader import PromptDataLoader")
    print("loader = PromptDataLoader()")
    print("batch = loader.get_training_batch(batch_size=4)") 