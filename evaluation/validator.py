"""
Validation Evaluator
===================

VLM GRPO 학습의 검증 및 평가를 수행하는 모듈입니다.

주요 기능:
1. 실시간 성능 모니터링
2. 다양한 평가 메트릭
3. 카테고리/난이도별 분석
4. 시각화 및 리포트 생성

Author: AI Assistant
Date: 2025-01-22
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import time

logger = logging.getLogger(__name__)

class ValidationEvaluator:
    """
    검증 평가를 수행하는 클래스
    
    이 클래스는:
    1. VLM 성능 평가
    2. 이미지 생성 품질 평가
    3. 전체 파이프라인 성능 측정
    4. 상세한 분석 리포트 생성
    """
    
    def __init__(self, vlm, sd_generator, clip_calculator):
        """
        Validator 초기화
        
        Args:
            vlm: VLM 모델
            sd_generator: SD3 생성기
            clip_calculator: CLIP 보상 계산기
        """
        self.vlm = vlm
        self.sd_generator = sd_generator
        self.clip_calculator = clip_calculator
        
        # 평가 통계
        self.evaluation_history = []
        
        logger.info("✅ Validation Evaluator initialized")
    
    def evaluate_batch(self, validation_data: List[Dict], save_images: bool = True, output_dir: str = "vlm_grpo_results", iteration: int = 0) -> Dict[str, Any]:
        """
        배치 데이터에 대한 종합 평가
        
        Args:
            validation_data (List[Dict]): 검증 데이터
            save_images (bool): 이미지 저장 여부
            output_dir (str): 출력 디렉토리
            iteration (int): 현재 반복 횟수
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        logger.info(f"🔍 Evaluating batch of {len(validation_data)} items")
        
        # 이미지 저장 디렉토리 설정
        if save_images:
            import os
            images_dir = os.path.join(output_dir, f"validation_images_iter_{iteration}")
            os.makedirs(images_dir, exist_ok=True)
            logger.info(f"📁 Saving validation images to: {images_dir}")
        
        results = {
            'timestamp': time.time(),
            'iteration': iteration,
            'total_samples': len(validation_data),
            'success_rate': 0.0,
            'avg_clip_score': 0.0,
            'quality_score': 0.0,
            'processing_time': 0.0,
            'category_results': {},
            'difficulty_results': {},
            'detailed_results': [],
            'saved_images': [] if save_images else None
        }
        
        start_time = time.time()
        successful_evaluations = 0
        clip_scores = []
        quality_scores = []
        
        for idx, item in enumerate(validation_data):
            try:
                # 개별 아이템 평가
                item_result = self._evaluate_single_item(item, save_images, images_dir if save_images else None, idx)
                results['detailed_results'].append(item_result)
                
                if item_result['success']:
                    successful_evaluations += 1
                    clip_scores.append(item_result['clip_score'])
                    quality_scores.append(item_result['quality_score'])
                
                # 저장된 이미지 정보 추가
                if save_images and 'saved_image_path' in item_result:
                    saved_image_info = {
                        'prompt': item.get('user_prompt', ''),
                        'enhanced_prompt': item_result.get('enhanced_prompt', ''),
                        'image_path': item_result['saved_image_path'],
                        'clip_score': item_result.get('clip_score', 0.0)
                    }
                    
                    # 추가 경로 정보가 있다면 포함
                    if 'saved_original_path' in item_result:
                        saved_image_info['saved_original_path'] = item_result['saved_original_path']
                    if 'saved_prompts_path' in item_result:
                        saved_image_info['saved_prompts_path'] = item_result['saved_prompts_path']
                    
                    results['saved_images'].append(saved_image_info)
                
                # 카테고리별 통계
                category = item.get('category', 'unknown')
                if category not in results['category_results']:
                    results['category_results'][category] = {'count': 0, 'success': 0}
                results['category_results'][category]['count'] += 1
                if item_result['success']:
                    results['category_results'][category]['success'] += 1
                
                # 난이도별 통계
                difficulty = item.get('difficulty', 'unknown')
                if difficulty not in results['difficulty_results']:
                    results['difficulty_results'][difficulty] = {'count': 0, 'success': 0}
                results['difficulty_results'][difficulty]['count'] += 1
                if item_result['success']:
                    results['difficulty_results'][difficulty]['success'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to evaluate item {idx}: {e}")
                results['detailed_results'].append({
                    'prompt': item.get('user_prompt', ''),
                    'success': False,
                    'error': str(e)
                })
        
        # 전체 통계 계산
        results['success_rate'] = successful_evaluations / len(validation_data)
        results['avg_clip_score'] = np.mean(clip_scores) if clip_scores else 0.0
        results['quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
        results['processing_time'] = time.time() - start_time
        
        # 카테고리별 성공률 계산
        for category, stats in results['category_results'].items():
            stats['success_rate'] = stats['success'] / stats['count'] if stats['count'] > 0 else 0
        
        # 난이도별 성공률 계산
        for difficulty, stats in results['difficulty_results'].items():
            stats['success_rate'] = stats['success'] / stats['count'] if stats['count'] > 0 else 0
        
        # 히스토리에 추가
        self.evaluation_history.append(results)
        
        logger.info(f"✅ Evaluation completed: {results['success_rate']:.2%} success rate")
        if save_images:
            logger.info(f"💾 Saved {len(results.get('saved_images', []))} validation images")
        
        return results
    
    def _evaluate_single_item(self, item: Dict, save_image: bool = False, images_dir: str = None, idx: int = 0) -> Dict[str, Any]:
        """
        단일 아이템 평가
        
        Args:
            item (Dict): 평가할 아이템
            save_image (bool): 이미지 저장 여부
            images_dir (str): 이미지 저장 디렉토리
            idx (int): 아이템 인덱스
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        user_prompt = item.get('user_prompt', '')
        category = item.get('category', 'unknown')
        difficulty = item.get('difficulty', 'unknown')
        
        result = {
            'prompt': user_prompt,
            'category': category,
            'difficulty': difficulty,
            'success': False,
            'enhanced_prompt': '',
            'clip_score': 0.0,
            'quality_score': 0.0,
            'improvement_length': 0,
            'processing_time': 0.0
        }
        
        try:
            start_time = time.time()
            
            # 1. VLM으로 프롬프트 개선
            enhanced_prompt = self.vlm.enhance_prompt(user_prompt)
            result['enhanced_prompt'] = enhanced_prompt
            result['improvement_length'] = len(enhanced_prompt) - len(user_prompt)
            
            # 2. SD3로 이미지 생성
            image = self.sd_generator.generate_image(enhanced_prompt)
            
            # 3. 이미지 저장 (요청된 경우)
            if save_image and images_dir and image is not None:
                try:
                    import os
                    from PIL import Image as PILImage
                    
                    # 안전한 파일명 생성
                    safe_prompt = "".join(c for c in user_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_prompt = safe_prompt[:50]  # 길이 제한
                    if not safe_prompt:
                        safe_prompt = f"prompt_{idx}"
                    
                    # 원본 prompt로 이미지 생성 및 저장
                    image_original = self.sd_generator.generate_image(user_prompt)
                    if image_original and hasattr(image_original, 'save'):
                        original_filename = f"{idx:03d}_{safe_prompt}_original.png"
                        original_path = os.path.join(images_dir, original_filename)
                        image_original.save(original_path)
                        result['saved_original_path'] = original_path
                        logger.debug(f"💾 Saved original image: {original_path}")
                    
                    # Enhanced prompt로 이미지 저장
                    enhanced_filename = f"{idx:03d}_{safe_prompt}_enhanced.png"
                    enhanced_path = os.path.join(images_dir, enhanced_filename)
                    
                    # 이미지 저장
                    if hasattr(image, 'save'):  # PIL Image인 경우
                        image.save(enhanced_path)
                        result['saved_image_path'] = enhanced_path
                        logger.debug(f"💾 Saved enhanced image: {enhanced_path}")
                        
                        # 프롬프트 텍스트 파일도 저장
                        prompt_filename = f"{idx:03d}_{safe_prompt}_prompts.txt"
                        prompt_path = os.path.join(images_dir, prompt_filename)
                        with open(prompt_path, 'w', encoding='utf-8') as f:
                            f.write(f"Original Prompt:\n{user_prompt}\n\n")
                            f.write(f"Enhanced Prompt:\n{enhanced_prompt}\n")
                        result['saved_prompts_path'] = prompt_path
                        logger.debug(f"💾 Saved prompts: {prompt_path}")
                        
                    else:
                        logger.warning(f"⚠️ Cannot save image for prompt {idx}: not a PIL Image")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save image for prompt {idx}: {e}")
            
            # 4. CLIP 점수 계산
            if hasattr(self.clip_calculator, 'calculate_comprehensive_reward'):
                # 새로운 종합 보상 계산 방식
                rewards = self.clip_calculator.calculate_comprehensive_reward(image, user_prompt, enhanced_prompt)
                result['clip_score'] = rewards.get('clip_similarity', 0.0)
                result['quality_score'] = rewards.get('image_quality', 0.0)
            else:
                # 기존 방식 (호환성)
                clip_score = self.clip_calculator.calculate_reward(image, enhanced_prompt)
                result['clip_score'] = clip_score
                
                quality_score = self.clip_calculator.calculate_quality_reward(image)
                result['quality_score'] = quality_score
            
            # 5. 처리 시간 기록
            result['processing_time'] = time.time() - start_time
            
            # 성공 기준: CLIP 점수가 임계값 이상
            if result['clip_score'] > 0.3:  # 임계값은 조정 가능
                result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.warning(f"⚠️ Single item evaluation failed: {e}")
        
        return result
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        전체 평가 히스토리 요약
        
        Returns:
            Dict[str, Any]: 평가 요약
        """
        if not self.evaluation_history:
            return {'message': 'No evaluation history available'}
        
        # 최신 평가 결과
        latest = self.evaluation_history[-1]
        
        # 성능 트렌드 계산
        success_rates = [eval_result['success_rate'] for eval_result in self.evaluation_history]
        clip_scores = [eval_result['avg_clip_score'] for eval_result in self.evaluation_history]
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'latest_success_rate': latest['success_rate'],
            'latest_clip_score': latest['avg_clip_score'],
            'avg_success_rate': np.mean(success_rates),
            'avg_clip_score': np.mean(clip_scores),
            'success_rate_trend': self._calculate_trend(success_rates),
            'clip_score_trend': self._calculate_trend(clip_scores),
            'best_evaluation': max(self.evaluation_history, key=lambda x: x['success_rate']),
            'category_performance': latest.get('category_results', {}),
            'difficulty_performance': latest.get('difficulty_results', {})
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        값들의 트렌드 계산
        
        Args:
            values (List[float]): 값 리스트
            
        Returns:
            str: 트렌드 ("improving", "declining", "stable")
        """
        if len(values) < 3:
            return "insufficient_data"
        
        recent_avg = np.mean(values[-3:])
        earlier_avg = np.mean(values[:-3]) if len(values) > 3 else np.mean(values[:3])
        
        change = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
        
        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "declining"
        else:
            return "stable"
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        상세한 평가 리포트 생성
        
        Args:
            save_path (str, optional): 리포트 저장 경로
            
        Returns:
            str: 생성된 리포트 텍스트
        """
        summary = self.get_evaluation_summary()
        
        report = f"""
VLM GRPO Validation Report
=========================

📊 전체 통계:
- 총 평가 횟수: {summary.get('total_evaluations', 0)}
- 최신 성공률: {summary.get('latest_success_rate', 0):.2%}
- 평균 성공률: {summary.get('avg_success_rate', 0):.2%}
- 최신 CLIP 점수: {summary.get('latest_clip_score', 0):.4f}
- 평균 CLIP 점수: {summary.get('avg_clip_score', 0):.4f}

📈 성능 트렌드:
- 성공률 트렌드: {summary.get('success_rate_trend', 'N/A')}
- CLIP 점수 트렌드: {summary.get('clip_score_trend', 'N/A')}

📂 카테고리별 성능:
"""
        
        for category, stats in summary.get('category_performance', {}).items():
            success_rate = stats.get('success_rate', 0)
            count = stats.get('count', 0)
            report += f"- {category}: {success_rate:.2%} ({count} samples)\n"
        
        report += "\n🎯 난이도별 성능:\n"
        for difficulty, stats in summary.get('difficulty_performance', {}).items():
            success_rate = stats.get('success_rate', 0)
            count = stats.get('count', 0)
            report += f"- {difficulty}: {success_rate:.2%} ({count} samples)\n"
        
        # 리포트 저장
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"📋 Report saved to {save_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save report: {e}")
        
        return report


if __name__ == "__main__":
    # Validator 테스트 코드
    print("🧪 Validation Evaluator Test")
    print("=" * 35)
    
    try:
        # Mock 컴포넌트들
        class MockVLM:
            def enhance_prompt(self, prompt):
                return f"enhanced {prompt} with detailed description"
        
        class MockSDGenerator:
            def generate_image(self, prompt):
                return f"image_for_{prompt[:20]}"
        
        class MockCLIPCalculator:
            def calculate_reward(self, image, prompt):
                return 0.75  # 임의의 점수
            
            def calculate_quality_reward(self, image):
                return 0.65  # 임의의 품질 점수
        
        # Validator 초기화
        validator = ValidationEvaluator(
            MockVLM(),
            MockSDGenerator(), 
            MockCLIPCalculator()
        )
        
        # 테스트 데이터
        test_data = [
            {"user_prompt": "a cat", "category": "basic", "difficulty": "easy"},
            {"user_prompt": "sunset", "category": "basic", "difficulty": "easy"},
            {"user_prompt": "abstract art", "category": "creative", "difficulty": "hard"}
        ]
        
        print("✅ Validator initialized successfully")
        
        # 배치 평가 테스트
        print("\n🔄 Testing batch evaluation:")
        results = validator.evaluate_batch(test_data)
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Average CLIP score: {results['avg_clip_score']:.4f}")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        
        # 요약 테스트
        print("\n📊 Testing summary generation:")
        summary = validator.get_evaluation_summary()
        print(f"  Total evaluations: {summary['total_evaluations']}")
        print(f"  Success rate trend: {summary['success_rate_trend']}")
        
        # 리포트 생성 테스트
        print("\n📋 Testing report generation:")
        report = validator.generate_report()
        print("  Report generated successfully")
        print(f"  Report length: {len(report)} characters")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\nUsage:")
    print("from evaluation.validator import ValidationEvaluator")
    print("validator = ValidationEvaluator(vlm, sd_gen, clip_calc)")
    print("results = validator.evaluate_batch(validation_data)") 