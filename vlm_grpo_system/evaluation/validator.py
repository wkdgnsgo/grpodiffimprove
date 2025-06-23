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
    
    def evaluate_batch(self, validation_data: List[Dict]) -> Dict[str, Any]:
        """
        배치 데이터에 대한 종합 평가
        
        Args:
            validation_data (List[Dict]): 검증 데이터
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        logger.info(f"🔍 Evaluating batch of {len(validation_data)} items")
        
        results = {
            'timestamp': time.time(),
            'total_samples': len(validation_data),
            'success_rate': 0.0,
            'avg_clip_score': 0.0,
            'quality_score': 0.0,
            'processing_time': 0.0,
            'category_results': {},
            'difficulty_results': {},
            'detailed_results': []
        }
        
        start_time = time.time()
        successful_evaluations = 0
        clip_scores = []
        quality_scores = []
        
        for item in validation_data:
            try:
                # 개별 아이템 평가
                item_result = self._evaluate_single_item(item)
                results['detailed_results'].append(item_result)
                
                if item_result['success']:
                    successful_evaluations += 1
                    clip_scores.append(item_result['clip_score'])
                    quality_scores.append(item_result['quality_score'])
                
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
                logger.warning(f"⚠️ Failed to evaluate item: {e}")
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
        return results
    
    def _evaluate_single_item(self, item: Dict) -> Dict[str, Any]:
        """
        단일 아이템 평가
        
        Args:
            item (Dict): 평가할 아이템
            
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
            
            # 3. CLIP 점수 계산
            clip_score = self.clip_calculator.calculate_reward(image, enhanced_prompt)
            result['clip_score'] = clip_score
            
            # 4. 품질 점수 계산
            quality_score = self.clip_calculator.calculate_quality_reward(image)
            result['quality_score'] = quality_score
            
            # 5. 처리 시간 기록
            result['processing_time'] = time.time() - start_time
            
            # 성공 기준: CLIP 점수가 임계값 이상
            if clip_score > 0.3:  # 임계값은 조정 가능
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