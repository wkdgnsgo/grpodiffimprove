#!/usr/bin/env python3
"""
순수 GRPO 트레이너 (easyr1 스타일)
Value Network 없이 오직 Policy Network만 사용
그룹 평균을 implicit baseline으로 사용하는 올바른 GRPO 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import math
import re
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PureGRPOConfig:
    """순수 GRPO 설정 (Value Network 없음)"""
    learning_rate: float = 1e-6
    batch_size: int = 4
    num_rollouts: int = 5  # 그룹별 롤아웃 수
    max_prompt_length: int = 77
    max_new_tokens: int = 30
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 100
    kl_coef: float = 0.02
    clip_ratio: float = 0.1
    entropy_coef: float = 0.02
    vocab_size: int = 32000
    enable_step_logging: bool = True  # 상세 스텝 로깅 활성화
    log_dir: str = "training_logs"    # 로그 저장 디렉토리

class StepLogger:
    """각 스텝의 상세 정보를 기록하는 로거"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_data = []
        self.episode_counter = 0
        
        # 이미지 저장 디렉토리
        self.image_dir = os.path.join(log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 에피소드별 디렉토리
        self.episodes_dir = os.path.join(log_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)
        
        # 요약 통계 저장
        self.summary_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'best_reward': 0.0,
            'worst_reward': 0.0,
            'reward_history': []
        }
    
    def log_step(self, step_info: Dict):
        """스텝 정보 로깅"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_info['timestamp'] = timestamp
        self.step_data.append(step_info)
        
        # 콘솔 출력
        self._print_step_summary(step_info)
        
        # JSON 파일로 저장
        self._save_to_json()
    
    def _print_step_summary(self, step_info: Dict):
        """스텝 요약 정보 출력"""
        print("\n" + "="*80)
        print(f"📊 STEP {step_info.get('step', 'N/A')} - {step_info.get('timestamp', '')}")
        print("="*80)
        
        print(f"🔤 Original Prompt: '{step_info.get('original_prompt', 'N/A')}'")
        print(f"✨ Enhanced Prompt: '{step_info.get('enhanced_prompt', 'N/A')}'")
        
        if 'reward_components' in step_info:
            rewards = step_info['reward_components']
            print(f"🎯 Rewards:")
            print(f"   - Original→Image: {rewards.get('original_reward', 0):.3f}")
            print(f"   - Enhanced→Image: {rewards.get('enhanced_reward', 0):.3f}")
            print(f"   - Final Reward: {rewards.get('final_reward', 0):.3f}")
        
        if 'action_info' in step_info:
            action = step_info['action_info']
            print(f"🎬 Action: Token {action.get('token_id', 'N/A')} → '{action.get('token_text', 'N/A')}'")
            print(f"   - Log Prob: {action.get('log_prob', 0):.4f}")
        
        if 'images_saved' in step_info:
            print(f"🖼️  Images saved: {step_info['images_saved']}")
        
        print("="*80)
    
    def _save_to_json(self):
        """JSON 파일로 저장"""
        json_path = os.path.join(self.log_dir, "step_logs.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.step_data, f, indent=2, ensure_ascii=False)
    
    def save_image(self, image, filename: str) -> str:
        """이미지 저장"""
        image_path = os.path.join(self.image_dir, filename)
        image.save(image_path)
        return image_path
    
    def start_new_episode(self, episode_id: str, original_prompt: str):
        """새 에피소드 시작"""
        self.episode_counter += 1
        episode_dir = os.path.join(self.episodes_dir, f"episode_{self.episode_counter:03d}_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 에피소드 메타데이터 저장
        episode_meta = {
            'episode_id': episode_id,
            'episode_number': self.episode_counter,
            'original_prompt': original_prompt,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'steps': []
        }
        
        meta_path = os.path.join(episode_dir, "episode_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(episode_meta, f, indent=2, ensure_ascii=False)
        
        return episode_dir
    
    def log_episode_step(self, episode_dir: str, step_data: Dict):
        """에피소드 내 스텝 로깅"""
        # 에피소드 메타데이터 업데이트
        meta_path = os.path.join(episode_dir, "episode_meta.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            episode_meta = json.load(f)
        
        episode_meta['steps'].append(step_data)
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(episode_meta, f, indent=2, ensure_ascii=False)
    
    def finish_episode(self, episode_dir: str, final_reward: float, total_steps: int):
        """에피소드 완료 처리"""
        # 에피소드 메타데이터 업데이트
        meta_path = os.path.join(episode_dir, "episode_meta.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            episode_meta = json.load(f)
        
        episode_meta.update({
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'final_reward': final_reward,
            'total_steps': total_steps,
            'completed': True
        })
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(episode_meta, f, indent=2, ensure_ascii=False)
        
        # 통계 업데이트
        self.summary_stats['total_episodes'] += 1
        self.summary_stats['total_steps'] += total_steps
        self.summary_stats['reward_history'].append(final_reward)
        
        if len(self.summary_stats['reward_history']) == 1:
            self.summary_stats['best_reward'] = final_reward
            self.summary_stats['worst_reward'] = final_reward
        else:
            self.summary_stats['best_reward'] = max(self.summary_stats['best_reward'], final_reward)
            self.summary_stats['worst_reward'] = min(self.summary_stats['worst_reward'], final_reward)
        
        self.summary_stats['average_reward'] = np.mean(self.summary_stats['reward_history'])
        
        # 요약 통계 저장
        stats_path = os.path.join(self.log_dir, "summary_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 에피소드 완료: 리워드={final_reward:.3f}, 스텝={total_steps}")
    
    def create_comparison_html(self):
        """이미지 비교를 위한 HTML 보고서 생성"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>GRPO 훈련 결과 비교</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .episode { border: 1px solid #ddd; margin: 20px 0; padding: 20px; }
        .step { border-left: 3px solid #007bff; margin: 10px 0; padding: 10px; }
        .image-comparison { display: flex; gap: 20px; margin: 10px 0; }
        .image-container { text-align: center; }
        .image-container img { max-width: 300px; height: auto; border: 1px solid #ddd; }
        .reward-info { background: #f8f9fa; padding: 10px; margin: 10px 0; }
        .prompt-info { background: #e9ecef; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🎯 GRPO 훈련 결과 비교</h1>
    <div class="summary">
        <h2>📊 요약 통계</h2>
        <p>총 에피소드: {total_episodes}</p>
        <p>총 스텝: {total_steps}</p>
        <p>평균 리워드: {average_reward:.3f}</p>
        <p>최고 리워드: {best_reward:.3f}</p>
        <p>최저 리워드: {worst_reward:.3f}</p>
    </div>
""".format(**self.summary_stats)
        
        # 각 에피소드 정보 추가
        for episode_dir in sorted(os.listdir(self.episodes_dir)):
            episode_path = os.path.join(self.episodes_dir, episode_dir)
            meta_path = os.path.join(episode_path, "episode_meta.json")
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    episode_meta = json.load(f)
                
                html_content += f"""
    <div class="episode">
        <h3>에피소드 {episode_meta['episode_number']}: {episode_meta['original_prompt']}</h3>
        <p>최종 리워드: {episode_meta.get('final_reward', 0):.3f}</p>
        <p>총 스텝: {episode_meta.get('total_steps', 0)}</p>
"""
                
                # 각 스텝의 이미지 비교
                for step in episode_meta.get('steps', []):
                    if 'images_saved' in step:
                        html_content += f"""
        <div class="step">
            <h4>스텝 {step['step']}</h4>
            <div class="prompt-info">
                <p><strong>원본 프롬프트:</strong> {step['original_prompt']}</p>
                <p><strong>향상된 프롬프트:</strong> {step['enhanced_prompt']}</p>
            </div>
            <div class="reward-info">
                <p><strong>리워드:</strong> 원본→이미지 {step['reward_components']['original_reward']:.3f}, 
                   향상→이미지 {step['reward_components']['enhanced_reward']:.3f}</p>
            </div>
            <div class="image-comparison">
                <div class="image-container">
                    <img src="{step['images_saved']['original']}" alt="원본 이미지">
                    <p>원본 프롬프트 이미지</p>
                </div>
                <div class="image-container">
                    <img src="{step['images_saved']['enhanced']}" alt="향상된 이미지">
                    <p>향상된 프롬프트 이미지</p>
                </div>
            </div>
        </div>
"""
                
                html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        # HTML 파일 저장
        html_path = os.path.join(self.log_dir, "comparison_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📄 HTML 비교 보고서 생성됨: {html_path}")
        return html_path

class EnglishTokenFilter:
    """영어 토큰만 허용하는 필터"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.english_token_ids = self._build_english_vocab()
        logger.info(f"영어 토큰 필터 초기화: {len(self.english_token_ids)}/{len(tokenizer.get_vocab())} 토큰")
    
    def _build_english_vocab(self) -> set:
        """영어 토큰 ID 집합 구성"""
        vocab = self.tokenizer.get_vocab()
        english_tokens = set()
        
        # 영어 패턴 정의
        english_pattern = re.compile(r'^[a-zA-Z0-9\s\.,!?;:\-_\'\"()\[\]{}@#$%^&*+=<>/\\|`~]*$')
        
        for token, token_id in vocab.items():
            # 토큰 디코딩
            try:
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
                # 영어 패턴 매칭
                if english_pattern.match(decoded.strip()):
                    english_tokens.add(token_id)
            except:
                continue
        
        # 특수 토큰들 추가 (EOS, BOS, PAD 등)
        special_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None,
            self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None,
            self.tokenizer.unk_token_id if hasattr(self.tokenizer, 'unk_token_id') else None,
        ]
        
        for token_id in special_tokens:
            if token_id is not None:
                english_tokens.add(token_id)
        
        return english_tokens
    
    def filter_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """영어가 아닌 토큰의 로짓을 -inf로 설정"""
        filtered_logits = logits.clone()
        
        # 모든 토큰을 -inf로 설정
        filtered_logits.fill_(float('-inf'))
        
        # 영어 토큰만 원래 값으로 복원
        for token_id in self.english_token_ids:
            if token_id < logits.size(-1):
                filtered_logits[..., token_id] = logits[..., token_id]
        
        return filtered_logits

class PureGRPOPolicy(nn.Module):
    """순수 GRPO 정책 네트워크 (Value Head 없음)"""
    
    def __init__(self, qwen_model, config: PureGRPOConfig):
        super().__init__()
        self.qwen_model = qwen_model
        self.config = config
        
        # 영어 토큰 필터 초기화
        self.english_filter = EnglishTokenFilter(qwen_model.tokenizer)
        
        # GPU 디바이스 설정
        self.qwen_device = "cuda:0"  # QWEN은 GPU 0
        self.policy_device = "cuda:0"  # Policy head도 GPU 0에서 학습
        
        self.hidden_size = qwen_model.model.config.hidden_size
        self.vocab_size = len(qwen_model.tokenizer.get_vocab())
        
        logger.info(f"순수 GRPO 정책 - Hidden: {self.hidden_size}, Vocab: {self.vocab_size}")
        logger.info(f"GPU 배치: QWEN={self.qwen_device}, Policy={self.policy_device}")
        logger.info(f"영어 토큰 필터링 활성화: {len(self.english_filter.english_token_ids)} 토큰")
        
        # 오직 정책 헤드만! (Value Head 없음) - GPU 0에 배치 (float16으로 통일)
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.vocab_size)
        ).to(self.policy_device).half()  # float16으로 변환
        
        self._init_weights()
        
        logger.info(f"순수 GRPO 정책 네트워크 초기화 완료 - Action Space: {self.vocab_size}")
        logger.info("✅ Value Network 없음 - 그룹 평균을 implicit baseline으로 사용")
    
    def _init_weights(self):
        """가중치 초기화"""
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                gain = 0.02 if layer.out_features == self.vocab_size else 0.1
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """오직 정책 로짓만 반환 (Values 없음)"""
        batch_size = input_ids.size(0)
        
        # 입력 텐서를 QWEN GPU(0번)로 이동
        input_ids = input_ids.to(self.qwen_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.qwen_device)
        
        with torch.no_grad():
            outputs = self.qwen_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # QWEN2VL 모델은 last_hidden_state 대신 hidden_states를 사용
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # hidden_states는 튜플이므로 마지막 레이어 선택
                hidden_states = outputs.hidden_states[-1]
            else:
                # 대안: logits에서 히든 스테이트 추출 시도
                raise AttributeError("Cannot find hidden states in model output")
        
        if attention_mask is not None:
            last_valid_indices = attention_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            last_hidden = hidden_states[torch.arange(batch_size, device=self.qwen_device), last_valid_indices]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        # Hidden states를 Policy GPU로 이동하고 float16으로 변환
        last_hidden = last_hidden.to(self.policy_device).half()
        
        # 오직 정책 로짓만 반환!
        policy_logits = self.policy_head(last_hidden)
        
        return policy_logits  # Values 없음!
    
    def get_action_and_log_prob(self, state: Dict):
        """액션 선택과 로그 확률 (Value 없음) - 영어 토큰 필터링 적용"""
        input_ids = state['input_ids'].unsqueeze(0)
        attention_mask = state['attention_mask'].unsqueeze(0)
        
        policy_logits = self(input_ids, attention_mask)
        
        # 영어 토큰 필터링 적용
        filtered_logits = self.english_filter.filter_logits(policy_logits)
        
        scaled_logits = filtered_logits / self.config.temperature
        scaled_logits = torch.clamp(scaled_logits, min=-10, max=10)
        
        # Top-k 필터링
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(scaled_logits, self.config.top_k, dim=-1)
            scaled_logits = torch.full_like(scaled_logits, float('-inf'))
            scaled_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        token_probs = F.softmax(scaled_logits, dim=-1)
        
        # Top-p 필터링
        if self.config.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(token_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            token_probs = token_probs.masked_fill(indices_to_remove, 0.0)
            
            prob_sum = token_probs.sum(dim=-1, keepdim=True)
            token_probs = token_probs / (prob_sum + 1e-8)
        
    
        token_dist = torch.distributions.Categorical(token_probs)
        action = token_dist.sample()
        action_log_prob = token_dist.log_prob(action).half()  # float16으로 변환
    
        
        # Value 없음! 오직 action, log_prob, logits만 반환 (모두 float16)
        return action.item(), action_log_prob, scaled_logits.squeeze(0).half()

class PureGRPOPromptEnvironment:
    """순수 GRPO용 프롬프트 환경"""
    
    def __init__(self, qwen_model, reward_model, sd_pipeline, config: PureGRPOConfig):
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        self.config = config
        self.tokenizer = qwen_model.tokenizer
        self.vocab_size = len(self.tokenizer.get_vocab())
        
        # 스텝 로거 초기화
        if config.enable_step_logging:
            self.step_logger = StepLogger(config.log_dir)
        else:
            self.step_logger = None
        
        # GPU 디바이스 설정
        self.qwen_device = "cuda:0"  # QWEN (토큰화)
        self.sd_device = "cuda:1"    # Stable Diffusion (이미지 생성)
        self.reward_device = "cuda:2"  # CLIP Reward (리워드 계산)
        
        self.current_prompt = ""
        self.original_prompt = ""
        self.step_count = 0
        self.current_episode_dir = None
        
        logger.info(f"순수 GRPO 환경 초기화 - Vocab: {self.vocab_size}")
        logger.info(f"GPU 배치: QWEN={self.qwen_device}, SD={self.sd_device}, Reward={self.reward_device}")
        if self.step_logger:
            logger.info(f"상세 스텝 로깅 활성화: {config.log_dir}")
    
    def reset(self, user_prompt: str):
        """환경 리셋 - GPU 0으로 토큰 이동 + 새 에피소드 시작"""
        self.original_prompt = user_prompt
        self.current_prompt = user_prompt
        self.step_count = 0
        
        # 새 에피소드 시작
        if self.step_logger:
            episode_id = user_prompt.replace(' ', '_')[:20]  # 간단한 ID 생성
            self.current_episode_dir = self.step_logger.start_new_episode(episode_id, user_prompt)
            logger.info(f"🎬 새 에피소드 시작: {episode_id}")
        
        # 현재 프롬프트를 토큰화하고 QWEN GPU(0번)로 이동
        tokens = self.tokenizer.encode(
            self.current_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            padding='max_length'
        ).to(self.qwen_device)
        
        attention_mask = (tokens != self.tokenizer.pad_token_id).long().to(self.qwen_device)
        
        return {
            'input_ids': tokens.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'current_prompt': self.current_prompt,
            'original_prompt': self.original_prompt
        }
    
    def step(self, action: int):
        """환경 스텝 - GPU 간 데이터 이동 처리 + 상세 로깅"""
        # 액션(토큰)을 텍스트로 변환
        try:
            token_text = self.tokenizer.decode([action], skip_special_tokens=True)
            
            # 프롬프트에 토큰 추가
            if token_text.strip():
                if self.current_prompt.endswith(' ') or token_text.startswith(' '):
                    self.current_prompt += token_text
                else:
                    self.current_prompt += ' ' + token_text
            
            self.step_count += 1
            
            # 종료 조건
            done = (self.step_count >= self.config.max_new_tokens or 
                   action == self.tokenizer.eos_token_id or
                   len(self.current_prompt) >= self.config.max_prompt_length * 4)
            
            # 리워드 계산 (에피소드 끝에만) - GPU 간 이동 처리
            if done:
                try:
                    logger.info(f"🖼️  이미지 생성 시작 (GPU {self.sd_device})")
                    
                    # SD3 파이프라인을 GPU 1로 이동하여 이미지 생성
                    with torch.cuda.device(1):
                        # 원본 프롬프트로 이미지 생성 (비교용)
                        original_result = self.sd_pipeline(
                            prompt=self.original_prompt,
                            num_inference_steps=20,
                            guidance_scale=7.0,
                            height=1024,
                            width=1024
                        )
                        original_image = original_result.images[0]
                        
                        # 향상된 프롬프트로 이미지 생성
                        enhanced_result = self.sd_pipeline(
                            prompt=self.current_prompt,
                            num_inference_steps=20,
                            guidance_scale=7.0,
                            height=1024,
                            width=1024
                        )
                        enhanced_image = enhanced_result.images[0]
                    
                    logger.info(f"🎯 리워드 계산 시작 (GPU {self.reward_device})")
                    
                    # CLIP 리워드를 GPU 2에서 계산
                    with torch.cuda.device(2):
                        # 원본 프롬프트 vs 원본 이미지
                        original_reward = self.reward_model.calculate_reward(
                            self.original_prompt,
                            self.original_prompt,
                            original_image
                        )
                        
                        # 원본 프롬프트 vs 향상된 이미지 (실제 리워드)
                        enhanced_reward = self.reward_model.calculate_reward(
                            self.original_prompt,
                            self.current_prompt,
                            enhanced_image
                        )
                    
                    # 길이 보너스
                    length_bonus = min(self.step_count / self.config.max_new_tokens, 1.0) * 0.1
                    total_reward = enhanced_reward + length_bonus
                    
                    logger.info(f"✅ 리워드 계산 완료: {total_reward:.4f}")
                    
                    # 상세 로깅
                    if self.step_logger:
                        # 이미지 저장
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        original_img_path = self.step_logger.save_image(
                            original_image, 
                            f"step_{self.step_count:03d}_{timestamp}_original.png"
                        )
                        enhanced_img_path = self.step_logger.save_image(
                            enhanced_image, 
                            f"step_{self.step_count:03d}_{timestamp}_enhanced.png"
                        )
                        
                        # 스텝 정보 로깅
                        step_info = {
                            'step': self.step_count,
                            'original_prompt': self.original_prompt,
                            'enhanced_prompt': self.current_prompt,
                            'action_info': {
                                'token_id': action,
                                'token_text': token_text,
                                'log_prob': 0.0  # 나중에 업데이트됨
                            },
                            'reward_components': {
                                'original_reward': float(original_reward),
                                'enhanced_reward': float(enhanced_reward),
                                'length_bonus': float(length_bonus),
                                'final_reward': float(total_reward)
                            },
                            'images_saved': {
                                'original': original_img_path,
                                'enhanced': enhanced_img_path
                            }
                        }
                        
                        # 전역 스텝 로깅
                        self.step_logger.log_step(step_info)
                        
                        # 에피소드별 스텝 로깅
                        if self.current_episode_dir:
                            self.step_logger.log_episode_step(self.current_episode_dir, step_info)
                        
                        # 에피소드 완료 처리
                        if done and self.current_episode_dir:
                            self.step_logger.finish_episode(
                                self.current_episode_dir, 
                                float(total_reward), 
                                self.step_count
                            )
                            # HTML 보고서 생성
                            self.step_logger.create_comparison_html()
                    
                except Exception as e:
                    logger.warning(f"Reward calculation failed: {e}")
                    total_reward = 0.0
                    original_image = None
                    enhanced_image = None
            else:
                total_reward = 0.0
                original_image = None
                enhanced_image = None
            
            # 다음 상태 (GPU 0으로 이동)
            if not done:
                next_tokens = self.tokenizer.encode(
                    self.current_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                    padding='max_length'
                ).to(self.qwen_device)
                
                next_attention_mask = (next_tokens != self.tokenizer.pad_token_id).long().to(self.qwen_device)
                
                # 다음 상태를 QWEN GPU(0번)로 이동
                next_state = {
                    'input_ids': next_tokens.squeeze(0),
                    'attention_mask': next_attention_mask.squeeze(0),
                    'current_prompt': self.current_prompt,
                    'original_prompt': self.original_prompt
                }
            else:
                next_state = None
            
            info = {
                'current_prompt': self.current_prompt,
                'step_count': self.step_count,
                'token_added': token_text,
                'original_image': original_image,
                'enhanced_image': enhanced_image
            }
            
            return next_state, total_reward, done, info
            
        except Exception as e:
            logger.warning(f"Step failed: {e}")
            return None, 0.0, True, {'error': str(e)}

class PureGRPOTrainer:
    """순수 GRPO 트레이너 (Value Network 없음)"""
    
    def __init__(self, qwen_model, reward_model, sd_pipeline, config: PureGRPOConfig):
        self.config = config
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        
        self.env = PureGRPOPromptEnvironment(qwen_model, reward_model, sd_pipeline, config)
        
        # 오직 정책 네트워크만! (Value Network 없음)
        self.policy = PureGRPOPolicy(qwen_model, config)
        
        # 참조 정책 (float16으로 통일)
        self.ref_policy = PureGRPOPolicy(qwen_model, config)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()
        self.ref_policy.half()  # float16으로 변환
        
        # 오직 정책 파라미터만 학습
        trainable_params = list(self.policy.policy_head.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-7)
        
        logger.info("🎯 순수 GRPO 트레이너 초기화 완료")
        logger.info(f"✅ Value Network 없음 - 오직 Policy Network만 사용")
        logger.info(f"📊 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def collect_rollouts(self, prompts: List[str]) -> List[Dict]:
        """롤아웃 수집 (Value 수집 없음)"""
        all_experiences = []
        
        for prompt_idx, user_prompt in enumerate(prompts):
            logger.info(f"Processing prompt {prompt_idx+1}/{len(prompts)}: '{user_prompt}'")
            
            for rollout_idx in range(self.config.num_rollouts):
                episode_experiences = []
                state = self.env.reset(user_prompt)
                done = False
                
                logger.info(f"  Rollout {rollout_idx+1}/{self.config.num_rollouts}")
                
                step_count = 0
                while not done and step_count < self.config.max_new_tokens:
                    # 정책에서 액션 선택 (Value 없음!)
                    action, log_prob, logits = self.policy.get_action_and_log_prob(state)
                    
                    # 참조 정책의 로그 확률
                    with torch.no_grad():
                        ref_logits = self.ref_policy(
                            state['input_ids'].unsqueeze(0),
                            state['attention_mask'].unsqueeze(0)
                        )
                        ref_log_prob = F.log_softmax(ref_logits, dim=-1)[0, action]
                    
                    next_state, reward, done, info = self.env.step(action)
                    
                    # 스텝 로거에 log_prob 업데이트 (에피소드 끝에서)
                    if done and self.env.step_logger and len(self.env.step_logger.step_data) > 0:
                        last_step_info = self.env.step_logger.step_data[-1]
                        if 'action_info' in last_step_info:
                            last_step_info['action_info']['log_prob'] = float(log_prob)
                            # JSON 파일 다시 저장
                            self.env.step_logger._save_to_json()
                    
                    # Value 없는 경험 저장!
                    experience = {
                        'state': {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()},
                        'action': action,
                        'log_prob': log_prob,
                        'ref_log_prob': ref_log_prob,
                        'reward': reward,
                        'done': done,
                        'prompt_idx': prompt_idx,
                        'rollout_idx': rollout_idx,
                        'info': info
                    }
                    
                    episode_experiences.append(experience)
                    state = next_state
                    step_count += 1
                
                all_experiences.extend(episode_experiences)
                
                if episode_experiences:
                    final_prompt = episode_experiences[-1]['info']['current_prompt']
                    final_reward = episode_experiences[-1]['reward']
                    logger.info(f"    Generated: '{final_prompt}' (reward: {final_reward:.3f})")
        
        return all_experiences
    
    def compute_grpo_advantages(self, experiences: List[Dict]) -> List[Dict]:
        """순수 GRPO Advantage 계산 (easyr1과 동일)"""
        # 프롬프트별 리워드 그룹화
        prompt_rewards = defaultdict(list)
        for exp in experiences:
            if exp['done']:
                prompt_rewards[exp['prompt_idx']].append(exp['reward'])
        
        # 그룹별 정규화 (easyr1과 동일한 방식)
        advantages = {}
        for prompt_idx, rewards in prompt_rewards.items():
            if len(rewards) > 1:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards) + 1e-8
                normalized_rewards = [(r - mean_reward) / std_reward for r in rewards]
            else:
                normalized_rewards = [0.0]
            
            advantages[prompt_idx] = normalized_rewards
            logger.info(f"Prompt {prompt_idx} GRPO: rewards={rewards} -> advantages={normalized_rewards}")
        
        # 경험에 advantage 할당
        rollout_counters = defaultdict(int)
        for exp in experiences:
            if exp['done']:
                prompt_idx = exp['prompt_idx']
                rollout_idx = rollout_counters[prompt_idx]
                if prompt_idx in advantages and rollout_idx < len(advantages[prompt_idx]):
                    exp['advantage'] = advantages[prompt_idx][rollout_idx]
                else:
                    exp['advantage'] = 0.0
                rollout_counters[prompt_idx] += 1
            else:
                exp['advantage'] = 0.0
        
        return experiences
    
    def train_step(self, experiences: List[Dict]) -> Dict:
        """순수 GRPO 학습 스텝 (Value Loss 없음)"""
        if not experiences:
            return {}
        
        valid_experiences = [exp for exp in experiences if exp.get('advantage', 0) != 0]
        if not valid_experiences:
            logger.warning("No valid experiences for training")
            return {}
        
        batch_states = []
        actions = []
        old_log_probs = []
        ref_log_probs = []
        advantages = []
        
        for exp in valid_experiences:
            batch_states.append(exp['state'])
            actions.append(exp['action'])
            old_log_probs.append(exp['log_prob'])
            ref_log_probs.append(exp['ref_log_prob'])
            advantages.append(exp['advantage'])
            # ❌ values.append(exp['value'])  # Value 없음!
        
        if len(batch_states) == 0:
            return {}
        
        # 패딩을 위한 최대 길이 찾기
        max_length = max(state['input_ids'].size(0) for state in batch_states)
        
        # 패딩된 텐서 생성 및 GPU 0으로 이동
        padded_input_ids = []
        padded_attention_masks = []
        
        for state in batch_states:
            input_ids_tensor = state['input_ids']
            attention_mask_tensor = state['attention_mask']
            
            # 패딩 필요한 길이 계산
            pad_length = max_length - input_ids_tensor.size(0)
            
            if pad_length > 0:
                # 패딩 추가 (오른쪽에 패딩) - GPU 0에서, dtype 보존
                padded_input = torch.cat([
                    input_ids_tensor.to("cuda:0"),
                    torch.zeros(pad_length, dtype=input_ids_tensor.dtype, device="cuda:0")
                ])
                padded_mask = torch.cat([
                    attention_mask_tensor.to("cuda:0"),
                    torch.zeros(pad_length, dtype=attention_mask_tensor.dtype, device="cuda:0")
                ])
            else:
                padded_input = input_ids_tensor.to("cuda:0")
                padded_mask = attention_mask_tensor.to("cuda:0")
            
            padded_input_ids.append(padded_input)
            padded_attention_masks.append(padded_mask)
        
        # 모든 텐서를 GPU 0으로 이동하고 적절한 dtype 설정
        input_ids = torch.stack(padded_input_ids).to("cuda:0")  # int 타입 유지
        attention_masks = torch.stack(padded_attention_masks).to("cuda:0")  # int 타입 유지
        actions = torch.tensor(actions).to("cuda:0")  # int 타입 유지
        old_log_probs = torch.stack(old_log_probs).to("cuda:0").half()  # float16
        ref_log_probs = torch.stack(ref_log_probs).to("cuda:0").half()  # float16
        advantages = torch.tensor(advantages, dtype=torch.float16).to("cuda:0")  # float16
        
        # 오직 정책 로짓만 계산! (Values 없음)
        policy_logits = self.policy(input_ids, attention_masks)
        
        new_log_probs = F.log_softmax(policy_logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # PPO 정책 손실
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 페널티
        kl_penalty = (new_log_probs - ref_log_probs).mean()
        
        # 엔트로피
        entropy = -(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)).sum(-1).mean()
        
        # 순수 GRPO 총 손실 (Value Loss 없음!)
        total_loss = (policy_loss + 
                     self.config.kl_coef * kl_penalty - 
                     self.config.entropy_coef * entropy)
        
        # 역전파
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy': entropy.item(),
            'avg_advantage': advantages.mean().item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'num_valid_experiences': len(valid_experiences)
        }
    
    def train(self, train_prompts: List[str], num_epochs: int = 10):
        """순수 GRPO 학습"""
        logger.info(f"🚀 순수 GRPO 학습 시작 (Value Network 없음)")
        logger.info(f"프롬프트: {len(train_prompts)}개, 에포크: {num_epochs}개")
        logger.info(f"Action Space: {self.env.vocab_size}개 토큰 (전체 어휘)")
        logger.info(f"✅ easyr1과 동일한 구조: 그룹 평균을 implicit baseline으로 사용")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 50)
            
            experiences = self.collect_rollouts(train_prompts)
            experiences = self.compute_grpo_advantages(experiences)
            metrics = self.train_step(experiences)
            
            logger.info(f"Epoch {epoch + 1} metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            if epoch % 2 == 0:
                self._log_sample_outputs(train_prompts[:2])
    
    def _log_sample_outputs(self, sample_prompts: List[str]):
        """샘플 출력 로깅"""
        logger.info("📝 Sample outputs:")
        for prompt in sample_prompts:
            state = self.env.reset(prompt)
            original_prompt = self.env.current_prompt
            
            # 몇 스텝 실행
            for _ in range(5):
                action, _, _ = self.policy.get_action_and_log_prob(state)
                state, _, done, info = self.env.step(action)
                if done:
                    break
            
            enhanced_prompt = self.env.current_prompt
            logger.info(f"  Original: {original_prompt}")
            logger.info(f"  Enhanced: {enhanced_prompt}")

def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    class MockQwenModel:
        def __init__(self):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            class MockModel:
                def __init__(self):
                    class Config:
                        hidden_size = 4096
                    self.config = Config()
                
                def __call__(self, **kwargs):
                    batch_size, seq_len = kwargs['input_ids'].shape
                    class Output:
                        last_hidden_state = torch.randn(batch_size, seq_len, 4096)
                    return Output()
            
            self.model = MockModel()
    
    class MockReward:
        def calculate_reward(self, original, enhanced, image):
            return np.random.uniform(5.0, 9.0)
    
    class MockSD:
        def __call__(self, **kwargs):
            from PIL import Image
            class Result:
                images = [Image.new('RGB', (1024, 1024), color='red')]
            return Result()
    
    config = PureGRPOConfig(
        learning_rate=1e-6,
        batch_size=2,
        num_rollouts=3,
        max_new_tokens=10,
        top_k=50
    )
    
    qwen = MockQwenModel()
    reward = MockReward()
    sd = MockSD()
    
    trainer = PureGRPOTrainer(qwen, reward, sd, config)
    
    test_prompts = ["a cat sitting", "beautiful sunset"]
    trainer.train(test_prompts, num_epochs=2)

if __name__ == "__main__":
    main() 