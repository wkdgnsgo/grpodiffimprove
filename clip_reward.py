from transformers import CLIPProcessor, CLIPModel
import torch
import logging
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from sentence_transformers import SentenceTransformer, util
from typing import List

logger = logging.getLogger(__name__)

class CLIPReward:

    def __init__(self, model_name = "openai/clip-vit-base-patch32", device = "cuda"):
        self.model_name = model_name
        self.device = device

        self._load_model()
        logger.info(f"CLIP reward init")

    def _load_model(self):
        logger.info(f"Loading CLIP model: {self.model_name}")
            
        # CLIP 프로세서 로드
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # CLIP 모델 로드
        self.model = CLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        )

        self.aesthetic_predictor_model, self.aesthetic_predictor_preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.aesthetic_predictor_model = self.aesthetic_predictor_model.to(torch.float16).to(self.device)

        self.sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        
        # 디바이스로 이동
        self.model = self.model.to(self.device)
        self.model.eval()  # 평가 모드 설정
    
    def calculate_reward(self, user_prompt, enhance_prompt, image, aesthetic_weight_factor = 20.0, clip_pen_threshold = 0.28, semantic_sim_threshold = 0.7):
        inputs = self.processor(
                text=[user_prompt], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 이미지와 텍스트 임베딩 추출
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            clip_sim = torch.sum(text_embeds * image_embeds, dim=-1).item()
            logger.info(f"clip similiarity: {clip_sim}")

        pixel_values = (
            self.aesthetic_predictor_preprocessor(images=image, return_tensors="pt").pixel_values.to(torch.float16)
            .to(self.device)
        )

        with torch.inference_mode():
            aesthetic_score = self.aesthetic_predictor_model(pixel_values).logits.squeeze().item()
            original_embed = self.sentence_transformer_model.encode(user_prompt, convert_to_tensor=True)
            enhanced_embed = self.sentence_transformer_model.encode(enhance_prompt, convert_to_tensor=True)
            embed_cosine_sim = util.cos_sim(original_embed, enhanced_embed).item()
        logger.info(f"aesthetic score: {aesthetic_score}")

        clip_pen_score = aesthetic_weight_factor * min(clip_sim - clip_pen_threshold, 0)
        semantic_penalty_term = 0.0

        if embed_cosine_sim < semantic_sim_threshold:
            semantic_penalty_term = -10 * (semantic_sim_threshold - embed_cosine_sim)

        reward = aesthetic_score + clip_pen_score + semantic_penalty_term

        return reward
    
    def calculate_batch_rewards(self, user_prompt: str, enhanced_prompts: List[str], images: List, 
                              aesthetic_weight_factor: float = 20.0, clip_pen_threshold: float = 0.28, 
                              semantic_sim_threshold: float = 0.7) -> List[float]:
        """배치 리워드 계산 - 여러 이미지를 한번에 처리"""
        logger.info(f"🔍 배치 CLIP 리워드 계산 ({len(images)}개)")
        
        batch_rewards = []
        
        try:
            # 1단계: 배치 CLIP 유사도 계산
            user_prompts = [user_prompt] * len(images)  # 사용자 프롬프트 복제
            
            inputs = self.processor(
                text=user_prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 배치 임베딩 정규화
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # 배치 CLIP 유사도 계산
                clip_similarities = torch.sum(text_embeds * image_embeds, dim=-1).cpu().tolist()
            
            # 2단계: 배치 미적 점수 계산
            aesthetic_scores = []
            for image in images:
                pixel_values = (
                    self.aesthetic_predictor_preprocessor(images=image, return_tensors="pt")
                    .pixel_values.to(torch.float16).to(self.device)
                )
                
                with torch.inference_mode():
                    aesthetic_score = self.aesthetic_predictor_model(pixel_values).logits.squeeze().item()
                    aesthetic_scores.append(aesthetic_score)
            
            # 3단계: 배치 의미적 유사도 계산
            original_embed = self.sentence_transformer_model.encode(user_prompt, convert_to_tensor=True)
            enhanced_embeds = self.sentence_transformer_model.encode(enhanced_prompts, convert_to_tensor=True)
            semantic_similarities = util.cos_sim(original_embed, enhanced_embeds).squeeze().cpu().tolist()
            
            # 단일 값인 경우 리스트로 변환
            if not isinstance(semantic_similarities, list):
                semantic_similarities = [semantic_similarities]
            
            # 4단계: 최종 리워드 계산
            for i, (clip_sim, aesthetic_score, semantic_sim) in enumerate(
                zip(clip_similarities, aesthetic_scores, semantic_similarities)
            ):
                # CLIP 패널티
                clip_pen_score = aesthetic_weight_factor * min(clip_sim - clip_pen_threshold, 0)
                
                # 의미적 패널티
                semantic_penalty_term = 0.0
                if semantic_sim < semantic_sim_threshold:
                    semantic_penalty_term = -10 * (semantic_sim_threshold - semantic_sim)
                
                # 최종 리워드
                reward = aesthetic_score + clip_pen_score + semantic_penalty_term
                batch_rewards.append(reward)
                
                logger.info(f"  이미지 {i+1}: CLIP={clip_sim:.3f}, 미적={aesthetic_score:.3f}, "
                          f"의미={semantic_sim:.3f}, 리워드={reward:.3f}")
        
        except Exception as e:
            logger.error(f"❌ 배치 리워드 계산 실패: {e}")
            # 에러 시 기본 리워드
            batch_rewards = [0.2] * len(images)
        
        logger.info(f"✅ 배치 리워드 완료: 평균 {sum(batch_rewards)/len(batch_rewards):.3f}")
        return batch_rewards