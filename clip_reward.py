from transformers import CLIPProcessor, CLIPModel
import torch
import logging
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from sentence_transformers import SentenceTransformer, util

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