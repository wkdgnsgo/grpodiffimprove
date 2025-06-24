import torch
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
import re


logger = logging.getLogger(__name__)

class QWENModel:

    def __init__(self, model_name = "Qwen/Qwen2-VL-7B-Instruct", device = "cuda", temperature = 0.7):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature

        self._load_model()
        self._setup_prompt_template()
        logger.info(f"Qwen init : {self.model_name}")

    def _load_model(self):
        
        self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        self.tokenizer = self.processor.tokenizer
  
        model_kwargs = {
            'torch_dtype': torch.float16,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True
        }

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
        ).to(self.device)

        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 생성 설정
        self.generation_config = GenerationConfig(
            max_new_tokens=77,
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info("Model loaded")
        
    def _setup_prompt_template(self):
         
        # 생성 지시를 위한 시스템 프롬프트
        self.system_prompt = """You are an expert at enhancing image generation prompts. 
        Given a simple user prompt, expand it into a detailed, and high-quality prompt for image generation.

        Guidelines:
        - Keep the original concept unchanged
        - Add artistic style, mood, and atmosphere
        - Include technical specifications (lighting, composition, resolution)
        - Add creative details that make the image more realistic
        - Make each enhancement unique and varied
        - Be descriptive but concise (aim for 20-40 additional words) """

                # 사용자 입력 템플릿
        self.user_template = """Original prompt: {user_prompt}

        Enhanced version:"""
    def enhance_prompt(self, user_prompt):
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
        ]
        
        # 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 후처리
        enhanced_prompt = self._post_process_output(generated_text)
        
        result = {
            'original_prompt': user_prompt,
            'enhanced_prompt': enhanced_prompt,
            'raw_output': generated_text
        }
        
        logger.info(f"Enhanced prompt: '{user_prompt}' -> '{enhanced_prompt[:50]}...'")
        return result
    def _post_process_output(self, raw_output):
        """생성된 출력 후처리"""
        # 불필요한 텍스트 제거
        enhanced = raw_output.strip()
        
        # "Enhanced prompt:" 등의 레이블 제거
        enhanced = re.sub(r'^(Enhanced prompt:|Prompt:|Result:)\s*', '', enhanced, flags=re.IGNORECASE)
        enhanced = enhanced.strip()
        
        # 따옴표 제거
        enhanced = enhanced.strip('"\'')
        
        return enhanced
    
    def enhance_prompts_batch(self, user_prompts):
        """배치로 여러 프롬프트 향상"""
        results = []
        for prompt in user_prompts:
            result = self.enhance_prompt(prompt)
            results.append(result)
        return results
