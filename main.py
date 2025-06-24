import torch
from diffusers import StableDiffusion3Pipeline
from qwen import QWENModel
from clip_reward import CLIPReward
import logging

logger = logging.getLogger(__name__)

CHALLENGING_PROMPTS = [
    # 기본 동물/객체
    "a cat sitting on a chair",
    "a beautiful sunset over mountains",
    "a robot playing guitar",
    "a flower garden in spring",
    "an old castle on a hill",
    
    # SD3 어려운 색상 조합
    "a purple rabbit sitting in grass",
    "a green cat with yellow eyes",
    "a blue elephant in the desert",
    "a red bird with black wings",
    "a yellow dog with pink spots",
    
    # 모순적인 개념들
    "a square wheel rolling down a hill",
    "an upside down tree growing in the sky",
    "a transparent fish swimming in air",
    "a silent thunderstorm with visible sound waves",
    "a car with legs instead of wheels",
    
    # 추상적 개념들
    "the concept of happiness visualized as colors",
    "time flowing backwards in a clock",
    "music made visible as geometric shapes",
    "the feeling of nostalgia as a landscape",
    "dreams transforming into reality",
    
    # 복잡한 재질/텍스처
    "a glass sculpture of a dragon",
    "a metallic chrome rose on black velvet",
    "a wooden elephant with crystal eyes",
    "a paper airplane made of liquid mercury",
    "a stone butterfly with feather wings",
    
    # 환상적/초현실적
    "a floating island with waterfalls going upward",
    "a library where books fly like birds",
    "a mirror that shows different seasons",
    "a doorway leading to another dimension",
    "a phoenix made of pure light",
    
    # 고급 조명/분위기
    "a portrait lit by candlelight",
    "neon lights reflecting on wet streets",
    "sunbeams through stained glass windows",
    "aurora borealis over a frozen lake",
    "a lighthouse beam cutting through fog"
]


def generate_image(prompt, pipeline):

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            num_images_per_prompt=1
        )

    image = result.images[0]
    return image
def generate_images_batch(prompts, pipe):
        """
        배치로 여러 이미지 생성
        
        Args:
            prompts (List[str]): 프롬프트 리스트
            seeds (List[int], optional): 시드 리스트
            
        Returns:
            List[Image.Image]: 생성된 이미지들
        """
        images = []
        
        for i, prompt in enumerate(prompts):
            image = generate_image(prompt, pipe)
            images.append(image)
        return images


def main():
    device = "cuda:3"

    pipeline = StableDiffusion3Pipeline.from_pretrained(
         "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16 
    )
            
    # 디바이스로 이동
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    qwen = QWENModel(device="cuda:2")

    reward_function = CLIPReward(device="cuda:2")

    for prompt in CHALLENGING_PROMPTS:
        qwen_output = qwen.enhance_prompt(prompt)
        logger.info(f"user prompt: {prompt}")
        logger.info(f"enhanced prompt: {qwen_output['enhanced_prompt']}")

        print(f"user prompt: {prompt}")
        print(f"enhanced prompt: {qwen_output['enhanced_prompt']}")
        original_image = generate_image(qwen_output["original_prompt"], pipeline)
        enhanced_image = generate_image(qwen_output["enhanced_prompt"], pipeline)
        original_image.save(f"./images/{prompt}.png")
        enhanced_image.save(f"./images/{qwen_output['enhanced_prompt'][:50]}.png")
        original_image_reward = reward_function.calculate_reward(qwen_output["original_prompt"], qwen_output["enhanced_prompt"], original_image)
        enhanced_image_reward = reward_function.calculate_reward(qwen_output["original_prompt"], qwen_output["enhanced_prompt"], enhanced_image)

        logger.info(f"original reward: {original_image_reward}")
        logger.info(f"enhanced reward: {enhanced_image_reward}")
        print(f"original reward: {original_image_reward}")
        print(f"enhanced reward: {enhanced_image_reward}")

        

if __name__ == "__main__":
     main()

            