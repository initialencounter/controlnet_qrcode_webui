import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from fastapi import FastAPI
from pydantic import BaseModel
from requests import get
import io
import uvicorn
import numpy as np
from typing import Optional
import base64
import gradio as gr

controlnet = ControlNetModel.from_pretrained("DionTimmer/controlnet_qrcode-control_v1p_sd15",
                                             torch_dtype=torch.float16)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)

pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def get_image(url):
    image = get(url, headers={'responseType': 'arraybuffer'})
    image = Image.open(io.BytesIO(image.content))
    return image

def predict(source_image,
            init_image,
            seed=123121231,
            prompt = "a bilboard in NYC with a qrcode" ,
            negative_prompt = "ugly, disfigured, low quality, blurry, nsfw",
            width=768,
            height=768,
            guidance_scale=20,
            num_inference_steps=150,
            controlnet_conditioning_scale=1.5,
            strength=0.9
            ):
    condition_image = resize_for_condition_image(source_image, 768)
    init_image = resize_for_condition_image(init_image, 768)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=prompt,
             negative_prompt=negative_prompt, 
             image=init_image,
             control_image=condition_image,
             width=int(width),
             height=int(height),
             guidance_scale=int(guidance_scale),
             controlnet_conditioning_scale=float(controlnet_conditioning_scale),
             generator=generator,
             strength=float(strength), 
             num_inference_steps=int(num_inference_steps)
            )
    return image.images[0]

app = FastAPI()
class Item(BaseModel):
    source_image_url: str
    init_image_url: str
    seed: int = 123121231
    prompt: str = "a bilboard in NYC with a qrcode"
    negative_prompt: str = "ugly, disfigured, low quality, blurry, nsfw"
    width: int = 768
    height: int = 768
    guidance_scale: int = 20
    controlnet_conditioning_scale: float = 1.5
    strength: float = 0.9
    num_inference_steps: int = 150

@app.post("/predict")
def create_item(item:Item):    
    result = predict(
             source_image=get_image(item.source_image_url),
             init_image=get_image(item.init_image_url),
             seed = item.seed,
             prompt=item.prompt,
             negative_prompt=item.negative_prompt, 
             width=item.width,
             height=item.height,
             guidance_scale=item.guidance_scale,
             controlnet_conditioning_scale=item.controlnet_conditioning_scale,
             strength=item.strength, 
             num_inference_steps=item.num_inference_steps)
    return {
        "image": base64.b64encode(result.tobytes()).decode('utf-8')
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
    