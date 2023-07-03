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

source_image=gr.inputs.Image(type="pil",label="qrcode_img")
init_image=gr.inputs.Image(type="pil",label="bg_img")
seed=gr.inputs.Number(default=123123123,label="seed")
prompt=gr.inputs.Textbox(lines=2,placeholder="input your prompt",default="a bilboard in NYC with a qrcode",label="prompt")
negative_prompt=gr.inputs.Textbox(lines=2,placeholder="input your neg_prompt",default="ugly, disfigured, low quality, blurry, nsfw",label="neg_prompt")
width=gr.inputs.Number(default=768,label="width")
height=gr.inputs.Number(default=768,label="heigth")
guidance_scale=gr.inputs.Number(default=20,label="guidance_scale")
num_inference_steps=gr.inputs.Number(default=20,label="steps")
controlnet_conditioning_scale=gr.inputs.Number(default=1.5,label="controlnet_conditioning_scale")
strength=gr.inputs.Number(default=0.9,label="strength")

app2 = gr.Interface(
    fn=predict,
    inputs=[
        source_image,
        init_image,
        seed,
        prompt,
        negative_prompt,
        width,
        height,
        guidance_scale,
        num_inference_steps,
        controlnet_conditioning_scale,
        strength
    ],
    outputs="image"
    )
if __name__ == "__main__":
    app2.launch(server_name="0.0.0.0",server_port=7860)
    