import io
import base64
import random
import requests
import os.path as osp
from PIL import Image
from fastapi import FastAPI, HTTPException
from dataclass.controlnet import *
from dataclass.payloads import InputPayload, SdFwdPaylodWithImage, OutputPayload

REF_IMAGES_DIR = "../static"
REF_IMAGES = {
    f"ref_{i}": [osp.join(REF_IMAGES_DIR, f"ref_{i}_{j}.jpg") for j in range(4)] for i in range(2)
}

DEFAULT_POS_PROMPT_LIST = ["(masterpiece, best quality, highly detailed, absurdres)", 
                           "flat design", 
                           # "(clean:1.3) lines", 
                           # "geometric shapes", 
                           # "simple icons", 
                           # "sans-serif fonts", 
                           # "(responsive:0.8) layout", 
                           # "(user-friendly:1.2) interface",
                           "mobile-first design", ]
DEFAULT_NEG_PROMPT_LIST = ["lowres", 
                           "bad anatomy", 
                           "bad hands", 
                           "error", 
                           "missing fingers", 
                           "extra digit", 
                           "fewer digits", 
                           "cropped", 
                           "worst quality", 
                           "low quality", 
                           "normal quality", 
                           "jpeg artifacts", 
                           "signature", 
                           "watermark", 
                           "username", 
                           "blurry"]
ENDPOINT_URL = 'http://127.0.0.1:7860/sdapi/v1/txt2img'

app = FastAPI()

@app.post("/generate")
async def generate(payload: InputPayload):
    if payload.preset in REF_IMAGES:
        ref_img_file = random.choice(REF_IMAGES[payload.preset])
    else:
        raise HTTPException(status_code=400, detail="invalid reference")
    ref_img_b64, width, height = _process_ref_img(ref_img_file)
    prompt = f"{','.join(DEFAULT_POS_PROMPT_LIST)},{payload.user_input} <lora:uiyemianv2:0.9>"
    negative_prompt = f"{','.join(DEFAULT_NEG_PROMPT_LIST)} verybadimagenegative_v1.3"
    controlnet_configs = _config_ctrlnet(ref_img_b64, payload.n_output)
    fwd_payloads = [SdFwdPaylodWithImage(prompt=prompt,
                                         negative_prompt=negative_prompt,
                                         width=width,
                                         height=height,
                                         alwayson_scripts=config)
                                         for config in controlnet_configs]
    imgs = _fwd_sd(ENDPOINT_URL, fwd_payloads)
    return OutputPayload(imgs=imgs)

def _fwd_sd(url: str, payloads: List[SdFwdPaylodWithImage]):
    imgs = []
    for payload in payloads:
        response = requests.post(url, data=payload.model_dump_json())
        for img in response.json()["images"]:
            imgs.append(img)
        imgs.pop()
    return imgs

def _process_ref_img(ref_img_file):
    with open(ref_img_file, 'rb') as f:
        ref_img = Image.open(f)
    width, height = _process_img_wh(ref_img)
    buffered = io.BytesIO()
    ref_img.save(buffered, format="JPEG")
    ref_img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return ref_img_b64, width, height

def _process_img_wh(img: Image):
    w, h = img.size
    vertical = True if w <= h else False
    shorter = w if vertical else h
    longer = h if vertical else w
    if longer >= 1024:
        shorter = 1024 * (w / h) if vertical else 1024 * (h / w)
        shorter = int(round(shorter, 0))
        longer = 1024
    return shorter if vertical else longer, longer if vertical else shorter

def _config_ctrlnet(ref_img_b64, n_images=1):
    weight_choices = [round(random.uniform(0.6, 1.6), 2) for i in range(n_images)]
    controlnet_args = [ControlNetArgs(input_image=ref_img_b64, weight=w) for w in weight_choices]
    controlnets = [ControlNet(args=[arg]) for arg in controlnet_args] 
    return controlnets