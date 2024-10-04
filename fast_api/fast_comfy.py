# ---
# cmd: ["python", "06_gpu_and_ml/comfyui/comfyclient.py", "--modal-workspace", "modal-labs", "--prompt", "Spider-Man visits Yosemite, rendered by Blender, trending on artstation"]
# output-directory: "/tmp/comfyui"
# ---

import pathlib
import sys
import json
import time
import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import uvicorn
import aiohttp

OUTPUT_DIR = pathlib.Path("/tmp/comfyui")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)



model_workspace = os.getenv("MODAL_WORKSPACE", "thanabordeen")
APP_NAME = os.getenv("APP_NAME", "COMFYUI")
config_path = "./prompt.json"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
URL = f"{OLLAMA_HOST}/api/generate"
HEADERS = {'Content-type': 'application/json'}
with open(config_path, encoding="UTF-8") as f:
    DEFAULT_PROMPT = json.load(f)

# --- FastAPI App ---
app = FastAPI(
    title="Image to Image ComfyUI API",
    description="API for generating images using ComfyUI",
    version="1.0.0",
    docs_url="/docs",
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
class Receive(BaseModel):
    prompt: str
    image_base64: str
    img_name: str
    
@app.post("/generate-prompt")
async def generate_prompt(receive: Receive):

    prompt_data = DEFAULT_PROMPT.copy() 
    prompt_data["prompt"] = receive.prompt

    try:
        response = await make_post_request(URL, prompt_data, HEADERS)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {e}")

    prompt_info = extract_prompt_and_negative_prompt(json.loads(response)['response'])
    pos_prompt = prompt_info["prompt"]
    neg_prompt = prompt_info["negative_prompt"]

    images_list = await generate_image(pos_prompt, neg_prompt, receive.image_base64, 0.59, receive.img_name)
    data ={"images": images_list}
        
    return data

    
    
    
async def generate_image(pos_prompt: str, neg_prompt: str, image: str, effect_rate: float, image_name: str, dev: bool = True):

    url = f"https://{model_workspace}--{APP_NAME}-comfyui-api{'-dev' if dev else ''}.modal.run/"
    data = {
        "pos_prompt": pos_prompt,
        "neg_prompt": neg_prompt,
        "image_data": image,
        "effect_rate": effect_rate,
        "image_name": image_name
    }
    print(f"Sending request to {url} with prompt: {data['pos_prompt']}")
    print("Waiting for response...")
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as res:
            print(f"Response received in {time.time() - start_time} seconds.")
            if res.status != 200:
                raise HTTPException(status_code=res.status, detail=await res.text())
            try:
                images = await res.content.read()
                images = json.loads(images) # This make it's slow to load
            except aiohttp.ContentTypeError as e:
                raise HTTPException(status_code=500, detail=f"Error parsing response JSON: {e}")
            return images
            
        
async def make_post_request(url, prompt, headers):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=json.dumps(prompt), headers=headers) as response:
            return await response.text()
        
def extract_prompt_and_negative_prompt(response: str):
    import re
    prompt_match = re.search(r'<Prompt>(.*?)</Prompt>', response, re.DOTALL)
    negative_prompt_match = re.search(r'<Negative Prompt>(.*?)</Negative Prompt>', response, re.DOTALL)
    return {
        "prompt": prompt_match.group(1).strip() if prompt_match else None,
        "negative_prompt": negative_prompt_match.group(1).strip() if negative_prompt_match else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)