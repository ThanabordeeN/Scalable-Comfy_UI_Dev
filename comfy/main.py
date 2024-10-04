import json
import subprocess
from pathlib import Path
from typing import Dict
import base64
import os
import modal

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("comfy-cli==1.2.4")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
    .run_commands(  # download the flux models
        "comfy --skip-prompt model download --url https://huggingface.co/a34384300/XSarchitectural-InteriorDesign-ForXSLora/resolve/main/xsarchitectural_v11.ckpt --relative-path models/checkpoints",
    )
    .run_commands(  # download a custom node
        "comfy node install Comfy-Photoshop-SD"
    )
    # can layer additional models and custom node downloads as needed
)
APP_NAME = os.getenv("APP_NAME", "exep-comfyui")

app = modal.App(name=APP_NAME, image=image)


# For more on how to run web services on Modal, check out [this guide](https://modal.com/docs/guide/webhooks).
@app.cls(
    allow_concurrent_inputs=10,
    container_idle_timeout=2,
    gpu="A10G",
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "img2img_comfy.json",
            "/root/img2img_comfy.json",
        ),
    ],
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/img2img_comfy.json"):
        # runs the comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200"
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        # completed workflows write output images to this directory
        output_dir = "/root/comfy/ComfyUI/output"
        print(f"Output images saved to {output_dir}")
        # looks up the name of the output image file based on the workflow
        workflow = json.loads(Path(workflow_path).read_text())
        file_prefix = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
        ][0]["filename_prefix"]
        images_list= []
        # returns the image as bytes
        for f in Path(output_dir).iterdir():
            if f.name.startswith(file_prefix):
                images_list.append(f)
        
        # Ensure that the list is not empty before attempting to read bytes
        if images_list:
            return [base64.b64encode(f.read_bytes()).decode('utf-8') for f in images_list]
        else:
            raise FileNotFoundError(f"No output images found with prefix {file_prefix}")

    @modal.web_endpoint(method="POST")
    def api(self, item: Dict):
        from fastapi import Response
        import random

        workflow_data = json.loads(
            (Path(__file__).parent / "img2img_comfy.json").read_text()
        )

        # insert the prompt

        workflow_data["6"]["inputs"]["text"] = item["pos_prompt"]
        workflow_data["7"]["inputs"]["text"] = item["neg_prompt"]
        workflow_data["3"]["inputs"]["seed"] = random.randint(0, 99999999999999)
        workflow_data["17"]["inputs"]["image_base64"] = item["image_data"]
        workflow_data["3"]["inputs"]["denoise"] = item["effect_rate"]
        
        # give the output image a unique id per client request
        client_id = item["image_name"]
        workflow_data["9"]["inputs"]["filename_prefix"] = client_id

        # save this updated workflow to a new file
        new_workflow_file = f"{client_id}.json"
        # json.dump(workflow_data, Path(new_workflow_file).open("w"))
        with open(new_workflow_file, 'w') as f:
            json.dump(workflow_data, f)

        # run inference on the currently running container
        img_base_64 = self.infer.local(f"/root/{new_workflow_file}")

        
        return Response(content=json.dumps(img_base_64), status_code=200)


# ### The workflow for developing workflows
#
# When you run this script with `modal deploy 06_gpu_and_ml/comfyui/comfyapp.py`, you'll see a link that includes `ui`.
# Head there to interactively develop your ComfyUI workflow. All of your models and custom nodes specified in the image build step will be loaded in.
#
# To serve the workflow after you've developed it, first export it as "API Format" JSON:
# 1. Click the gear icon in the top-right corner of the menu
# 2. Select "Enable Dev mode Options"
# 3. Go back to the menu and select "Save (API Format)"
#
# Save the exported JSON to the `workflow_api.json` file in this directory.
#
# Then, redeploy the app with this new workflow by running `modal deploy 06_gpu_and_ml/comfyui/comfyapp.py` again.
#
# ## Further optimizations
# - To decrease inference latency, you can process multiple inputs in parallel by setting `allow_concurrent_inputs=1`, which will run each input on its own container. This will reduce overall response time, but will cost you more money. See our [Scaling ComfyUI](https://modal.com/blog/scaling-comfyui) blog post for more details.
# - If you're noticing long startup times for the ComfyUI server (e.g. >30s), this is likely due to too many custom nodes being loaded in. Consider breaking out your deployments into one App per unique combination of models and custom nodes.
# - For those who prefer to run a ComfyUI workflow directly as a Python script, see [this blog post](https://modal.com/blog/comfyui-prototype-to-production).
