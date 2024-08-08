import os, json, requests, runpod

import random
import torch
import numpy as np
from PIL import Image
import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_custom_sampler
from comfy_extras import nodes_flux
from comfy import model_management

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

with torch.inference_mode():
    unet, clip, vae = CheckpointLoaderSimple.load_checkpoint("flux1-dev-fp8-all-in-one.safetensors")

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values['positive_prompt']
    width = values['width']
    height = values['height']
    seed = values['seed']
    steps = values['steps']
    guidance = values['guidance']
    lora_strength_model = values['lora_strength_model']
    lora_strength_clip = values['lora_strength_clip']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']

    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    global unet, clip
    unet, clip = LoraLoader.load_lora(unet, clip, "flux_realism_lora.safetensors", lora_strength_model, lora_strength_clip)
    cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    cond = FluxGuidance.append(cond, guidance)[0]
    noise = RandomNoise.get_noise(seed)[0]
    guider = BasicGuider.get_guider(unet, cond)[0]
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
    latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
    sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/flux.png")

    result = "/content/flux.png"
    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})