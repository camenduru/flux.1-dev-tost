FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1

RUN git clone -b xlabs_flux_controlnet https://github.com/camenduru/ComfyUI /content/ComfyUI

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8-all-in-one.safetensors -d /content/ComfyUI/models/checkpoints -o flux1-dev-fp8-all-in-one.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/aesthetic10k.safetensors -d /content/ComfyUI/models/loras -o advokat_aesthetic10k.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_anime_lora.safetensors -d /content/ComfyUI/models/loras -o xlabs_anime.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_art_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_art.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_disney_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_disney.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_mjv6_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_mjv6.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_realism_lora_comfui.safetensors -d /content/ComfyUI/models/loras -o xlabs_realism.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_scenery_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_scenery.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py