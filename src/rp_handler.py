import os
import base64
import concurrent.futures

import torch
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline, AutoencoderKL
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #

class ModelHandler:
    def __init__(self):
        self.inpaint_pipe = None
        self.base_pipe = None
        self.safety_checker = None
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

    def load_inpaint(self):
        if self.inpaint_pipe is None:
            self.inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "mrcuddle/urpm-inpaint-sdxl", vae=self.vae,
                torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True, add_watermarker=False
            ).to("cuda")
            self.inpaint_pipe.enable_xformers_memory_efficient_attention()
        return self.inpaint_pipe

    def load_base(self):
        if self.base_pipe is None:
            self.base_pipe = StableDiffusionXLPipeline.from_pretrained(
                "mrcuddle/urpm-inpaint-sdxl", vae=self.vae,
                torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True, add_watermarker=False
            ).to("cuda")
            self.base_pipe.enable_xformers_memory_efficient_attention()
        return self.base_pipe

    def load_safety_checker(self):
        if self.safety_checker is None:
            self.safety_checker = StableDiffusionXLPipeline.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=torch.float16
            ).to("cuda")
        return self.safety_checker

MODELS = ModelHandler()

@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image using inpainting or base pipeline with safety checking
    '''
    job_input = job["input"]
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    
    MODELS.load_inpaint()
    MODELS.load_base()
    MODELS.load_safety_checker()

    # Load images
    init_image = load_image(job_input['image_url']).convert("RGB")
    mask_image = load_image(job_input['mask_url']).convert("RGB")
    
    try:
        if job_input.get('use_inpaint', False):
            output = MODELS.inpaint_pipe(
                prompt=job_input['prompt'],
                negative_prompt=job_input['negative_prompt'],
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=job_input['num_inference_steps'],
                guidance_scale=job_input['guidance_scale'],
                generator=generator
            ).images
        else:
            output = MODELS.base_pipe(
                prompt=job_input['prompt'],
                negative_prompt=job_input['negative_prompt'],
                num_inference_steps=job_input['num_inference_steps'],
                guidance_scale=job_input['guidance_scale'],
                generator=generator
            ).images

        # Safety check
        safe_output = MODELS.safety_checker(
            images=output
        ).images

    except RuntimeError as err:
        return {
            "error": f"RuntimeError: {err}",
            "refresh_worker": True
        }
    
    return {
        "images": safe_output,
        "seed": job_input['seed']
    }

runpod.serverless.start({"handler": generate_image})
