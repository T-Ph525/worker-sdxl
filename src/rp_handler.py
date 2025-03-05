import os
import base64
import torch
import concurrent.futures
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ----------------------------- Model Handler ----------------------------- #

class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "John6666/uber-realistic-porn-merge-xl-urpmxl-v6final-sdxl", 
            vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_refiner(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", 
            vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        refiner_pipe.to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)
            self.base = future_base.result()
            self.refiner = future_refiner.result()

MODELS = ModelHandler()

# ----------------------------- Helper Functions ----------------------------- #

def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

# ----------------------------- Image Generation ----------------------------- #

@torch.inference_mode()
def generate_image(job):
    job_input = job["input"]
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input.get('image_url')
    job_input['seed'] = job_input.get('seed', int.from_bytes(os.urandom(2), "big"))
    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    MODELS.base.scheduler = make_scheduler(job_input['scheduler'], MODELS.base.scheduler.config)

    try:
        if starting_image:
            init_image = load_image(starting_image).convert("RGB")
            output = MODELS.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=init_image,
                generator=generator
            ).images
        else:
            image = MODELS.base(
                prompt=job_input['prompt'],
                negative_prompt=job_input['negative_prompt'],
                height=job_input['height'],
                width=job_input['width'],
                num_inference_steps=job_input['num_inference_steps'],
                guidance_scale=job_input['guidance_scale'],
                denoising_end=job_input['high_noise_frac'],
                output_type="latent",
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images

            output = MODELS.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=image,
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "CUDA out of memory. Try reducing image size or steps."}
    except RuntimeError as err:
        return {"error": f"RuntimeError: {err}"}

    image_urls = _save_and_upload_images(output, job['id'])
    results = {"images": image_urls, "image_url": image_urls[0], "seed": job_input['seed']}

    if starting_image:
        results['refresh_worker'] = True

    return results

runpod.serverless.start({"handler": generate_image})
