import torch
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub with retry logic.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL inpainting pipeline for all use cases.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    inpaint_pipe = fetch_pretrained_model(
        StableDiffusionXLInpaintPipeline, "mrcuddle/urpm-inpaint-sdxl", **common_args
    )
    vae = fetch_pretrained_model(
        AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    print("Loaded inpainting pipeline for all use cases successfully.")
    return inpaint_pipe, vae


if __name__ == "__main__":
    get_diffusion_pipelines()
