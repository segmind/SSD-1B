import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import UniPCScheduler

path = 'segmind/SDXL-Mini' #model card
prompt = "hyperrealistic glamour portrait of an old weary wizard surrounded by elemental magic, arcane, freckles, skin pores, pores, velus hair, macro, extreme details, looking at viewer"
negative_prompt = "sketch, cartoon, drawing, anime:1.4, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions"

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
with torch.inference_mode():
    pipe = StableDiffusionXLPipeline.from_pretrained(
      path, torch_dtype=torch.float16, use_safetensors=True
    )
    pipe.to(f"cuda")
    pipe.scheduler = UniPCScheduler.from_config(pipe.scheduler.config)
    pipe.unet.to(device=f"cuda", dtype=torch.float16, memory_format=torch.channels_last)
    img = pipe(prompt=prompt,negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale = 9, num_images_per_prompt=1,generator=gen).images[0]
    img.save(f"image.png")