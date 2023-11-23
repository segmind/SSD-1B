
# Segmind Stable Diffusion 1B (SSD-1B) Model Card

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/WveKcu7q5PyZEwNezyyMC.png)

## 🔥🔥Join our [Discord](https://discord.gg/rF44ueRG) to give feedback on our smaller v2 version and early access🔥🔥

## 📣 AUTOMATIC1111 compatibility added. Supporting file [here](https://huggingface.co/segmind/SSD-1B/blob/main/SSD-1B-A1111.safetensors)

## Demo

Try out the model at [Segmind SSD-1B](https://www.segmind.com/models/ssd-1b?utm_source=hf) for ⚡ fastest inference. You can also try it on [🤗 Spaces](https://huggingface.co/spaces/segmind/Segmind-Stable-Diffusion)

## Model Description

The Segmind Stable Diffusion Model (SSD-1B) is a **distilled 50% smaller** version of the Stable Diffusion XL (SDXL), offering a **60% speedup** while maintaining high-quality text-to-image generation capabilities. It has been trained on diverse datasets, including Grit and Midjourney scrape data, to enhance its ability to create a wide range of visual content based on textual prompts.

This model employs a knowledge distillation strategy, where it leverages the teachings of several expert models in succession, including SDXL, ZavyChromaXL, and JuggernautXL, to combine their strengths and produce impressive visual outputs.

Special thanks to the HF team 🤗 especially [Sayak](https://huggingface.co/sayakpaul), [Patrick](https://github.com/patrickvonplaten) and [Poli](https://huggingface.co/multimodalart) for their collaboration and guidance on this work.

## Image Comparision (SDXL-1.0 vs SSD-1B)

<img width="1257" alt="mOM_OMxbivVBELad1QQYj" src="https://github.com/segmind/SSD-1B/assets/95569637/c8638787-f8c1-4845-a9e3-1a826cc22cef">

## Usage:
This model can be used via the 🧨 Diffusers library. 

Make sure to install diffusers from source by running
```
pip install git+https://github.com/huggingface/diffusers
```

In addition, please install `transformers`, `safetensors` and `accelerate`:
```
pip install transformers accelerate safetensors
```

To use the model, you can run the following:

```py
from diffusers import StableDiffusionXLPipeline
import torch
pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
prompt = "An astronaut riding a green horse" # Your prompt here
neg_prompt = "ugly, blurry, poor quality" # Negative prompt here
image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
```
### Update: Our model should now be usable in ComfyUI.
### Please do use negative prompting, and a CFG around 9.0 for the best quality!
### Model Description

- **Developed by:** [Segmind](https://www.segmind.com/)
- **Developers:** [Yatharth Gupta](https://huggingface.co/Warlord-K) and [Vishnu Jaddipal](https://huggingface.co/Icar).
- **Model type:** Diffusion-based text-to-image generative model
- **License:** Apache 2.0
- **Distilled From** [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)


### Key Features

- **Text-to-Image Generation:** The model excels at generating images from text prompts, enabling a wide range of creative applications.

- **Distilled for Speed:** Designed for efficiency, this model offers a 60% speedup, making it a practical choice for real-time applications and scenarios where rapid image generation is essential.

- **Diverse Training Data:** Trained on diverse datasets, the model can handle a variety of textual prompts and generate corresponding images effectively.

- **Knowledge Distillation:** By distilling knowledge from multiple expert models, the Segmind Stable Diffusion Model combines their strengths and minimizes their limitations, resulting in improved performance.

### Model Architecture

The SSD-1B Model is a 1.3B Parameter Model which has several layers removed from the Base SDXL Model

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/Qa8Ow-moLQhOvzp-5kGt4.png)

### Training info

These are the key hyperparameters used during training:

* Steps: 251000
* Learning rate: 1e-5
* Batch size: 32
* Gradient accumulation steps: 4
* Image resolution: 1024
* Mixed-precision: fp16

### Multi-Resolution Support

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/IwIaIB4nBdMx6Vs5q82cL.jpeg)

SSD-1B can support the following output resolutions.

* 1024 x 1024 (1:1 Square)
* 1152 x 896 (9:7)
* 896 x 1152 (7:9)
* 1216 x 832 (19:13)
* 832 x 1216 (13:19)
* 1344 x 768 (7:4 Horizontal)
* 768 x 1344 (4:7 Vertical)
* 1536 x 640 (12:5 Horizontal)
* 640 x 1536 (5:12 Vertical)
    

### Speed Comparision

We have observed that SSD-1B is upto 60% faster than the Base SDXL Model. Below is a comparision on an A100 80GB.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/TyymF1OkUjXLrHUp1XF0t.png)

Below are the speed up metrics on a RTX 4090 GPU.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/moMZrlDr-HTFkZlqWHUjQ.png)

### Model Sources

For research and development purposes, the SSD-1B Model can be accessed via the Segmind AI platform. For more information and access details, please visit [Segmind](https://www.segmind.com/models/ssd-1b).

## Uses


### Direct Use

The Segmind Stable Diffusion Model is suitable for research and practical applications in various domains, including:

- **Art and Design:** It can be used to generate artworks, designs, and other creative content, providing inspiration and enhancing the creative process.

- **Education:** The model can be applied in educational tools to create visual content for teaching and learning purposes.

- **Research:** Researchers can use the model to explore generative models, evaluate its performance, and push the boundaries of text-to-image generation.

- **Safe Content Generation:** It offers a safe and controlled way to generate content, reducing the risk of harmful or inappropriate outputs.

- **Bias and Limitation Analysis:** Researchers and developers can use the model to probe its limitations and biases, contributing to a better understanding of generative models' behavior.

### Downstream Use

The Segmind Stable Diffusion Model can also be used directly with the 🧨 Diffusers library training scripts for further training, including:

- **[LoRA](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py):**
```bash
export MODEL_NAME="segmind/SSD-1B"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora-ssd" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  --push_to_hub
```

- **[Fine-Tune](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py):**
```bash
export MODEL_NAME="segmind/SSD-1B"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="ssd-pokemon-model" \
  --push_to_hub
```
- **[Dreambooth LoRA](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py):**
```bash
export MODEL_NAME="segmind/SSD-1B"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

### Out-of-Scope Use

The SSD-1B Model is not suitable for creating factual or accurate representations of people, events, or real-world information. It is not intended for tasks requiring high precision and accuracy.

## Limitations and Bias

Limitations & Bias
The SSD-1B Model has some challenges in embodying absolute photorealism, especially in human depictions. While it grapples with incorporating clear text and maintaining the fidelity of complex compositions due to its autoencoding approach, these hurdles pave the way for future enhancements. Importantly, the model's exposure to a diverse dataset, though not a panacea for ingrained societal and digital biases, represents a foundational step towards more equitable technology. Users are encouraged to interact with this pioneering tool with an understanding of its current limitations, fostering an environment of conscious engagement and anticipation for its continued evolution.
