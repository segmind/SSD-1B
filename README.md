# Segmind Stable Diffusion 1B (SSD-1B) Model Card

![Screenshot from 2023-11-01 23-23-41](https://github.com/segmind/SSD-1B/assets/82945616/b9321eb0-0eac-4105-8784-e1ab800ee200)

## Model Description

The Segmind Stable Diffusion Model (SSD-1B) is a distilled **50% smaller version** of the Stable Diffusion XL (SDXL), offering a **60% speedup** while maintaining high-quality text-to-image generation capabilities. It has been trained on diverse datasets, including Grit and Midjourney scrape data, to enhance its ability to create a wide range of visual content based on textual prompts.

This model employs a knowledge distillation strategy, where it leverages the teachings of several expert models in succession, including SDXL, ZavyChromaXL, and JuggernautXL, to combine their strengths and produce impressive visual outputs.

Special thanks to the HF team 🤗 especially [Sayak](https://huggingface.co/sayakpaul), [Patrick](https://github.com/patrickvonplaten) and [Poli](https://huggingface.co/multimodalart) for their collaboration and guidance on this work.

## Demo

Try out the model at [Segmind SSD-1B](https://www.segmind.com/models/ssd-1b) for ⚡ fastest inference. You can also try it on [🤗 Spaces](https://huggingface.co/spaces/segmind/Segmind-Stable-Diffusion)

## Image Comparision (SDXL-1.0 vs SSD-1B)

![image](https://github.com/segmind/SSD-1B/assets/82945616/a5583e8a-6a05-4680-a540-f80502feed0b)

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

### Please do use negative prompting for the best quality!

### Key Features

- **Text-to-Image Generation:** The model excels at generating images from text prompts, enabling a wide range of creative applications.

- **Distilled for Speed:** Designed for efficiency, this model offers a 60% speedup, making it a practical choice for real-time applications and scenarios where rapid image generation is essential.

- **Diverse Training Data:** Trained on diverse datasets, the model can handle a variety of textual prompts and generate corresponding images effectively.

- **Knowledge Distillation:** By distilling knowledge from multiple expert models, the Segmind Stable Diffusion Model combines their strengths and minimizes their limitations, resulting in improved performance.

### Model Architecture

The SSD-1B Model is a 1.3B Parameter Model which has several layers removed from the Base SDXL Model

![image](https://github.com/segmind/SSD-1B/assets/82945616/9e208225-2454-4b8c-b5e4-91e035c4208d)

### Training info

These are the key hyperparameters used during training:

* Steps: 251000
* Learning rate: 1e-5
* Batch size: 32
* Gradient accumulation steps: 4
* Image resolution: 1024
* Mixed-precision: fp16

### Speed Comparision

We have observed that SSD-1B is upto 60% faster than the Base SDXL Model. Below is a comparision on an A100 80GB.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/TyymF1OkUjXLrHUp1XF0t.png)

Below are the speed up metrics on a RTX 4090 GPU.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/moMZrlDr-HTFkZlqWHUjQ.png)

### Model Sources

The SSD-1B Model API can be accessed via [Segmind SSD-1B](https://www.segmind.com/models/ssd-1b). For more information and access details, please visit [Segmind](https://www.segmind.com/models/ssd).

### Downstream Use

The Segmind Stable Diffusion Model can also be used directly with the 🧨 Diffusers library training scripts for further training, including:

- **[Fine-Tune](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py):**
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
  --output_dir="sd-pokemon-model-lora-sdxl" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  --push_to_hub
```
- **[LoRA](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py):**
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
  --output_dir="sdxl-pokemon-model" \
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

### Direct Use

The Segmind Stable Diffusion Model is suitable for research and practical applications in various domains, including:

- **Art and Design:** It can be used to generate artworks, designs, and other creative content, providing inspiration and enhancing the creative process.

- **Education:** The model can be applied in educational tools to create visual content for teaching and learning purposes.

- **Research:** Researchers can use the model to explore generative models, evaluate its performance, and push the boundaries of text-to-image generation.

- **Safe Content Generation:** It offers a safe and controlled way to generate content, reducing the risk of harmful or inappropriate outputs.

- **Bias and Limitation Analysis:** Researchers and developers can use the model to probe its limitations and biases, contributing to a better understanding of generative models' behavior.

### Out-of-Scope Use

The SSD-1B Model is not suitable for creating factual or accurate representations of people, events, or real-world information. It is not intended for tasks requiring high precision and accuracy.

## Limitations and Bias

Limitations & Bias
The SSD-1B Model has some challenges in embodying absolute photorealism, especially in human depictions. While it grapples with incorporating clear text and maintaining the fidelity of complex compositions due to its autoencoding approach, these hurdles pave the way for future enhancements. Importantly, the model's exposure to a diverse dataset, though not a panacea for ingrained societal and digital biases, represents a foundational step towards more equitable technology. Users are encouraged to interact with this pioneering tool with an understanding of its current limitations, fostering an environment of conscious engagement and anticipation for its continued evolution.
