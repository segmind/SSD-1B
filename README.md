# Segmind Stable Diffusion 1B (SSD-1B) Model Card

![ssd1b_panda_cover](https://github.com/segmind/SSD-1B/assets/82945616/da12cc15-6b88-445e-bc9a-0c0aa54755f1)

## Model Description

The Segmind Stable Diffusion Model (SSD-1B) is a distilled 50% smaller version of the Stable Diffusion XL (SDXL), offering a 60% speedup while maintaining high-quality text-to-image generation capabilities. It has been trained on diverse datasets, including Grit and Midjourney scrape data, to enhance its ability to create a wide range of visual content based on textual prompts.

This model employs a knowledge distillation strategy, where it leverages the teachings of several expert models in succession, including SDXL, ZavyChromaXL, and JuggernautXL, to combine their strengths and produce impressive visual outputs.

Special thanks to the HF team 🤗 especially [Sayak](https://huggingface.co/sayakpaul), [Patrick](https://github.com/patrickvonplaten) and [Poli](https://huggingface.co/multimodalart) for their collaboration and guidance on this work.

## Demo

Try out the model at [Segmind SSD-1B](https://www.segmind.com/models/ssd-1b) for ⚡ fastest inference. You can also try it on [🤗 Spaces](https://huggingface.co/spaces/segmind/Segmind-Stable-Diffusion)

## Image Comparision (SDXL-1.0 vs SSD-1B)

![image](https://github.com/segmind/SSD-1B/assets/82945616/a149add3-cb8a-4b24-82e5-bf59dd0949b0)

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

![image](https://github.com/segmind/SSD-1B/assets/82945616/4df4b27f-187b-43a9-a2df-b4d808e9262f)


### Training info

These are the key hyperparameters used during training:

* Steps: 251000
* Learning rate: 1e-5
* Batch size: 32
* Gradient accumulation steps: 4
* Image resolution: 1024
* Mixed-precision: fp16

### Speed Comparision

We have observed that SSD-1B is upto 60% faster than the Base SDXL Model. Below is a comparision on an A100 40GB.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62039c2d91d53938a643317d/f7BcTrz5PjYGC5htLUVge.png)

### Model Sources

The SSD-1B Model API can be accessed via the Segmind. For more information and access details, please visit [Segmind](https://www.segmind.com/models/ssd).

## Uses


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

### Limitations

- **Photorealism:** The model does not achieve perfect photorealism and may produce images with artistic or stylized qualities.

- **Legible Text:** Generating legible text within images is a challenge for the model, and text within images may appear distorted or unreadable.

- **Compositionality:** Complex tasks involving composition, such as rendering images based on intricate descriptions, may pose challenges for the model.

- **Faces and People:** While the model can generate a wide range of content, it may not consistently produce realistic or high-quality images of faces and people.

- **Lossy Autoencoding:** The autoencoding aspect of the model is lossy, which means that some details in the input text may not be perfectly retained in the generated images.

### Bias

The SSD-1B Model is trained on a diverse dataset, but like all generative models, it may exhibit biases present in the training data. Users are encouraged to be mindful of potential biases in the model's outputs and take appropriate steps to mitig
