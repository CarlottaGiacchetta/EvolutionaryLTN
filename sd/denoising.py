import os
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel
import torch
import logging
from network_rrdbnet import RRDBNet as net  # Importa la rete per BSRGAN
import utils_image as util  # Funzioni utili per la gestione delle immagini


DEVICE = "mps"

ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# TEXT TO IMAGE
#prompt = "A beautiful renaissance woman with a highly detailed dress"
prompt = "image of a beatiful dog"
prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified"
prompt = "A happy child playing volleyball on a sunny day. The child is wearing a bright, colorful sports outfit and is jumping to hit the ball over the net. The scene takes place on a sandy volleyball court, with a few friends cheering in the background."
uncond_prompt = ""  # Prompt negativo
do_cfg = True
cfg_scale = 8  # Scala di configurazione

# IMAGE TO IMAGE
input_image = None
image_path = "images/dog.jpg"
strength = 0.9

# SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image, list_of_images = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Save the generated image
output_image_pil = Image.fromarray(output_image)
output_image_pil.save("images/bimbo/tmp_image_50.png.png")

# Similarity Check using CLIP
# ---------------------------------

# Import CLIPModel and CLIPProcessor
from transformers import CLIPModel, CLIPProcessor

# Load pre-trained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
import numpy as np

from skimage import img_as_float
from skimage.restoration import denoise_tv_chambolle
import numpy as np
def compute_total_variation(image):
    # Converti l'immagine in float se non lo è già
    image = img_as_float(image)
    
    # Applica il denoising con TV Chambolle per ottenere una versione denoised
    denoised = denoise_tv_chambolle(image, weight=0.1)
    
    # Calcola la Total Variation come somma delle differenze assolute
    tv = np.sum(np.abs(image - denoised))
    return tv


for i in range(len(list_of_images)):
    #list_of_images[i] = Image.fromarray(list_of_images[i])
    img = list_of_images[i]

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Preprocess the image and the prompt
    inputs = clip_processor(text=prompt, images=img, return_tensors="pt", padding=True).to(DEVICE)

    # Forward pass through CLIP
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # Get the image and text embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize the embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.matmul(text_embeds, image_embeds.t()).squeeze().item()

    print(f"CLIP similarity between the prompt and the image: {similarity:.4f}")
    print(f"Total Variation: {compute_total_variation(np.array(img))}")
