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
from skimage import img_as_float
from skimage.restoration import denoise_tv_chambolle
import numpy as np

ALLOW_CUDA = False
ALLOW_MPS = False
GENERATE = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
else: 
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# TEXT TO IMAGE
#prompt = "A beautiful renaissance woman with a highly detailed dress"
prompt = "image of a beatiful dog"
prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified"
prompt = "A happy child playing volleyball on a sunny day. The child is wearing a bright, colorful sports outfit and is jumping to hit the ball over the net. The scene takes place on a sandy volleyball court, with a few friends cheering in the background."
uncond_prompt = ""  # Prompt negativo
do_cfg = True
cfg_scale = 8  # Scala di configurazione


if GENERATE:
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

else:
    dir = 'images/cane/'
    list_of_images = []

    # Estensioni accettate
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    # Itera sui file nella directory
    for filename in os.listdir(dir):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(dir, filename)
            if filename == 'tmp_image_0.png':
                print('salvo immagine iniziale')
                init_img = Image.open(file_path)
            try:
                # Apri l'immagine
                img = Image.open(file_path)
                list_of_images.append(img)
                print(f"Aperta immagine: {filename}")
            except Exception as e:
                print(f"Errore nell'apertura di {filename}: {e}")



# Similarity Check using CLIP and total variation
# ---------------------------------


def compute_total_variation(image):
    # Converti l'immagine in float se non lo è già
    image = img_as_float(image)
    
    # Applica il denoising con TV Chambolle per ottenere una versione denoised
    denoised = denoise_tv_chambolle(image, weight=0.1)
    
    # Calcola la Total Variation come somma delle differenze assolute
    tv = np.sum(np.abs(image - denoised))
    return tv

def fitness(sim, tv):
    fitness = - 2 * sim*100 + tv/1000
    return fitness

def fitness1(tv_iniziale, sim, tv):
    delta_tv = tv-tv_iniziale
    print(delta_tv)
    fitness = sim + delta_tv
    return -fitness, delta_tv


# Nome del file di testo in cui salvare i risultati
output_file = os.path.join(dir, "fitness_results.txt")
output_file1 = os.path.join(dir, "fitness_results1.txt")


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
    total_variation = compute_total_variation(np.array(img))
    fitness_value = fitness(similarity, total_variation)
    fitness_value1, delta_tv = fitness1(compute_total_variation(np.array(init_img)), similarity, total_variation)

    with open(output_file, "w") as f:
        # Scrivi i risultati nel file
        f.write(f"Image {i + 1}:\n")
        f.write(f"CLIP similarity: {similarity:.4f}\n")
        f.write(f"Total Variation: {total_variation}\n")
        f.write(f"Fitness: {fitness_value}\n\n")

    # Apri il file in modalità scrittura
    with open(output_file1, "w") as f:
        # Scrivi i risultati nel file
        f.write(f"Image {i + 1}:\n")
        f.write(f"CLIP similarity: {similarity:.4f}\n")
        f.write(f"Total Variation: {total_variation}\n")
        f.write(f"Fitness: {fitness_value1}\n\n")
        f.write(f"Fitness: {delta_tv}\n\n")
    

print(f"Results written to file {output_file}.")
