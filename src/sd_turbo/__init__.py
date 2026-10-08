import torch
from diffusers import AutoPipelineForText2Image

print("cargardo modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
)

modelo = modelo.to("cpu")


prompt = input("Escribe el prompt de la imagen:")
negative_prompt ="blurry, low resolution, low quality, jpeg artifacts, bad anatomy, deformed hands, extra fingers, missing limbs, poorly drawn face, watermark, signature, text, username, error"

print("Generando la imagen...")

imagen = modelo(
    prompt = prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10,
    guidance_scale=2.0,
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")
print("imagen guardada")
