import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
def generate_image(prompt, steps=30, guidance=7.5):
    # Enhanced negative prompt for better results
    negative_prompt = "ugly, blurry, bad quality, distorted, deformed, low resolution, worst quality, low quality, jpeg artifacts, watermark, text, signature, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, bad faces, broken arm and legs, deformed body"
    
    # Generate image with improved parameters
    image = pipe(
        prompt + ", high quality, detailed, sharp focus, professional, accurate, clear, realistic, high resolution, high definition, high res, high def",
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=768,
        height=768
    ).images[0]
    return image
    return image

gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter Prompt"),
        gr.Slider(20, 50, value=30, step=1, label="Inference Steps"),  # Increased default steps
        gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")  # Wider range, better default
    ],
    outputs="image"
).launch(share=True, debug=True)