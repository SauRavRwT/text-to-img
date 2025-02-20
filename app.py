import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Model and device setup
model_id = "sd-legacy/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize pipeline with optimized defaults
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    use_auth_token=True if "sd-legacy" in model_id else False
).to(device)

# Use optimized DPMSolver++ scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    solver_order=2,
    predict_epsilon=True,
    thresholding=True,
    dynamic_thresholding_ratio=0.995
)

# Enable all optimizations
pipe.enable_attention_slicing(slice_size="auto")
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()

def calculate_adaptive_guidance(prompt, base_guidance):
    """Enhanced adaptive guidance calculation with term weighting"""
    prompt_lower = prompt.lower()
    
    # Extended guidance factors with refined weights
    term_categories = {
        "style": ['realistic', 'detailed', 'photographic', 'artistic', 'cartoon', 'anime', 'digital art', 'oil painting', 'watercolor', 'sketch', '3d render', 'cinematic', 'studio photo'],
        "color": ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'golden', 'silver', 'metallic', 'neon', 'pastel', 'vibrant', 'muted'],
        "composition": ['portrait', 'landscape', 'close-up', 'wide shot', 'aerial view', 'side view', 'front view', 'macro', 'ultra wide'],
        "lighting": ['sunlight', 'studio lighting', 'dramatic lighting', 'soft light', 'hard light', 'backlight', 'natural light']
    }
    
    # Calculate complexity with refined weights
    complexity = 1.0
    for category, terms in term_categories.items():
        matches = sum(term in prompt_lower for term in terms)
        if category == "style":
            complexity += matches * 0.4  # Higher weight for style terms
        elif category == "color":
            complexity += matches * 0.2  # Lower weight for color terms
        elif category == "composition":
            complexity += matches * 0.3  # Medium weight for composition terms
        elif category == "lighting":
            complexity += matches * 0.3  # Medium weight for lighting terms
    
    # Add length-based complexity
    complexity += len(prompt.split()) * 0.05
    
    # Apply logarithmic scaling to prevent excessive guidance
    complexity = min(complexity, 10.0)  # Cap complexity
    final_guidance = base_guidance * (1 + torch.log1p(torch.tensor(complexity - 1)).item())
    
    return min(max(final_guidance, 7.0), 25.0)

def generate_image(
    prompt,
    steps=50,
    guidance_base=24,
    width=768,
    height=768,
    seed=-1,
    use_adaptive_guidance=True
):
    try:
        # Dynamic negative prompt based on input
        negative_prompt = """
        multiple people, extra heads, multiple faces, multiple bodies, extra limbs, clones, duplicates, twin, group, crowd, second person, additional figures,(deformed body:1.4), (bad anatomy:1.4), (duplicate:1.5), (cloned:1.5), (repeating:1.5), (merged body parts:1.5), low quality, worst quality, bad quality, jpeg artifacts, compression artifacts
        low quality, worst quality, bad quality, jpeg artifacts, compression artifacts,
        blurry, ugly, deformed, mutated, distorted, disfigured, poorly drawn, amateur, messy, sloppy, unprofessional, broken, glitched, corrupted,
        (deformed body:1.4), (deformed face:1.4), (deformed limbs:1.4), (bad anatomy:1.4),
        bad proportions, wrong proportions, out of proportion, anatomical errors,
        extra fingers, missing fingers, fused fingers, too many fingers, mutated hands,
        extra limbs, missing limbs, floating limbs, disconnected limbs, broken limbs,
        extra joints, missing joints, broken joints, dislocated joints, mutated joints, dislocated bones, (dislocated arms and legs:1.3), (duplicated limbs:1.2),( duplicated body parts:1.4), (merged body parts: 1.5),
        (duplicate:1.5), (duplicated:1.5), (cloned:1.5), (repeating:1.5), (multiple:1.5),
        (clone artifacts:1.5), (repetitive:1.5), (duplicated elements:1.5),
        duplicate faces, cloned faces, multiple faces, copied faces,
        duplicate objects, cloned objects, (multiple objects:1.5), (copied objects:1.3), (duplicated limbs:1.2), duplicated body parts, (merged body parts: 1.5),
        bad composition, unbalanced composition, poor composition, amateurish composition,
        improper perspective, wrong perspective, bad perspective, distorted perspective,
        bad foreshortening, incorrect foreshortening, perspective errors,
        bad camera angle, wrong camera angle, tilted horizon, crooked horizon,
        bad lighting, harsh lighting, uneven lighting, poor lighting, incorrect shadows,
        wrong shadows, missing shadows, inconsistent lighting, lighting errors,
        bad exposure, overexposed, underexposed, blown out highlights, crushed blacks,
        color bleeding, color artifacts, wrong colors, unnatural colors,
        watermark, text, signature, logo, timestamp, border, frame,
        aliasing, pixelation, noise, grain, banding, moire patterns,
        chromatic aberration, lens distortion, vignetting, halation,
        inconsistent style, mixed styles, conflicting styles, wrong style,
        out of character, style break, aesthetic mismatch, artistic inconsistency
        """.replace('\n', ' ').replace('    ', '').replace('# ', '')

        # Enhanced positive prompt additions
        enhancement_prompt = """
        (masterpiece:1.2), (best quality:1.2), (ultra high resolution:1.2),
        (highly detailed:1.1), (sharp focus:1.1), (crystal clear:1.1),
        professional photography, studio quality, perfect composition,
        accurate proportions, precise details, beautiful lighting,
        exquisite texturing, proper anatomy, cohesive style,
        8k resolution, ultra HD, ray tracing, physically based rendering,
        professional color grading, perfect shadows and highlights, realistic rendering, cinematic quality, vibrant colors, stunning visuals, breathtaking scenery
        """.strip()

        # Calculate final guidance scale
        final_guidance = calculate_adaptive_guidance(prompt, guidance_base) if use_adaptive_guidance else guidance_base

        # Set random seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)

        # Generate image with optimized parameters
        image = pipe(
            prompt + ", " + enhancement_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=final_guidance,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=1,
        ).images[0]
        
        return image

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# Enhanced Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Image Generator with Stable Diffusion")
    gr.Markdown("### This demo uses the Stable Diffusion model to generate high-quality images from text prompts. The model is optimized for high performance and quality, with advanced features like adaptive guidance and negative prompts.")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Enter your prompt", lines=3, placeholder="Describe the image you want to generate...")
            steps = gr.Slider(30, 150, value=50, step=1, label="Quality Steps")
            guidance = gr.Slider(7, 30, value=22, step=0.5, label="Guidance Scale")
            width = gr.Slider(512, 1024, value=768, step=128, label="Width")
            height = gr.Slider(512, 1024, value=768, step=128, label="Height")
            seed = gr.Number(label="Seed (-1 for random)", value=-1)
            adaptive_guidance = gr.Checkbox(label="Use Adaptive Guidance", value=True, info="Automatically adjusts guidance based on prompt complexity")
            generate_button = gr.Button("Generate Image")
            random_prompt = gr.Button("Suprise Me!")
        with gr.Column(scale=1):
            output_image = gr.Image(label="AI Generated Image", type="pil")
            clear_button = gr.Button("Clear")
    
    # Example prompts
    prompt_suggestions = [
        "A majestic lion in the African savanna at sunset, detailed fur, golden hour lighting, realistic, 8k photography",
        "A futuristic cyberpunk city street at night, neon lights, rain reflections, highly detailed, cinematic lighting",
        "A serene Japanese garden with cherry blossoms, traditional architecture, soft natural lighting, spring season",
        "A professional portrait of a young woman, studio lighting, shallow depth of field, high fashion photography",
        "A magical forest scene with glowing mushrooms, fantasy elements, mystical atmosphere, volumetric lighting",
        "A realistic digital painting of a sci-fi spaceship, detailed textures, metallic surfaces, space background",
        "A low quality pixelated image with jpeg artifacts, blurry, distorted, poorly drawn, bad composition",
    ]
    
    # Event handlers
    generate_button.click(generate_image, inputs=[prompt, steps, guidance, width, height, seed, adaptive_guidance], outputs=output_image)
    random_prompt.click(lambda: random.choice(prompt_suggestions), outputs=prompt)
    clear_button.click(lambda: [None, 50, 20, 768, 768, -1, True], outputs=[prompt, steps, guidance, width, height, seed, adaptive_guidance])   

# Launch the interface
demo.launch(share=True, debug=True)