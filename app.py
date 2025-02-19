import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize pipeline with optimized defaults
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
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

def calculate_adaptive_guidance(prompt, base_guidance):
    """Enhanced adaptive guidance calculation"""
    prompt_lower = prompt.lower()
    
    # Extended guidance factors
    style_terms = ['realistic', 'detailed', 'photographic', 'artistic', 'cartoon', 'anime', 'digital art', 'oil painting', 'watercolor', 'sketch', '3d render', 'cinematic', 'studio photo']
    color_terms = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'golden', 'silver', 'metallic', 'neon', 'pastel', 'vibrant', 'muted']
    composition_terms = ['portrait', 'landscape', 'close-up', 'wide shot', 'aerial view', 'side view', 'front view', 'macro', 'ultra wide']
    lighting_terms = ['sunlight', 'studio lighting', 'dramatic lighting', 'soft light', 'hard light', 'backlight', 'natural light']
    
    # Enhanced complexity calculation
    complexity = 1.0
    complexity += sum(term in prompt_lower for term in style_terms) * 0.35
    complexity += sum(term in prompt_lower for term in color_terms) * 0.25
    complexity += sum(term in prompt_lower for term in composition_terms) * 0.3
    complexity += sum(term in prompt_lower for term in lighting_terms) * 0.3
    complexity += len(prompt.split()) * 0.06
    
    # Optimized guidance scaling
    return min(max(base_guidance * complexity, 8.0), 25.0)

def generate_image(
    prompt,
    steps=50,
    guidance_base=20,
    width=768,
    height=768,
    seed=-1,
    use_adaptive_guidance=True
):
    # Comprehensive negative prompt
    negative_prompt = """
    # Critical Quality Issues
    low quality, worst quality, bad quality, jpeg artifacts, compression artifacts,
    blurry, ugly, deformed, mutated, distorted, disfigured, poorly drawn, amateur, messy, sloppy, unprofessional, broken, glitched, corrupted,
    
    # Anatomical Issues
    (deformed body:1.4), (deformed face:1.4), (deformed limbs:1.4), (bad anatomy:1.4),
    bad proportions, wrong proportions, out of proportion, anatomical errors,
    extra fingers, missing fingers, fused fingers, too many fingers, mutated hands,
    extra limbs, missing limbs, floating limbs, disconnected limbs, broken limbs,
    extra joints, missing joints, broken joints, dislocated joints, mutated joints, dislocated bones, (dislocated arms and legs:1.3), (duplicated limbs:1.2),( duplicated body parts:1.4), (merged body parts: 1.5),
    
    # Duplication and Repetition
    (duplicate:1.5), (duplicated:1.5), (cloned:1.5), (repeating:1.5), (multiple:1.5),
    (clone artifacts:1.5), (repetitive:1.5), (duplicated elements:1.5),
    duplicate faces, cloned faces, multiple faces, copied faces,
    duplicate objects, cloned objects, (multiple objects:1.5), (copied objects:1.3), (duplicated limbs:1.2), duplicated body parts, (merged body parts: 1.5),
    
    # Composition and Technical
    bad composition, unbalanced composition, poor composition, amateurish composition,
    improper perspective, wrong perspective, bad perspective, distorted perspective,
    bad foreshortening, incorrect foreshortening, perspective errors,
    bad camera angle, wrong camera angle, tilted horizon, crooked horizon,
    
    # Lighting and Color
    bad lighting, harsh lighting, uneven lighting, poor lighting, incorrect shadows,
    wrong shadows, missing shadows, inconsistent lighting, lighting errors,
    bad exposure, overexposed, underexposed, blown out highlights, crushed blacks,
    color bleeding, color artifacts, wrong colors, unnatural colors,
    
    # Additional Artifacts
    watermark, text, signature, logo, timestamp, border, frame,
    aliasing, pixelation, noise, grain, banding, moire patterns,
    chromatic aberration, lens distortion, vignetting, halation,
    
    # Style Inconsistencies
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
        generator = None

    # Enhanced prompt weighting
    prompt_elements = prompt.split(',')
    weighted_prompt = ""
    for i, element in enumerate(prompt_elements):
        element = element.strip()
        if i == 0:  # Main subject gets highest emphasis
            weighted_prompt += f"({element}:1.4)"  # Increased weight for main subject
        elif i == 1:  # Secondary elements get medium emphasis
            weighted_prompt += f", ({element}:1.2)"
        else:  # Tertiary elements get normal weight
            weighted_prompt += f", {element}"

    # Generate image with optimized parameters
    image = pipe(
        weighted_prompt + ", " + enhancement_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=final_guidance,
        width=width,
        height=height,
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]
    
    return image

# Enhanced Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=3, placeholder="Describe the image you want to generate..."),
        gr.Slider(30, 150, value=50, step=1, label="Quality Steps (higher = better quality)"),
        gr.Slider(7, 30, value=20, step=0.5, label="Base Guidance Scale (higher = stronger prompt adherence)"),
        gr.Slider(512, 1024, value=768, step=128, label="Width"),
        gr.Slider(512, 1024, value=768, step=128, label="Height"),
        gr.Number(label="Seed (-1 for random)", value=-1),
        gr.Checkbox(label="Use Adaptive Guidance", value=True, info="Automatically adjusts guidance based on prompt complexity")
    ],
    outputs=gr.Image(label="Generated Image", type="pil"),
    title="Professional Stable Diffusion Generator",
    description="""
    Advanced image generation with optimized settings for professional results:
    - Enhanced prompt handling with automatic emphasis adjustment
    - Comprehensive negative prompts to prevent common issues
    - Advanced quality control and guidance scaling
    - Optimized for preventing duplications and deformities
    """
)

interface.launch(share=True, debug=True)