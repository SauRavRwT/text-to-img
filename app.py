import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
import random
import logging
from functools import partial
from typing import Optional, Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StableDiffusionGenerator:
    """A class to handle Stable Diffusion image generation with optimizations."""
    
    def __init__(self, model_id: str = "sd-legacy/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self._setup_token()
        self._initialize_pipeline()
        
    def _setup_token(self):
        """Set up Hugging Face token from environment or user input."""
        # Check if running in Colab
        try:
            IN_COLAB = 'google.colab' in str(get_ipython())
        except NameError:
            IN_COLAB = False
            
        if IN_COLAB:
            from google.colab import userdata
            try:
                self.hf_token = userdata.get('HF_TOKEN')
            except Exception:
                self.hf_token = None

            if not self.hf_token:
                from getpass import getpass
                self.hf_token = getpass('Enter your Hugging Face token: ')

                try:
                    userdata.set('HF_TOKEN', self.hf_token)
                    logger.info("Token saved successfully!")
                except Exception as e:
                    logger.error(f"Could not save token: {e}")
        else:
            self.hf_token = os.getenv('HF_TOKEN')

        if not self.hf_token:
            raise ValueError("Please provide a Hugging Face token via environment variable HF_TOKEN")
    
    def _initialize_pipeline(self):
        """Initialize and optimize the Stable Diffusion pipeline."""
        logger.info(f"Initializing pipeline with model {self.model_id} on {self.device}")
        
        # Enable CUDA optimizations if available
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Initialize pipeline with optimized settings
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            token=self.hf_token
        ).to(self.device)
        
        # Use optimized scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            predict_epsilon=True,
            thresholding=True,
            dynamic_thresholding_ratio=0.995
        )
        
        # Apply optimizations
        self.optimizations()
        
        logger.info("Pipeline initialized successfully")
    
    def optimizations(self):
        """Apply various optimizations to the pipeline."""
        # Common optimizations
        self.pipe.enable_attention_slicing(slice_size="auto")
        
        # CUDA-specific optimizations
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers optimization enabled")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {e}")
                
            self.pipe.enable_vae_slicing()
            
            # Memory optimization (optional, as they can slow down inference)
            if torch.cuda.get_device_properties(0).total_memory < 8 * 1024 * 1024 * 1024:  # Less than 8GB VRAM
                logger.info("Enabling CPU offload for low VRAM setup")
                self.pipe.enable_sequential_cpu_offload()
            else:
                logger.info("Sufficient VRAM detected, using model_cpu_offload")
                self.pipe.enable_model_cpu_offload()
    
    def adaptive_guidance(self, prompt: str, base_guidance: float) -> float:
        """Calculate adaptive guidance scale based on prompt complexity."""
        prompt_lower = prompt.lower()

        # Term categories with weights
        term_categories = {
            "style": {
                "weight": 0.4,
                "terms": ['realistic', 'detailed', 'photographic', 'artistic', 'cartoon', 'anime', 
                         'digital art', 'oil painting', 'watercolor', 'sketch', '3d render', 'cinematic', 
                         'studio photo']
            },
            "color": {
                "weight": 0.2,
                "terms": ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 
                         'golden', 'silver', 'metallic', 'neon', 'pastel', 'vibrant', 'muted']
            },
            "composition": {
                "weight": 0.3,
                "terms": ['portrait', 'landscape', 'close-up', 'wide shot', 'aerial view', 'side view', 
                         'front view', 'macro', 'ultra wide']
            },
            "lighting": {
                "weight": 0.3,
                "terms": ['sunlight', 'studio lighting', 'dramatic lighting', 'soft light', 'hard light', 
                         'backlight', 'natural light']
            },
            "ui_design": {
                "weight": 0.35,
                "terms": ['interface', 'ui', 'ux', 'website', 'app', 'dashboard', 'mockup', 'wireframe', 
                         'layout', 'responsive', 'mobile', 'web design', 'minimal', 'modern']
            },
            "ui_elements": {
                "weight": 0.35,
                "terms": ['button', 'card', 'menu', 'navigation', 'sidebar', 'footer', 'header', 'modal', 
                         'form', 'input', 'slider', 'dropdown', 'icon', 'grid']
            },
            "design_style": {
                "weight": 0.3,
                "terms": ['glassmorphism', 'neumorphism', 'material design', 'flat design', 'skeuomorphism', 
                         'minimalist', 'brutalist', 'metro style']
            },
            "design_properties": {
                "weight": 0.3,
                "terms": ['gradient', 'shadow', 'rounded', 'transparent', 'blur', 'dark mode', 'light mode', 
                         'responsive', 'grid-based']
            }
        }

        # Calculate complexity
        complexity = 1.0
        for category, config in term_categories.items():
            matches = sum(term in prompt_lower for term in config["terms"])
            complexity += matches * config["weight"]

        # Add length-based complexity
        complexity += len(prompt.split()) * 0.05

        # Apply logarithmic scaling with clamping
        complexity = min(complexity, 10.0)  # Cap complexity
        final_guidance = base_guidance * (1 + torch.log1p(torch.tensor(complexity - 1)).item())

        # Clamp to reasonable range
        final_guidance = min(max(final_guidance, 7.0), 25.0)
        
        logger.info(f"Calculated guidance: {final_guidance:.2f} (base: {base_guidance}, complexity: {complexity:.2f})")
        return final_guidance
    
    def get_example_prompts(self) -> List[str]:
        """Return flat list of example prompts."""
        return [
            # UI Design Systems
            "Modern design system components, light theme, organized grid layout, typography hierarchy, input fields and buttons, minimalist style",
            "Material Design 3.0 component library, rounded corners, elevation shadows, floating action buttons, system bars and navigation",
            "iOS 16 style UI kit, blur effects, dynamic island, widgets collection, control center elements, system components",

            # Web Layouts
            "SaaS product landing page, hero section with 3D elements, feature grid, testimonials section, pricing cards, modern web design",
            "Portfolio website design, masonry grid gallery, minimalist navigation, project cards, smooth transitions, creative layout",
            "Blog platform interface, clean typography, article cards, sidebar widgets, newsletter signup form, reading progress bar",

            # Dashboards & Analytics
            "Analytics dashboard dark theme, data visualization widgets, metric cards, line charts and bar graphs, admin panel layout",
            "Finance app dashboard, crypto widgets, stock charts, wallet interface, transaction history, modern fintech design",
            "Project management tool interface, kanban board layout, task cards, team collaboration features, calendar integration",

            # Mobile Interfaces
            "Social media app UI, stories carousel, feed layout, navigation tabs, profile screen, interaction buttons",
            "Food delivery app interface, restaurant cards, order process flow, cart interface, payment screens, tracking view",
            "Fitness tracking app UI, workout planner, progress charts, achievement badges, profile statistics, dark theme",

            # E-commerce
            "E-commerce mobile app, product grid view, shopping cart, checkout flow, payment form, order confirmation",
            "Fashion store website, lookbook gallery, category navigation, product details page, size selector, wishlist feature",

            # Photography & Art
            "A majestic lion in the African savanna at sunset, detailed fur, golden hour lighting, realistic, 8k photography",
            "A futuristic cyberpunk city street at night, neon lights, rain reflections, highly detailed, cinematic lighting",
            "A serene Japanese garden with cherry blossoms, traditional architecture, soft natural lighting, spring season"
        ]
    
    def get_negative_prompt(self) -> str:
        """Return optimized negative prompt."""
        return """
        (multiple people:1.7), (extra heads:1.6), (multiple faces:1.7), (multiple bodies:1.7),
        (extra limbs:1.6), (clones:1.8), (duplicates:1.8), (twin:1.7), (group:1.6),
        (crowd:1.6), (second person:1.6), (additional figures:1.6),
        (deformed body:1.8), (bad anatomy:1.7), (duplicate:1.8), (cloned:1.8),
        (repeating:1.8), (merged body parts:1.7),
        (low quality:1.6), (worst quality:1.7), (bad quality:1.6), (jpeg artifacts:1.4),
        (compression artifacts:1.4), (blurry:1.5), (ugly:1.6), (deformed:1.7),
        (mutated:1.7), (distorted:1.6), (disfigured:1.7), (poorly drawn:1.5),
        (amateur:1.4), (messy:1.4), (sloppy:1.4), (unprofessional:1.5),
        (broken:1.6), (glitched:1.5), (corrupted:1.6),
        (deformed face:1.8), (deformed limbs:1.7), (bad anatomy:1.7),
        (anatomical errors:1.6), (extra fingers:1.6), (missing fingers:1.5),
        (fused fingers:1.6), (too many fingers:1.6), (mutated hands:1.7),
        (floating limbs:1.7), (disconnected limbs:1.7), (broken limbs:1.7),
        (dislocated joints:1.6), (wrong proportions:1.6),
        (duplicate faces:1.8), (cloned faces:1.8), (multiple faces:1.8),
        (bad composition:1.5), (unbalanced composition:1.5), (poor composition:1.5),
        (improper perspective:1.5), (wrong perspective:1.5), (bad perspective:1.5),
        (bad lighting:1.4), (harsh lighting:1.4), (uneven lighting:1.4),
        (watermark:1.7), (text:1.6), (signature:1.6), (logo:1.6),
        (aliasing:1.4), (pixelation:1.4), (noise:1.4), (grain:1.4),
        (inconsistent style:1.5), (mixed styles:1.5), (conflicting styles:1.5),
        (wrong colors:1.5), (bad colors:1.5), (mismatched colors:1.5), (clashing colors:1.5),
        (poor resolution:1.4), (low resolution:1.4), (blurry details:1.4), (pixelated details:1.4)
        """.replace('\n', ' ').replace('    ', '').strip()
    
    def enhancement_prompt(self) -> str:
        """Return optimized enhancement prompt."""
        return """
        (masterpiece:1.5), (best quality:1.6), (ultra high resolution:1.3),
        (highly detailed:1.4), (sharp focus:1.2), (crystal clear:1.2),
        (clean design:1.4), (professional layout:1.3), (pixel-perfect:1.2),
        (modern aesthetics:1.3), (precise alignment:1.2), (balanced composition:1.2),
        (high-fidelity mockup:1.3), (clear typography:1.2), (polished interface:1.3),
        (8k resolution:1.2), (ultra HD:1.2), (perfect rendering:1.2), (sharp edges:1.2),
        (professional UI design:1.4), (pristine quality:1.3), (flawless execution:1.3)
        """.strip()
    
    def generate_image(
        self,
        prompt: str,
        steps: int = 50,
        guidance_base: float = 22,
        width: int = 768,
        height: int = 768,
        seed: int = -1,
        use_adaptive_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        enhance_prompt: bool = True
    ):
        """Generate an image using Stable Diffusion."""
        try:
            # Use default negative prompt if not provided
            if negative_prompt is None:
                negative_prompt = self.get_negative_prompt()
            
            # Add enhancement prompt if requested
            full_prompt = prompt
            if enhance_prompt:
                enhancement_prompt = self.enhancement_prompt()
                full_prompt = f"{prompt}, {enhancement_prompt}"
            
            # Calculate guidance scale
            final_guidance = (
                self.adaptive_guidance(prompt, guidance_base) 
                if use_adaptive_guidance else guidance_base
            )
            
            # Set random seed if needed
            if seed < 0:
                seed = random.randint(0, 2**32 - 1)
                
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Generating image with seed: {seed}")
            
            # Generate image
            result = self.pipe(
                full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=final_guidance,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=1,
            )
            
            image = result.images[0]
            
            # Log generation success
            logger.info(f"Image generated successfully: {width}x{height}, steps={steps}, guidance={final_guidance:.2f}")
            
            return image, seed
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory. Try reducing image size or model complexity.")
            return None, seed
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None, seed

def gradio_interface():
    """Create the Gradio interface for the image generator."""
    # Initialize generator
    generator = StableDiffusionGenerator()
    
    # Get example prompts
    example_prompts = generator.get_example_prompts()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Text-to-Image Generator")
        gr.Markdown("Generate high-quality images from text descriptions using Stable Diffusion.")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            steps = gr.Slider(30, 150, value=50, step=1, label="Quality Steps")
                            guidance = gr.Slider(7, 30, value=22, step=0.5, label="Guidance Scale")
                            adaptive_guidance = gr.Checkbox(
                                label="Adaptive Guidance", 
                                value=True,
                                info="Automatically adjust guidance based on prompt complexity"
                            )
                        with gr.Column(scale=1):
                            width = gr.Slider(512, 1024, value=768, step=128, label="Width")
                            height = gr.Slider(512, 1024, value=768, step=128, label="Height")
                            seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    
                    enhance_prompt = gr.Checkbox(
                        label="Enhance Prompt", 
                        value=True,
                        info="Add quality boosting terms to your prompt"
                    )
                    
                with gr.Row():
                    generate_button = gr.Button("Generate Image", variant="primary")
                    clear_button = gr.Button("Clear")
                    random_prompt_button = gr.Button("Random Prompt")
                
                seed_display = gr.Number(label="Used Seed", visible=True)
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")
                image_info = gr.Markdown("Generation details will appear here")
                
                with gr.Accordion("Example Prompts", open=False):
                    examples = gr.Examples(
                        examples=[[prompt_text] for prompt_text in example_prompts],
                        inputs=[prompt],
                        label="Click an example to use it as your prompt"
                    )
        
        # Event handlers
        def result(prompt, steps, guidance, width, height, seed, adaptive_guidance, enhance_prompt):
            if not prompt.strip():
                return None, seed, "Please enter a prompt"
                
            image, used_seed = generator.generate_image(
                prompt=prompt,
                steps=steps,
                guidance_base=guidance,
                width=width,
                height=height,
                seed=seed,
                use_adaptive_guidance=adaptive_guidance,
                enhance_prompt=enhance_prompt
            )
            
            if image is None:
                return None, seed, "Error generating image. Check the logs for details."
            
            info_text = f"""
            **Generation Details**
            - **Dimensions**: {width}x{height}
            - **Steps**: {steps}
            - **Guidance**: {guidance}{"+" if adaptive_guidance else ""}
            - **Seed**: {used_seed}
            """
            
            return image, used_seed, info_text
        
        # Wire up the event handlers
        generate_button.click(
            result,
            inputs=[prompt, steps, guidance, width, height, seed, adaptive_guidance, enhance_prompt],
            outputs=[output_image, seed_display, image_info]
        )
        
        clear_button.click(
            lambda: [None, 50, 22, 768, 768, -1, True, True, None, ""],
            outputs=[prompt, steps, guidance, width, height, seed, adaptive_guidance, enhance_prompt, output_image, image_info]
        )
        
        random_prompt_button.click(
            lambda: random.choice(example_prompts),
            outputs=[prompt]
        )
        
        gr.Markdown("*Note: This interface uses an optimized Stable Diffusion pipeline with advanced features including adaptive guidance, quality enhancements, and performance optimizations.*")
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = gradio_interface()
    demo.launch(share=True, debug=True)