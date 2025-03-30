import os
import gc
import tempfile
import numpy as np
import torch
import imageio
from PIL import Image

import gradio as gr
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline

# Optional: Enable dynamic memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def free_memory():
    """Helper function to clean up GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_video_model():
    """Load the Stable Video Diffusion model."""
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )
    return pipe

def load_anime_model():
    """Load the anime-style image-to-image model."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "eimiss/EimisAnimeDiffusion_1.0v",
        torch_dtype=torch.float16
    )
    return pipe

def convert_to_anime(image, anime_pipe):
    """Convert the input image into anime style using img2img pipeline."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    anime_pipe = anime_pipe.to("cuda")
    result = anime_pipe(prompt="anime style", image=image, strength=0.7).images[0]
    anime_pipe.to("cpu")
    del anime_pipe
    free_memory()
    return result

def generate_gif(image, anime_first=True):
    """Convert image to anime (optional), then generate animated GIF from video frames."""
    if anime_first:
        anime_pipe = load_anime_model()
        image = convert_to_anime(image, anime_pipe)

    video_pipe = load_video_model()
    video_pipe = video_pipe.to("cuda")
    result = video_pipe(image, decode_chunk_size=8, num_inference_steps=25).frames[0]
    video_pipe.to("cpu")
    del video_pipe
    free_memory()

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_path = tmp.name
        imageio.mimsave(gif_path, result, duration=0.1)

    return gif_path

# Gradio UI
def build_interface():
    css = """
    body { background-color: #fdf6f6; font-family: 'Comic Sans MS', cursive; }
    .gradio-container { border-radius: 20px; box-shadow: 0 0 10px #fcb3c2; padding: 20px; }
    h1, h2, h3 { color: #ff6699; text-align: center; }
    """

    description = """
    # 🌟 Anime Emoji GIF Generator 🌟
    Upload an image and turn it into a cute animated emoji with anime-style magic!
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown(description)
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            anime_toggle = gr.Checkbox(label="Convert to Anime Style First", value=True)
        generate_btn = gr.Button("✨ Generate GIF ✨")
        output_gif = gr.Video(label="Generated GIF")

        status_text = gr.Textbox(label="Status", visible=False)

        def generate_with_status(img, anime_flag):
            status_text.visible = True
            status_text.value = "Processing... Please wait."
            gif_path = generate_gif(img, anime_flag)
            status_text.value = "Done!"
            return gif_path

        generate_btn.click(fn=generate_with_status, inputs=[image_input, anime_toggle], outputs=output_gif)

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
