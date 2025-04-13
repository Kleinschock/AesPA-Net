import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import os
import argparse # To reuse args defaults if needed
# Remove the direct tqdm import if not used elsewhere, Gradio handles it
# from tqdm import tqdm

# Import necessary components from your project
from baseline import Baseline
from aespanet_models import size_arrange
from utils import _normalizer

# --- Configuration ---
MODEL_COMMENT = "aepapa_run1" # Default comment for finding weights
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 64
DEFAULT_UPSCALE_FACTOR = 1.0 # 1.0 means no upscaling
DEFAULT_INTERPOLATION = 1.0 # 1.0 means full stylized image

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
# (Assume model loading code is correct and working)
parser = argparse.ArgumentParser(description='AesPA-Net Gradio')
parser.add_argument('--comment', default=MODEL_COMMENT)
parser.add_argument('--train_result_dir', type=str, default='./train_results', help='Base directory for trained models')
parser.add_argument('--imsize', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--cencrop', action='store_true', default=False)
parser.add_argument('--cropsize', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--content_dir', type=str, default='./content')
parser.add_argument('--style_dir', type=str, default='./style')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=1)
parser.add_argument('--check_iter', type=int, default=1)
# Use a specific list if running interactively/debugging to avoid conflicts
# args = parser.parse_args([])
# Or use this if running as script
import sys
args = parser.parse_args(sys.argv[1:])

args.result_st_dir = os.path.join(args.train_result_dir, args.comment, 'log')
print("Initializing Baseline class...")
baseline_model = Baseline(args)
print("Setting up model for inference...")
baseline_model.setup_for_inference(DEVICE)
print("Model setup complete.")


# --- Helper Functions ---
# (Keep preprocess_image, postprocess_image, make_tent_kernel as before)
normalize = _normalizer(denormalize=False)
denormalize = _normalizer(denormalize=True)

def preprocess_image(img_pil, target_size=None):
    """Converts PIL Image to normalized tensor."""
    if target_size:
        img_pil.thumbnail(target_size, Image.Resampling.LANCZOS)

    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return normalize(img_tensor)

def postprocess_image(tensor):
    """Converts normalized tensor back to PIL Image."""
    if tensor is None or tensor.numel() == 0:
        return Image.new('RGB', (100, 100), color='grey') # Return placeholder
    tensor = denormalize(tensor.squeeze(0).cpu())
    img_np = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    return img_pil

def make_tent_kernel(size):
    """Creates a 2D pyramid/tent weighting kernel."""
    axis = np.linspace(-1, 1, size)
    x_kernel = 1 - np.abs(axis)
    y_kernel = 1 - np.abs(axis)
    kernel_2d = np.outer(y_kernel, x_kernel)
    return kernel_2d[:, :, np.newaxis].astype(np.float32)

# --- Core Logic ---

def stylize_image_wrapper(
    content_img_pil,
    style_img_pil,
    tile_size,
    overlap,
    content_upscale_factor,
    interpolation_alpha,
    progress=gr.Progress(track_tqdm=True) # Keep track_tqdm=True here
    ):
    """
    Main function called by Gradio. Handles tiling, optional upscaling,
    stylization, weighted stitching, and interpolation.
    """
    if content_img_pil is None or style_img_pil is None:
        raise gr.Error("Please provide both Content and Style images.")
    if tile_size <= overlap:
         raise gr.Error("Tile Size must be greater than Overlap.")
    if overlap < 16:
        print("Warning: Very small overlap detected. Seams might be more visible.")
    if content_upscale_factor < 1.0:
        raise gr.Error("Content Upscale Factor must be 1.0 or greater.")

    print(f"Input content size: {content_img_pil.size}")
    print(f"Input style size: {style_img_pil.size}")
    print(f"Tile size: {tile_size}, Overlap: {overlap}, Upscale: {content_upscale_factor}, Interpolation: {interpolation_alpha}")

    style_tensor = preprocess_image(style_img_pil).to(DEVICE)
    print(f"Style tensor shape: {style_tensor.shape}, Device: {style_tensor.device}")

    C_W, C_H = content_img_pil.size
    original_content_np = np.array(content_img_pil).astype(np.float32) / 255.0

    # --- Tiling Logic ---
    content_tiles_data = []
    step = tile_size - overlap

    print("Starting tiling...")
    # No progress.update needed here, tqdm will start with the loop

    for y in range(0, C_H, step):
        if y + tile_size > C_H and C_H > tile_size: y = C_H - tile_size
        y_start_coord = min(y, max(0, C_H - tile_size)) if C_H > overlap else 0
        for x in range(0, C_W, step):
            if x + tile_size > C_W and C_W > tile_size: x = C_W - tile_size
            x_start_coord = min(x, max(0, C_W - tile_size)) if C_W > overlap else 0

            crop_x_start, crop_y_start = x_start_coord, y_start_coord
            crop_x_end, crop_y_end = min(x_start_coord + tile_size, C_W), min(y_start_coord + tile_size, C_H)
            content_tile_pil = content_img_pil.crop((crop_x_start, crop_y_start, crop_x_end, crop_y_end))
            original_tile_w, original_tile_h = content_tile_pil.size

            current_w, current_h = content_tile_pil.size
            pad_right, pad_bottom = max(0, tile_size - current_w), max(0, tile_size - current_h)
            padded = pad_right > 0 or pad_bottom > 0
            if padded:
                content_tile_np = np.array(content_tile_pil)
                content_tile_np = np.pad(content_tile_np, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
                content_tile_pil = Image.fromarray(content_tile_np)

            upscaled_w, upscaled_h = content_tile_pil.size
            if content_upscale_factor > 1.0:
                upscaled_w = int(content_tile_pil.width * content_upscale_factor)
                upscaled_h = int(content_tile_pil.height * content_upscale_factor)
                content_tile_pil = content_tile_pil.resize((upscaled_w, upscaled_h), Image.Resampling.LANCZOS)

            content_tile_tensor = preprocess_image(content_tile_pil).to(DEVICE)

            content_tiles_data.append({
                "tensor": content_tile_tensor, "coords": (crop_x_start, crop_y_start),
                "original_size": (original_tile_w, original_tile_h),
                "processed_size": (upscaled_w, upscaled_h)
            })

            if x_start_coord + tile_size >= C_W: break
        if y_start_coord + tile_size >= C_H: break

    # --- Stylization ---
    stylized_tiles_data = []
    print(f"\nStylizing {len(content_tiles_data)} tiles...")
    # Use progress.tqdm here to automatically handle progress bar updates
    for i, tile_data in enumerate(progress.tqdm(content_tiles_data, desc="Stylizing Tiles")):
        content_tile_tensor = tile_data["tensor"]
        stylized_tile_tensor = baseline_model.run_single_stylization(content_tile_tensor, style_tensor, DEVICE)

        processed_w, processed_h = tile_data["processed_size"]
        target_w, target_h = processed_w, processed_h

        if content_upscale_factor > 1.0:
             target_w, target_h = tile_size, tile_size
             if stylized_tile_tensor.dim() == 3: stylized_tile_tensor = stylized_tile_tensor.unsqueeze(0)
             stylized_tile_tensor = F.interpolate(stylized_tile_tensor, size=(target_h, target_w), mode='area')

        stylized_tile_pil = postprocess_image(stylized_tile_tensor)
        original_w, original_h = tile_data["original_size"]
        stylized_tile_pil = stylized_tile_pil.crop((0, 0, original_w, original_h))

        stylized_tiles_data.append({"pil": stylized_tile_pil, "coords": tile_data["coords"]})
        # **** REMOVE progress.update CALL FROM HERE ****
        # progress.update(...) # <-- DELETE THIS LINE

    # --- Stitching with Weighted Blending ---
    print("\nStitching tiles with weighted blending...")
    output_accumulator = np.zeros((C_H, C_W, 3), dtype=np.float32)
    weight_accumulator = np.zeros((C_H, C_W, 1), dtype=np.float32)
    stitching_kernel = make_tent_kernel(tile_size)

    # Use progress.tqdm here as well for the stitching part
    for i, tile_data in enumerate(progress.tqdm(stylized_tiles_data, desc="Stitching Tiles")):
        stylized_tile_pil = tile_data["pil"]
        x_start, y_start = tile_data["coords"]
        tile_w, tile_h = stylized_tile_pil.size
        paste_x_end, paste_y_end = x_start + tile_w, y_start + tile_h
        kernel_crop = stitching_kernel[:tile_h, :tile_w, :]
        tile_np = np.array(stylized_tile_pil).astype(np.float32) / 255.0
        output_accumulator[y_start:paste_y_end, x_start:paste_x_end] += tile_np * kernel_crop
        weight_accumulator[y_start:paste_y_end, x_start:paste_x_end] += kernel_crop
        # **** REMOVE progress.update CALL FROM HERE ****

    stylized_result_np = output_accumulator / (weight_accumulator + 1e-7)

    # --- Final Interpolation ---
    print("Applying interpolation...")
    final_result_np = (1.0 - interpolation_alpha) * original_content_np + interpolation_alpha * stylized_result_np
    final_result_np = np.clip(final_result_np * 255, 0, 255).astype(np.uint8)
    final_output_img = Image.fromarray(final_result_np)

    print("Stylization and stitching complete.")
    return final_output_img

# --- Gradio Interface Definition ---
# (UI Definition remains the same)
css = """
#col-container { margin: 0 auto; max-width: 900px; }
"""
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AesPA-Net: Aesthetic Pattern-Aware Style Transfer (Tiled)
        Upload Content and Style images. Adjust tiling, upscaling, and interpolation as needed.
        Larger images require smaller tiles and more overlap for better results.
        Model: **{}** | Device: **{}**
        """.format(MODEL_COMMENT, DEVICE)
    )

    with gr.Row():
         content_img = gr.Image(label="Content Image", type="pil", height=400)
         style_img = gr.Image(label="Style Image", type="pil", height=400)

    with gr.Accordion("Tiling & Processing Options", open=False):
        with gr.Row():
            tile_size_slider = gr.Slider(label="Tile Size", minimum=256, maximum=1024, step=64, value=DEFAULT_TILE_SIZE, info="Size of square tiles processed by the model. Smaller uses less VRAM but may lose global context.")
            overlap_slider = gr.Slider(label="Tile Overlap", minimum=32, maximum=512, step=32, value=DEFAULT_OVERLAP, info="Overlap between tiles (pixels). Higher values improve blending but increase computation. Must be < Tile Size.")
        with gr.Row():
            content_upscale_slider = gr.Slider(label="Content Tile Upscale Factor", minimum=1.0, maximum=2.0, step=0.1, value=DEFAULT_UPSCALE_FACTOR, info="Upscale content tiles before stylization (1.0 = none). Increases detail potential but uses more VRAM.")
            interpolation_slider = gr.Slider(label="Stylization Strength (Alpha)", minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_INTERPOLATION, info="Blend between original content (0.0) and fully stylized (1.0).")

    stylize_button = gr.Button("Stylize Image", variant="primary")

    output_img = gr.Image(label="Stylized Result", type="pil", height=500)

    stylize_button.click(
        fn=stylize_image_wrapper,
        inputs=[
            content_img,
            style_img,
            tile_size_slider,
            overlap_slider,
            content_upscale_slider,
            interpolation_slider,
        ],
        outputs=output_img,
    )


# --- Launch the App ---
if __name__ == "__main__":
    # Renamed the input file to app.py as is convention
    demo.queue().launch(debug=True)