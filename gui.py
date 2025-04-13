import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import os
import argparse
from tqdm import tqdm
# <<< NEW IMPORT for color conversion >>>
from skimage import color # Use scikit-image for robust color space conversion

# Import necessary components from your project
from baseline import Baseline
from aespanet_models import size_arrange
from utils import _normalizer

# --- Configuration ---
MODEL_COMMENT = "aepapa_run1"
DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 64
DEFAULT_UPSCALE_FACTOR = 1.0
# DEFAULT_INTERPOLATION removed, now handled in Step 2
DEFAULT_BLEND_ALPHA = 0.7 # Default for the new blending slider
GPU_BATCH_SIZE = 1 # **** Adjust based on your GPU VRAM **** (Start small, e.g., 2 or 4)

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Using GPU Batch Size: {GPU_BATCH_SIZE}")

# --- Model Loading ---
parser = argparse.ArgumentParser(description='AesPA-Net Gradio')
# Add arguments... (same as before)
parser.add_argument('--comment', default=MODEL_COMMENT)
parser.add_argument('--train_result_dir', type=str, default='./train_results', help='Base directory for trained models')
parser.add_argument('--imsize', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=GPU_BATCH_SIZE) # Link arg to batch size
parser.add_argument('--cencrop', action='store_true', default=False)
parser.add_argument('--cropsize', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--content_dir', type=str, default='./content')
parser.add_argument('--style_dir', type=str, default='./style')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=1)
parser.add_argument('--check_iter', type=int, default=1)

import sys
# Handle potential argument parsing issues in environments like notebooks
try:
    args = parser.parse_args(sys.argv[1:])
except SystemExit:
    # Use default args if run without command-line arguments (e.g., in notebook)
    print("No command line args found or error parsing, using default args.")
    args = parser.parse_args([])


args.result_st_dir = os.path.join(args.train_result_dir, args.comment, 'log')
print("Initializing Baseline class...")
baseline_model = Baseline(args)
print("Setting up model for inference...")
baseline_model.setup_for_inference(DEVICE)
print("Model setup complete.")


# --- Helper Functions ---
normalize = _normalizer(denormalize=False)
denormalize = _normalizer(denormalize=True)

def preprocess_image(img_pil, target_size=None):
    """Converts PIL Image to normalized tensor."""
    if target_size:
        img_pil.thumbnail(target_size, Image.Resampling.LANCZOS)
    if img_pil.mode != 'RGB': img_pil = img_pil.convert('RGB')
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return normalize(img_tensor)

def postprocess_image(tensor):
    """Converts normalized tensor back to PIL Image."""
    if tensor is None or tensor.numel() == 0: return Image.new('RGB', (100, 100), color='grey')
    tensor = denormalize(tensor.squeeze(0).cpu()) # Process one image at a time from batch
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

def run_stylization_batch(content_batch, style_tensor_single, gray_content_batch, gray_style_tensor_single):
    """
    Runs stylization on a batch of content tiles using a single style reference.
    (Same as before)
    """
    if not isinstance(content_batch, torch.Tensor):
        raise TypeError("content_batch must be a torch.Tensor")

    actual_batch_size = content_batch.size(0)
    if actual_batch_size == 0:
        return torch.empty(0) # Return empty tensor if batch is empty

    # Repeat style tensors to match the content batch size
    style_batch = style_tensor_single.repeat(actual_batch_size, 1, 1, 1)
    gray_style_batch = gray_style_tensor_single.repeat(actual_batch_size, 1, 1, 1)

    with torch.no_grad(): # Alpha calculation should not require gradients
        single_alpha = (((baseline_model.adaptive_gram_weight(style_tensor_single, 1, 8) +
                          baseline_model.adaptive_gram_weight(style_tensor_single, 2, 8) +
                          baseline_model.adaptive_gram_weight(style_tensor_single, 3, 8)) / 3).unsqueeze(1).to(DEVICE) + \
                         ((baseline_model.adaptive_gram_weight(gray_style_tensor_single, 1, 8) +
                           baseline_model.adaptive_gram_weight(gray_style_tensor_single, 2, 8) +
                           baseline_model.adaptive_gram_weight(gray_style_tensor_single, 3, 8)) / 3).unsqueeze(1).to(DEVICE)
                        ) / 2
        adaptive_alpha_batch = single_alpha.repeat(actual_batch_size, 1) # Repeat along batch dimension

    stylized_batch, _, _, _, _ = baseline_model.network(
        content_batch,
        style_batch,
        adaptive_alpha_batch,
        gray_content_batch,
        style_batch
    )

    return stylized_batch


def stylize_image_step1( # Renamed from stylize_image_wrapper
    content_img_pil,
    style_img_pil,
    tile_size,
    overlap,
    content_upscale_factor,
    progress=gr.Progress(track_tqdm=True)
    ):
    """
    Step 1: Performs the full tiling, stylization, and stitching.
    Returns the *fully stylized* image (alpha=1.0 equivalent).
    """
    # --- Input Validation and Setup ---
    if content_img_pil is None: raise gr.Error("Content Image is required.")
    if style_img_pil is None: raise gr.Error("Style Image is required.")
    if tile_size <= overlap: raise gr.Error("Tile Size must be greater than Tile Overlap.")
    if content_upscale_factor < 1.0: raise gr.Error("Content Upscale Factor must be >= 1.0.")
    # Add more specific checks if needed

    # Ensure PIL images
    if not isinstance(content_img_pil, Image.Image):
         raise gr.Error(f"Invalid Content Image type: {type(content_img_pil)}")
    if not isinstance(style_img_pil, Image.Image):
         raise gr.Error(f"Invalid Style Image type: {type(style_img_pil)}")

    # --- Keep original content image for later ---
    # Ensure it's RGB
    if content_img_pil.mode != 'RGB':
        content_img_pil = content_img_pil.convert('RGB')

    print(f"Step 1 Start: Content={content_img_pil.size}, Style={style_img_pil.size}, Tile={tile_size}, Overlap={overlap}, Upscale={content_upscale_factor}")

    # --- Preprocess Style Image (Once) ---
    style_tensor_single = preprocess_image(style_img_pil).to(DEVICE)
    gray_style_tensor_single = baseline_model.invert_gray(style_tensor_single).to(DEVICE) # Precompute gray style
    print(f"Style tensor shape: {style_tensor_single.shape}, Device: {style_tensor_single.device}")

    C_W, C_H = content_img_pil.size
    # Keep the original numpy array for reference if needed, but PIL is primary now
    # original_content_np = np.array(content_img_pil).astype(np.float32) / 255.0

    # --- Tiling Logic (Prepare data for batching) ---
    content_tiles_data = [] # Will store dicts with tensors and metadata
    step = tile_size - overlap
    print("Starting tiling...")

    # --- Tiling Loop ---
    for y in range(0, C_H, step):
        if y + tile_size > C_H and C_H > tile_size: y = C_H - tile_size
        y_start_coord = min(y, max(0, C_H - tile_size)) if C_H > overlap else 0

        for x in range(0, C_W, step):
            if x + tile_size > C_W and C_W > tile_size: x = C_W - tile_size
            x_start_coord = min(x, max(0, C_W - tile_size)) if C_W > overlap else 0

            crop_x_start, crop_y_start = x_start_coord, y_start_coord
            crop_x_end = min(x_start_coord + tile_size, C_W)
            crop_y_end = min(y_start_coord + tile_size, C_H)

            content_tile_pil = content_img_pil.crop((crop_x_start, crop_y_start, crop_x_end, crop_y_end))
            original_tile_w, original_tile_h = content_tile_pil.size

            current_w, current_h = content_tile_pil.size
            pad_right = max(0, tile_size - current_w)
            pad_bottom = max(0, tile_size - current_h)
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
            gray_content_tile_tensor = baseline_model.invert_gray(content_tile_tensor).to(DEVICE)

            content_tiles_data.append({
                "content_tensor": content_tile_tensor,
                "gray_tensor": gray_content_tile_tensor,
                "coords": (crop_x_start, crop_y_start),
                "original_size": (original_tile_w, original_tile_h),
                "processed_size": (upscaled_w, upscaled_h)
            })

            if x_start_coord + tile_size >= C_W: break
        if y_start_coord + tile_size >= C_H: break

    # --- Batch Stylization ---
    stylized_tiles_data = [] # Stores final PIL tiles + coords
    num_tiles = len(content_tiles_data)
    print(f"\nStylizing {num_tiles} tiles in batches of {GPU_BATCH_SIZE}...")

    num_batches = math.ceil(num_tiles / GPU_BATCH_SIZE)
    processed_tiles_count = 0

    for i in progress.tqdm(range(num_batches), desc="Stylizing Batches"):
        batch_start_idx = i * GPU_BATCH_SIZE
        batch_end_idx = min((i + 1) * GPU_BATCH_SIZE, num_tiles)
        current_batch_data = content_tiles_data[batch_start_idx:batch_end_idx]

        if not current_batch_data: continue

        content_batch_list = [d["content_tensor"] for d in current_batch_data]
        gray_content_batch_list = [d["gray_tensor"] for d in current_batch_data]

        content_batch_gpu = torch.cat(content_batch_list, dim=0)
        gray_content_batch_gpu = torch.cat(gray_content_batch_list, dim=0)

        try:
            with torch.no_grad():
                 stylized_batch_gpu = run_stylization_batch(
                     content_batch_gpu,
                     style_tensor_single,
                     gray_content_batch_gpu,
                     gray_style_tensor_single
                 )
        except RuntimeError as e:
             if "out of memory" in str(e):
                 print("\nCUDA out of memory! Try reducing Tile Size, GPU Batch Size, or Content Upscale Factor.")
                 raise gr.Error("GPU out of memory. Reduce settings and retry.") from e
             else:
                 raise e

        for j, stylized_tile_tensor in enumerate(stylized_batch_gpu):
            tile_data = current_batch_data[j]
            stylized_tile_tensor = stylized_tile_tensor.unsqueeze(0)

            if content_upscale_factor > 1.0:
                 target_w, target_h = tile_size, tile_size
                 stylized_tile_tensor = F.interpolate(stylized_tile_tensor, size=(target_h, target_w), mode='area')

            stylized_tile_pil = postprocess_image(stylized_tile_tensor)

            original_w, original_h = tile_data["original_size"]
            stylized_tile_pil = stylized_tile_pil.crop((0, 0, original_w, original_h))

            stylized_tiles_data.append({
                "pil": stylized_tile_pil,
                "coords": tile_data["coords"]
            })
            processed_tiles_count += 1

        del content_batch_gpu, gray_content_batch_gpu, stylized_batch_gpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    # --- Stitching with Weighted Blending ---
    print("\nStitching tiles with weighted blending...")
    output_accumulator = np.zeros((C_H, C_W, 3), dtype=np.float32)
    weight_accumulator = np.zeros((C_H, C_W, 1), dtype=np.float32)
    stitching_kernel = make_tent_kernel(tile_size)

    for tile_data in progress.tqdm(stylized_tiles_data, desc="Stitching Tiles"):
        stylized_tile_pil = tile_data["pil"]
        x_start, y_start = tile_data["coords"]
        tile_w, tile_h = stylized_tile_pil.size
        paste_x_end = x_start + tile_w
        paste_y_end = y_start + tile_h

        kernel_h, kernel_w, _ = stitching_kernel.shape
        eff_tile_h, eff_tile_w = min(tile_h, kernel_h), min(tile_w, kernel_w)

        kernel_crop = stitching_kernel[:eff_tile_h, :eff_tile_w, :]
        tile_np = np.array(stylized_tile_pil).astype(np.float32) / 255.0

        output_accumulator[y_start:paste_y_end, x_start:paste_x_end] += tile_np[:eff_tile_h, :eff_tile_w, :] * kernel_crop
        weight_accumulator[y_start:paste_y_end, x_start:paste_x_end] += kernel_crop

    # Avoid division by zero
    weight_accumulator[weight_accumulator == 0] = 1e-7
    stylized_result_np = output_accumulator / weight_accumulator

    # --- NO Final Interpolation here ---
    # The result is the fully stylized image

    final_result_np = np.clip(stylized_result_np * 255, 0, 255).astype(np.uint8)
    fully_stylized_img_pil = Image.fromarray(final_result_np)

    # --- MODIFIED RETURN STATEMENT ---
    success_message = f"Step 1 finished: Stylized image generated ({fully_stylized_img_pil.width}x{fully_stylized_img_pil.height}). Ready for Step 2 blending."
    print(success_message)

    # Return values matching the listener's 'outputs' list
    return (
        content_img_pil,          # 1. For original_content_state
        fully_stylized_img_pil,   # 2. For stylized_image_state
        content_img_pil,          # 3. For step2_content_ref (update reference image)
        fully_stylized_img_pil,   # 4. For step1_stylized_ref (update reference image)
        success_message           # 5. For step1_output_status update
    )


# --- Step 2: Blending Logic ---

def blend_images(content_pil, stylized_pil, blend_alpha, blend_mode):
    """
    Step 2: Blends the content and stylized images based on alpha and mode.
    This function should be fast.
    """
    if content_pil is None or stylized_pil is None:
        # Return a placeholder or raise error if called before Step 1 is complete
        return Image.new('RGB', (200, 200), color='gray')

    # Ensure images are RGB and NumPy arrays
    if content_pil.mode != 'RGB': content_pil = content_pil.convert('RGB')
    if stylized_pil.mode != 'RGB': stylized_pil = stylized_pil.convert('RGB')

    # Ensure images are the same size (they should be after Step 1)
    if content_pil.size != stylized_pil.size:
        print(f"Warning: Content size {content_pil.size} != Stylized size {stylized_pil.size}. Resizing stylized image.")
        stylized_pil = stylized_pil.resize(content_pil.size, Image.Resampling.LANCZOS)
        # raise gr.Error("Content and Stylized images have different dimensions!")

    content_np = np.array(content_pil).astype(np.float32) / 255.0
    stylized_np = np.array(stylized_pil).astype(np.float32) / 255.0

    output_np = None

    if blend_mode == "Linear":
        output_np = (1.0 - blend_alpha) * content_np + blend_alpha * stylized_np
        print(f"Blending: Linear (alpha={blend_alpha})")

    elif blend_mode == "Preserve Content Color (LAB)":
        print(f"Blending: Preserve Content Color (LAB) (alpha={blend_alpha}) - Converting...")
        try:
            # Convert RGB [0,1] to LAB
            content_lab = color.rgb2lab(content_np)
            stylized_lab = color.rgb2lab(stylized_np)

            # Combine L from stylized, A and B from content
            stylized_structure_content_color_lab = np.stack(
                (stylized_lab[:, :, 0],  # L channel from stylized
                 content_lab[:, :, 1],   # A channel from content
                 content_lab[:, :, 2]),  # B channel from content
                axis=-1
            )

            # Convert back to RGB [0,1]
            stylized_structure_content_color_rgb = color.lab2rgb(stylized_structure_content_color_lab)
            print("Blending: LAB conversion complete.")

            # Blend between original content and the new color-preserved image
            output_np = (1.0 - blend_alpha) * content_np + blend_alpha * stylized_structure_content_color_rgb

        except Exception as e:
            print(f"Error during LAB color preservation: {e}")
            # Fallback to linear interpolation on error
            gr.Warning("Error during LAB color preservation. Falling back to Linear blend.")
            output_np = (1.0 - blend_alpha) * content_np + blend_alpha * stylized_np


    # Add more blend modes here if needed
    # elif blend_mode == "Preserve Style Color (LAB)":
    #    # Similar logic, but take L from content, A,B from style
    #    pass

    else:
        print(f"Unknown blend mode: {blend_mode}. Using Linear.")
        output_np = (1.0 - blend_alpha) * content_np + blend_alpha * stylized_np

    # Clip and convert back to PIL
    final_result_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    final_output_img_pil = Image.fromarray(final_result_np)

    return final_output_img_pil

# --- Gradio Interface Definition ---
css = """
#col-container { margin: 0 auto; max-width: 1200px; }
.step-group { border: 1px solid #E0E0E0; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
.step-header { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
"""
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # AesPA-Net: Two-Step Stylization Workflow
        **GPU Batch Size:** {GPU_BATCH_SIZE} | **Model:** {MODEL_COMMENT} | **Device:** {DEVICE}
        """
    )

    # --- State Variables ---
    # To store intermediate results between steps
    # Use PIL type for images stored in state
    original_content_state = gr.State(value=None)
    stylized_image_state = gr.State(value=None)

    with gr.Row(): # Main layout row

        # --- Column 1: Inputs & Step 1 ---
        with gr.Column(scale=1):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("### Step 1: Generate Stylized Image", elem_classes="step-header")
                content_img = gr.Image(label="Content Image", type="pil", height=300)
                style_img = gr.Image(label="Style Image", type="pil", height=300)

                with gr.Accordion("Tiling & Processing Options", open=False):
                     gr.Markdown("Adjust how the image is processed during stylization. Affects quality and VRAM usage.")
                     tile_size_slider = gr.Slider(label="Tile Size", minimum=32, maximum=1024, step=32, value=DEFAULT_TILE_SIZE, info="Size of square tiles processed (pixels).")
                     overlap_slider = gr.Slider(label="Tile Overlap", minimum=32, maximum=512, step=32, value=DEFAULT_OVERLAP, info="Overlap between tiles (pixels). Must be < Tile Size.")
                     content_upscale_slider = gr.Slider(label="Content Tile Upscale Factor", minimum=1.0, maximum=4.0, step=0.1, value=DEFAULT_UPSCALE_FACTOR, info="Upscale content tiles before stylization (1.0 = none).")

                stylize_button = gr.Button("Run Stylization (Step 1)", variant="primary")
                step1_output_status = gr.Markdown("", elem_id="step1_status") # For status updates

        # --- Column 2: Step 2 & Final Output ---
        with gr.Column(scale=2):
             with gr.Group(elem_classes="step-group"):
                gr.Markdown("### Step 2: Blend & Refine", elem_classes="step-header")
                with gr.Row():
                    # Display inputs for Step 2 (non-interactive)
                    step2_content_ref = gr.Image(label="Original Content (Reference)", type="pil", height=300, interactive=False)
                    step1_stylized_ref = gr.Image(label="Stylized Result (Step 1 Output)", type="pil", height=300, interactive=False)

                gr.Markdown("Adjust blending between Original and Stylized images:")
                blend_mode_dropdown = gr.Dropdown(
                    label="Blending Mode",
                    choices=["Linear", "Preserve Content Color (LAB)"],
                    value="Linear",
                    info="How to merge the images."
                )
                blend_alpha_slider = gr.Slider(
                    label="Blend Strength (Alpha)",
                    minimum=0.0, maximum=1.0, step=0.01, value=DEFAULT_BLEND_ALPHA,
                    info="0.0 = Original Content, 1.0 = Full Blend Effect"
                )

                final_output_img = gr.Image(label="Final Blended Result", type="pil", height=500)


    # --- Event Listeners ---

    # Step 1 Button Click
    stylize_button.click(
        fn=stylize_image_step1,
        inputs=[
            content_img, style_img, tile_size_slider, overlap_slider,
            content_upscale_slider,
        ],
        outputs=[
            # Store results in state
            original_content_state,
            stylized_image_state,
            # Update reference images in Step 2 panel
            step2_content_ref,
            step1_stylized_ref,
            # Clear status
            step1_output_status
        ]
    ).then( # Chain a second action to run the initial blend after stylization completes
        fn=blend_images,
        inputs=[
            original_content_state, # Read from state
            stylized_image_state,   # Read from state
            blend_alpha_slider,     # Read current slider value
            blend_mode_dropdown     # Read current dropdown value
        ],
        outputs=[final_output_img], # Update the final output image
        show_progress="hidden" # Don't show progress for the quick blend
    )

    # Step 2 Controls Change (Slider or Dropdown)
    # Use list comprehensions for inputs to trigger on change of any control
    step2_controls = [blend_alpha_slider, blend_mode_dropdown]
    for control in step2_controls:
        control.change( # Use .change() for interactive updates
            fn=blend_images,
            inputs=[
                original_content_state, # Read from state
                stylized_image_state,   # Read from state
                blend_alpha_slider,
                blend_mode_dropdown
            ],
            outputs=[final_output_img],
            show_progress="hidden" # Keep it fast, no progress indicator needed
        )

# --- Launch the App ---
if __name__ == "__main__":
    # Install scikit-image if you haven't: pip install scikit-image
    try:
        import skimage
        print("scikit-image found.")
    except ImportError:
        print("Error: scikit-image is required for color preservation.")
        print("Please install it: pip install scikit-image")
        # sys.exit(1) # Optional: exit if dependency is missing

    demo.queue().launch(debug=True) # Queue is generally recommended for ML models