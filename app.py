# app.py (Conceptual Enhancements)
import gradio as gr
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageFilter # For basic filters if needed
import numpy as np
import os
import argparse
import time
from skimage.color import rgb2lab, lab2rgb # For color preservation
from skimage.util import img_as_float, img_as_ubyte # For type conversions

# --- Imports from project files ---
try:
    from baseline import Baseline # Not strictly needed for inference, but keep for context
    from aespanet_models import Baseline_net, VGGEncoder, VGGDecoder, AdaptiveMultiAttn_Transformer_v2, size_arrange, calc_mean_std, mean_variance_norm
    from utils import denorm_2, gram_matrix, feature_wct_simple, GuidedFilter # Import GuidedFilter
    from smoothing_utils import l0_gradient_minimization_2d # Import L0 smoothing
    # Add any other necessary imports
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Make sure app.py is in the same directory as baseline.py, aespanet_models.py, utils.py, etc.")
    exit()
# --- End Imports ---


# --- Configuration ---
# Create dummy args for Baseline initialization
parser = argparse.ArgumentParser()
parser.add_argument('--comment', default='aespav_gui_run') # Give a default comment
parser.add_argument('--imsize', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1) # GUI processes one image at a time
parser.add_argument('--cencrop', action='store_true', default=False)
parser.add_argument('--cropsize', type=int, default=256) # Needed but might not be used directly in GUI inference
parser.add_argument('--num_workers', type=int, default=0) # Not relevant for GUI inference
parser.add_argument('--content_dir', type=str, default='') # Not needed for GUI
parser.add_argument('--style_dir', type=str, default='') # Not needed for GUI
parser.add_argument('--lr', type=float, default=1e-4) # Not needed for inference
parser.add_argument('--train_result_dir', default='./train_results') # For finding model weights
parser.add_argument('--test_result_dir', default='./test_results')
parser.add_argument('--max_iter', type=int, default=100000) # Not needed
parser.add_argument('--check_iter', type=int, default=1000) # Not needed
# Add any other args required by Baseline.__init__ with default values
args = parser.parse_args([])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
MODEL_LOADED = False
aespa_model = None
guided_filter = None # Instantiate GuidedFilter once

# Define paths relative to app.py
TRAINED_MODEL_DIR = './trained_models' # Directory where you placed the downloaded .pth files
DEC_WEIGHTS_PATH = os.path.join(TRAINED_MODEL_DIR, 'dec_model.pth')
TRANSFORMER_WEIGHTS_PATH = os.path.join(TRAINED_MODEL_DIR, 'transformer_model.pth')
VGG_WEIGHTS_PATH = './baseline_checkpoints/vgg_normalised_conv5_1.pth'

def load_model():
    global aespa_model, MODEL_LOADED
    if MODEL_LOADED:
        print("Model already loaded.")
        return aespa_model

    print("Loading AesPA-Net model...")
    start_time = time.time()
    try:
        # Ensure VGG weights exist
        if not os.path.exists(VGG_WEIGHTS_PATH):
             raise FileNotFoundError(f"VGG weights not found at {VGG_WEIGHTS_PATH}. Please download and place it correctly.")

        # Check if trained model weights exist
        if not os.path.exists(DEC_WEIGHTS_PATH):
            raise FileNotFoundError(f"Decoder weights not found at {DEC_WEIGHTS_PATH}. Please download and place it in {TRAINED_MODEL_DIR}/")
        if not os.path.exists(TRANSFORMER_WEIGHTS_PATH):
            raise FileNotFoundError(f"Transformer weights not found at {TRANSFORMER_WEIGHTS_PATH}. Please download and place it in {TRAINED_MODEL_DIR}/")

        # Load VGG state_dict first (as required by the fixed VGGEncoder)
        # Note: Using weights_only=True is safer if the file is trusted, but baseline.py didn't use it.
        # Let's stick to the original way for consistency unless errors occur.
        try:
             pretrained_vgg_state_dict = torch.load(VGG_WEIGHTS_PATH, map_location=torch.device('cpu'))
        except Exception as e:
             print(f"Error loading VGG state_dict from {VGG_WEIGHTS_PATH}: {e}")
             raise

        # Initialize Baseline_net directly (instead of Baseline class for simplicity)
        # The Baseline class is more for training setup. We just need the network.
        network = Baseline_net(pretrained_vgg=pretrained_vgg_state_dict) # Pass the state_dict

        # Load decoder weights
        dec_state_dict = torch.load(DEC_WEIGHTS_PATH, map_location=torch.device('cpu'))['state_dict']
        network.decoder.load_state_dict(dec_state_dict)
        print(f"Decoder weights loaded from {DEC_WEIGHTS_PATH}")

        # Load transformer weights
        transformer_state_dict = torch.load(TRANSFORMER_WEIGHTS_PATH, map_location=torch.device('cpu'))['state_dict']
        network.transformer.load_state_dict(transformer_state_dict)
        print(f"Transformer weights loaded from {TRANSFORMER_WEIGHTS_PATH}")

        # Set to evaluation mode and move to device
        network.eval()
        network.to(DEVICE)

        aespa_model = network # Store the network itself
        MODEL_LOADED = True
        print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        return aespa_model

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        # Optionally raise gr.Error("Model weights not found! Check paths.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        # Optionally raise gr.Error(f"Model loading failed: {e}")
        return None

# --- Image Processing ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Updated preprocessing to handle resolution
def preprocess_image(img_pil, target_size=512):
    """ Converts PIL Image to normalized tensor, resizes, and applies size_arrange. """
    # Resize first if needed (using PIL for antialiasing)
    if img_pil.size[0] != target_size or img_pil.size[1] != target_size:
         img_pil = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS) # Use LANCZOS for quality

    # Basic transforms
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    tensor = img_transforms(img_pil).unsqueeze(0) # Add batch dimension

    # Apply size_arrange AFTER initial resize and ToTensor
    tensor = size_arrange(tensor)
    return tensor.to(DEVICE)

def postprocess_image(tensor):
    """ Converts tensor back to PIL Image after denormalization. """
    denormalized_tensor = denorm_2(tensor.detach().cpu())
    pil_image = transforms.ToPILImage()(denormalized_tensor.squeeze(0))
    return pil_image

# --- Post-processing Functions ---

def apply_guided_filter(stylized_tensor, content_tensor, radius, eps):
    """ Applies Guided Filter """
    if guided_filter is None:
        print("Warning: Guided filter not initialized.")
        return stylized_tensor
    # Ensure tensors are on the correct device and potentially denormalized?
    # Guided filter paper often uses 0-1 range images. denorm_2 gives 0-1.
    content_guide = denorm_2(content_tensor).detach() # Use denormalized content as guide
    stylized_input = denorm_2(stylized_tensor).detach()

    # GuidedFilter expects CxHxW, ensure batch dim is handled if present
    if stylized_input.dim() == 4:
         stylized_input = stylized_input.squeeze(0)
         content_guide = content_guide.squeeze(0)

    # The GuidedFilter class seems to handle batches internally, let's try with batch dim
    stylized_input_batched = denorm_2(stylized_tensor).detach()
    content_guide_batched = denorm_2(content_tensor).detach()

    # Ensure guide is single channel (e.g., grayscale) or filter uses multi-channel guide?
    # Let's assume the filter works channel-wise if guide has same channels as input.
    # If guide must be grayscale:
    # content_guide_gray = torchvision.transforms.functional.rgb_to_grayscale(content_guide_batched)

    # Apply filter
    filtered_tensor = guided_filter(stylized_input_batched, content_guide_batched, radius, eps)

    # Result is in 0-1 range, normalize again for consistency if needed later?
    # For now, return the 0-1 tensor.
    return filtered_tensor # Denormalized output


def apply_l0_smoothing(img_pil, lambda_val):
    """ Applies L0 Gradient Minimization """
    img_np = img_as_float(img_pil) # Convert PIL to numpy float [0, 1]
    smoothed_np = l0_gradient_minimization_2d(img_np, lmd=lambda_val)
    smoothed_np = np.clip(smoothed_np, 0, 1)
    smoothed_pil = Image.fromarray(img_as_ubyte(smoothed_np)) # Convert back to PIL
    return smoothed_pil

def apply_color_preservation(stylized_pil, content_pil, strength):
    """ Blends color from content image to stylized image in Lab space """
    if strength <= 0.0: return stylized_pil
    if strength >= 1.0: # Return content color on stylized luminance
        content_np = img_as_float(content_pil)
        stylized_np = img_as_float(stylized_pil)
        content_lab = rgb2lab(content_np)
        stylized_lab = rgb2lab(stylized_np)
        output_lab = np.stack((stylized_lab[..., 0], content_lab[..., 1], content_lab[..., 2]), axis=-1)
        output_rgb = lab2rgb(output_lab)
        return Image.fromarray(img_as_ubyte(np.clip(output_rgb, 0, 1)))

    content_np = img_as_float(content_pil)
    stylized_np = img_as_float(stylized_pil)
    content_lab = rgb2lab(content_np)
    stylized_lab = rgb2lab(stylized_np)

    # Blend a and b channels
    blended_a = (1 - strength) * stylized_lab[..., 1] + strength * content_lab[..., 1]
    blended_b = (1 - strength) * stylized_lab[..., 2] + strength * content_lab[..., 2]

    # Combine with stylized luminance
    output_lab = np.stack((stylized_lab[..., 0], blended_a, blended_b), axis=-1)
    output_rgb = lab2rgb(output_lab)
    output_rgb_clipped = np.clip(output_rgb, 0, 1) # Clip potential out-of-gamut colors
    return Image.fromarray(img_as_ubyte(output_rgb_clipped))


# --- Main Stylization Function ---
def stylize_image(content_img_pil, style_img_pil, alpha_value,
                  blend_weight, # New: Deep Feature Mix
                  enable_guided, guided_radius, guided_eps, # New: Guided Filter
                  enable_l0, l0_lambda, # New: L0 Smoothing
                  color_preserve_strength, # New: Color Preservation
                  resolution): # New: Resolution
    """
    Performs style transfer with enhanced controls.
    """
    if content_img_pil is None or style_img_pil is None:
        raise gr.Error("Please upload both Content and Style images.")

    network = load_model()
    if network is None:
         raise gr.Error("Model failed to load. Check console for errors.")

    start_time = time.time()

    # 1. Preprocess images (with resolution)
    content_tensor = preprocess_image(content_img_pil, resolution)
    style_tensor = preprocess_image(style_img_pil, resolution)
    original_content_tensor_for_guide = preprocess_image(content_img_pil, resolution) # Keep original for guide

    # 2. Create grayscale versions (if needed by specific logic, maybe not needed if alpha overrides adaptive)
    gray_content_tensor = torchvision.transforms.functional.rgb_to_grayscale(content_tensor).repeat(1, 3, 1, 1)
    # gray_style needed for original adaptive alpha calc, but we override it. Style for WCT part.
    gray_style_tensor = torchvision.transforms.functional.rgb_to_grayscale(style_tensor).repeat(1, 3, 1, 1)


    # 3. Prepare alpha tensor
    alpha_tensor = torch.tensor([[alpha_value]], dtype=torch.float32).to(DEVICE)
    alpha_broadcastable = alpha_tensor.unsqueeze(-1).unsqueeze(-1)

    # 4. Perform Inference
    print(f"Running inference with alpha={alpha_value:.2f}, blend={blend_weight:.2f}...")
    with torch.no_grad():
        # Pass the new blend_weight to the network's forward method
        stylized_tensor, _, _, _, _ = network(
            content_tensor,
            style_tensor,
            alpha_broadcastable, # User alpha overrides adaptive calculation
            gray_content_tensor, # Used for the global WCT part
            style_tensor, # Style input for WCT part
            blend_weight=blend_weight # Pass the new parameter
        )
        print(f"Raw output stylized_tensor shape: {stylized_tensor.shape}")

    # 5. Post-processing Pipeline
    print("Post-processing...")
    processed_tensor = stylized_tensor # Start with the raw network output

    # --- Apply Guided Filter ---
    if enable_guided:
        print(f"Applying Guided Filter (r={guided_radius}, eps={guided_eps:.3f})...")
        # Ensure tensors are on the right device and potentially denormalized
        # Pass the *original* content tensor as guide
        processed_tensor = apply_guided_filter(processed_tensor, original_content_tensor_for_guide, guided_radius, guided_eps)
        print(f"Tensor shape after Guided Filter: {processed_tensor.shape}")
        # Output is denormalized (0-1 range)

    # --- Convert to PIL for next steps (if needed) ---
    # If Guided Filter wasn't applied, denormalize now.
    if not enable_guided:
         output_img_pil = postprocess_image(processed_tensor)
    else:
         # Guided filter output is already denormalized tensor
         output_img_pil = transforms.ToPILImage()(processed_tensor.cpu().squeeze(0))

    # --- Apply L0 Smoothing ---
    if enable_l0:
        print(f"Applying L0 Smoothing (lambda={l0_lambda:.4f})...")
        output_img_pil = apply_l0_smoothing(output_img_pil, l0_lambda)

    # --- Apply Color Preservation ---
    if color_preserve_strength > 0:
        print(f"Applying Color Preservation (strength={color_preserve_strength:.2f})...")
        output_img_pil = apply_color_preservation(output_img_pil, content_img_pil.resize(output_img_pil.size), color_preserve_strength) # Resize content PIL to match


    end_time = time.time()
    print(f"Stylization finished in {end_time - start_time:.2f} seconds.")

    return output_img_pil

# --- Gradio Interface Definition ---
css = """
# ... (existing css) ...
.control-group { border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.control-group-title { font-weight: bold; margin-bottom: 5px; }
"""

with gr.Blocks(css=css, title="AesPA-Net Style Transfer") as demo:
    gr.Markdown("# AesPA-Net: Aesthetic Pattern-Aware Style Transfer")
    gr.Markdown(
        "Upload Content and Style images. Adjust parameters to control the stylization."
    )

    with gr.Row():
        with gr.Column(scale=1):
            content_img = gr.Image(type="pil", label="Content Image")
            style_img = gr.Image(type="pil", label="Style Image")

            with gr.Accordion("Basic Controls", open=True):
                 alpha_slider = gr.Slider(
                     minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                     label="Style Strength (Alpha)",
                     info="Blend: 0.0=Global WCT <> 1.0=Local Attention"
                 )
                 blend_weight_slider = gr.Slider(
                     minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                     label="Deep Feature Mix",
                     info="Blend: 0.0=Conv4 Features <> 1.0=Conv5 Features"
                 )
                 resolution_dd = gr.Dropdown(
                     choices=[256, 512, 768], value=512, label="Processing Resolution",
                     info="Lower resolutions are faster but less detailed."
                 )

            with gr.Accordion("Structure Preservation (Post-processing)", open=False):
                 gr.Markdown("Use **one** of these for best results.")
                 with gr.Group(): # Group Guided Filter controls
                      enable_guided_cb = gr.Checkbox(label="Enable Guided Filter", value=False)
                      guided_radius_slider = gr.Slider(minimum=1, maximum=8, step=1, value=2, label="Guided Filter Radius (r)")
                      guided_eps_slider = gr.Slider(minimum=0.01, maximum=0.2, step=0.01, value=0.04, label="Guided Filter Epsilon (ε)")
                 with gr.Group(): # Group L0 Smoothing controls
                      enable_l0_cb = gr.Checkbox(label="Enable L0 Smoothing", value=False)
                      l0_lambda_slider = gr.Slider(minimum=0.005, maximum=0.05, step=0.001, value=0.015, label="L0 Lambda (λ)")

            with gr.Accordion("Color Control (Post-processing)", open=False):
                 color_preserve_slider = gr.Slider(
                     minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                     label="Content Color Preservation",
                     info="0.0=Stylized Color, 1.0=Content Color"
                 )

            submit_btn = gr.Button("Stylize!", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Stylized Output")

    gr.Markdown("---")
    gr.Markdown("Official Implementation: [GitHub](https://github.com/Kibeom-Hong/AesPA-Net) | Paper: [arXiv](https://arxiv.org/abs/2307.09724)")

    # Link components to the function
    inputs = [
        content_img, style_img, alpha_slider,
        blend_weight_slider,
        enable_guided_cb, guided_radius_slider, guided_eps_slider,
        enable_l0_cb, l0_lambda_slider,
        color_preserve_slider,
        resolution_dd
    ]
    submit_btn.click(
        fn=stylize_image,
        inputs=inputs,
        outputs=output_img
    )

    # Example loading (update defaults if needed)
    gr.Examples(
        examples=[
            ["examples/content1.png", "examples/style1.png", 0.7, 0.5, False, 2, 0.04, False, 0.015, 0.0, 512],
            ["examples/content2.png", "examples/style2.png", 0.5, 0.5, True, 3, 0.05, False, 0.015, 0.1, 512],
             ["examples/content3.png", "examples/style3.png", 0.9, 0.7, False, 2, 0.04, True, 0.020, 0.0, 512],
        ],
        inputs=inputs, # Ensure order matches function signature
        outputs=output_img,
        fn=stylize_image,
        cache_examples=False,
    )

# --- Launch the App ---
if __name__ == "__main__":
    # Pre-load model
    load_model()
    demo.launch(share=False)