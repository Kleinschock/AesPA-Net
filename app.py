# app.py
import gradio as gr
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
import time

# --- Imports from project files ---
# Assuming app.py is in the same directory as other .py files
try:
    from baseline import Baseline  # Imports Baseline class
    from aespanet_models import Baseline_net, VGGEncoder, VGGDecoder, AdaptiveMultiAttn_Transformer_v2, size_arrange, \
        calc_mean_std, mean_variance_norm # Need specific imports if Baseline doesn't expose enough


    from utils import denorm_2, gram_matrix, feature_wct_simple

    # Add any other necessary imports from your project files
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
args = parser.parse_args([]) # Parse empty list to use defaults

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
MODEL_LOADED = False
aespa_model = None

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
# Define transformations based on utils and training setup
# Normalization used in the original project (from utils.py _normalizer)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
img_transforms = transforms.Compose([
    # size_arrange is applied dynamically in the function
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def preprocess_image(img_pil):
    """ Converts PIL Image to normalized tensor and applies size_arrange. """
    # Apply basic transforms (ToTensor, Normalize)
    tensor = img_transforms(img_pil).unsqueeze(0) # Add batch dimension

    # Apply size_arrange dynamically
    tensor = size_arrange(tensor)
    return tensor.to(DEVICE)

def postprocess_image(tensor):
    """ Converts tensor back to PIL Image after denormalization. """
    # Use denorm_2 function from utils.py
    # Ensure tensor is detached and moved to CPU
    denormalized_tensor = denorm_2(tensor.detach().cpu()) # Use denorm_2

    # Squeeze batch dimension if necessary and convert to PIL
    # Assuming denorm_2 keeps the batch dim, we still need squeeze(0)
    pil_image = transforms.ToPILImage()(denormalized_tensor.squeeze(0))
    return pil_image

# --- Main Stylization Function ---
def stylize_image(content_img_pil, style_img_pil, alpha_value):
    """
    Performs style transfer using the loaded AesPA-Net model.

    Args:
        content_img_pil (PIL.Image): Content image.
        style_img_pil (PIL.Image): Style image.
        alpha_value (float): Style strength parameter (0.0 to 1.0).

    Returns:
        PIL.Image: Stylized output image.
    """
    if content_img_pil is None or style_img_pil is None:
        raise gr.Error("Please upload both Content and Style images.")

    print(f"[DEBUG app.stylize_image] Input Content PIL size: {content_img_pil.size}, Style PIL size: {style_img_pil.size}") # <-- ADDED DEBUG
    network = load_model()
    if network is None:
         raise gr.Error("Model failed to load. Check console for errors.")

    start_time = time.time()

    # 1. Preprocess images
    print("[DEBUG app.stylize_image] Preprocessing images...")  # <-- ADDED DEBUG
    content_tensor = preprocess_image(content_img_pil)
    style_tensor = preprocess_image(style_img_pil)
    print(
        f"[DEBUG app.stylize_image] Content tensor shape: {content_tensor.shape}, dtype: {content_tensor.dtype}")  # <-- ADDED DEBUG
    print(
        f"[DEBUG app.stylize_image] Style tensor shape: {style_tensor.shape}, dtype: {style_tensor.dtype}")  # <-- ADDED DEBUG

    # 2. Create grayscale versions
    print("[DEBUG app.stylize_image] Creating grayscale tensors...")  # <-- ADDED DEBUG
    gray_content_tensor = torchvision.transforms.functional.rgb_to_grayscale(content_tensor).repeat(1, 3, 1, 1)
    gray_style_tensor = torchvision.transforms.functional.rgb_to_grayscale(style_tensor).repeat(1, 3, 1, 1)
    print(f"[DEBUG app.stylize_image] Gray Content tensor shape: {gray_content_tensor.shape}")  # <-- ADDED DEBUG
    print(f"[DEBUG app.stylize_image] Gray Style tensor shape: {gray_style_tensor.shape}")  # <-- ADDED DEBUG

    # 3. Prepare alpha tensor
    alpha_tensor = torch.tensor([[alpha_value]], dtype=torch.float32).to(DEVICE)
    alpha_broadcastable = alpha_tensor.unsqueeze(-1).unsqueeze(-1)
    print(
        f"[DEBUG app.stylize_image] Alpha tensor shape: {alpha_broadcastable.shape}, value: {alpha_value}")  # <-- ADDED DEBUG

    # 4. Perform Inference
    print(f"[DEBUG app.stylize_image] Running inference with alpha = {alpha_value:.2f}...")  # <-- ADDED DEBUG
    with torch.no_grad():
        stylized_tensor, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = network(
            content_tensor,
            style_tensor,
            alpha_broadcastable,
            gray_content_tensor,
            style_tensor  # Original code uses style_tensor here, matching it
        )
        print(
            f"[DEBUG app.stylize_image] Raw output stylized_tensor shape: {stylized_tensor.shape}, dtype: {stylized_tensor.dtype}")  # <-- ADDED DEBUG

    # 5. Postprocess output image
    print("[DEBUG app.stylize_image] Postprocessing image...")  # <-- ADDED DEBUG
    output_img_pil = postprocess_image(stylized_tensor)
    print(f"[DEBUG app.stylize_image] Output PIL image size: {output_img_pil.size}")  # <-- ADDED DEBUG

    end_time = time.time()
    print(f"Stylization finished in {end_time - start_time:.2f} seconds.")

    return output_img_pil

# --- Gradio Interface Definition ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
footer { display: none !important; }
.gr-image { min-width: 256px !important; }
"""

with gr.Blocks(css=css, title="AesPA-Net Style Transfer") as demo:
    gr.Markdown("# AesPA-Net: Aesthetic Pattern-Aware Style Transfer")
    gr.Markdown(
        "Upload a **Content Image** and a **Style Image**, adjust the **Style Strength (Alpha)**, and click **Stylize!** "
        "Alpha controls the blend between global style patterns (low alpha) and fine-grained local patterns (high alpha)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            content_img = gr.Image(type="pil", label="Content Image")
            style_img = gr.Image(type="pil", label="Style Image")
            alpha_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05, value=0.7, # Default based on README visual results
                label="Style Strength (Alpha)",
                info="Higher values emphasize fine details from style."
            )
            submit_btn = gr.Button("Stylize!", variant="primary")
        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Stylized Output")

    gr.Markdown("---")
    gr.Markdown("Official Implementation: [GitHub](https://github.com/Kibeom-Hong/AesPA-Net) | Paper: [arXiv](https://arxiv.org/abs/2307.09724)")


    # Link components to the function
    submit_btn.click(
        fn=stylize_image,
        inputs=[content_img, style_img, alpha_slider],
        outputs=output_img
    )

    # Example loading
    gr.Examples(
        examples=[
            ["examples/content1.png", "examples/style1.png", 0.7],
            ["examples/content2.png", "examples/style2.png", 0.5],
            ["examples/content3.png", "examples/style3.png", 0.9],
        ],
        inputs=[content_img, style_img, alpha_slider],
        outputs=output_img,
        fn=stylize_image,
        cache_examples=False, # Re-run examples every time
    )


# --- Launch the App ---
if __name__ == "__main__":
    # Pre-load the model before starting the server for faster UI response
    load_model()
    # Launch the Gradio app
    demo.launch(share=False) # Set share=True to get a temporary public link