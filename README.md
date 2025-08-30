# AesPA-Net: Enhanced for High-Resolution and Artistic Control

This project enhances the official PyTorch implementation of "AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks" (ICCV 2023), focusing on overcoming practical limitations of the original model to make it a more powerful and flexible tool for artists and creators.

The core of this project is a sophisticated neural network that transfers the artistic style of one image to another. While the original implementation produces excellent results, it has several limitations that have been addressed in this fork.

![teaser](https://github.com/Kibeom-Hong/AesPA-Net/assets/77425614/8653065b-9554-4481-8673-caa797dab6e2)

## Overcoming the Limitations of the Original AesPA-Net

This fork introduces several key enhancements, each designed to solve a specific problem with the original implementation.

### 1. High-Resolution Image Support

*   **Problem:** The original model was trained on and optimized for 512x512 images, making it difficult to apply to larger, print-quality images without downscaling and losing significant detail.
*   **Solution:** I engineered a **tiling and stitching engine** that intelligently breaks down high-resolution images into smaller, manageable tiles. Each tile is processed individually by the model, and the results are then seamlessly stitched back together using a weighted blending algorithm to avoid visible seams. This allows for the stylization of virtually any image size, even on consumer-grade hardware.

### 2. User-Friendly Interface and Artistic Control

*   **Problem:** The original implementation is command-line only, creating a high barrier to entry for artists and other users not comfortable with terminals. Furthermore, the stylization process is a "one-shot" operation, offering little to no control over the final aesthetic.
*   **Solution:** I developed an **interactive web UI** using Gradio. This interface provides a simple, intuitive workflow for users to upload images, adjust settings, and see results in real-time. It also introduces a **two-step refinement process**, where users first generate a fully stylized image and then use interactive sliders to blend it with the original, allowing for precise control over the final look.

### 3. Color and Tone Fidelity

*   **Problem:** Standard style transfer techniques can often result in unnatural color shifts or a loss of detail in the dark and bright areas of the content image, compromising the final composition.
*   **Solution:** I implemented a suite of **advanced color preservation and blending modes**. By operating in different color spaces (LAB, HSV, YCbCr), the tool can preserve the original color palette of the content image while applying the style's structure. Additionally, I engineered techniques like **Luminance Matching and Dark Area Protection** to maintain the tonal range of the original image, resulting in more natural and aesthetically pleasing outputs.

## Getting Started

### Prerequisites

*   Python 3.7
*   CUDA 11.1
*   PyTorch 1.7.1
*   NumPy 1.19.2
*   Pillow 8.0.1
*   imageio 2.9.0
*   SciPy 1.5.2
*   Gradio
*   scikit-image

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Kleinschock/AesPA-Net.git
    cd AesPA-Net
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Pre-trained Models

1.  **VGG-19 Encoder:** Download the pre-trained VGG-19 encoder from [here](https://drive.google.com/drive/folders/1HsJNskEMC5HUimq6ixkSZk7W_hgFNp7J?usp=sharing) and place it in the `./baseline_checkpoints` directory.

2.  **AesPA-Net Decoder and Transformer:** Download the pre-trained decoder and transformer from the following links:
    *   [Decoder](https://drive.google.com/file/d/1nb7dQwj7RcQpi8_cURvErSwA-BxyZTT5/view?usp=sharing)
    *   [Transformer](https://drive.google.com/file/d/1YII45EfR3mVbyvqQlzvfiYFIoTCgGG_R/view?usp=sharing)

3.  Move the downloaded weights to the following directories:
    *   `transformer_model.pth` -> `./trained_models/`
    *   `dec_model.pth` -> `./trained_models/`

## Usage

### Web-Based GUI (Recommended)

To launch the interactive web interface, run the following command:

```bash
python gui.py
```

This will start a local web server and provide a URL to access the interface in your browser. The GUI provides a two-step workflow:

1.  **Step 1: Generate Stylized Image:** Upload your content and style images, adjust the tiling and upscaling options, and click "Run Stylization."
2.  **Step 2: Blend & Refine:** Use the blending controls to adjust the strength of the stylization and choose from various color preservation modes to fine-tune the result.

### Command-Line Interface (for batch processing)

For batch processing and training, you can use the original command-line interface.

#### Inference

```bash
python main.py --type test --content_dir <path_to_content_image> --style_dir <path_to_style_image> --test_result_dir <output_directory>
```

#### Training

```bash
python main.py --type train --content_dir <path_to_content_dataset> --style_dir <path_to_style_dataset> --train_result_dir <output_directory>
```

## Original Paper and Citation

This work is built upon the foundation of the original AesPA-Net paper. If you find this work useful for your research, please cite the original authors:

```
@InProceedings{Hong_2023_ICCV,
    author    = {Hong, Kibeom and Jeon, Seogkyu and Lee, Junsoo and Ahn, Namhyuk and Kim, Kunhee and Lee, Pilhyeon and Kim, Daesik and Uh, Youngjung and Byun, Hyeran},
    title     = {AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22758-2267}
}
```

## Contact

For questions about the original paper, please contact the first author, Kibeom Hong, at [kibeom9212@gmail.com](mailto:kibeom9212@gmail.com).
