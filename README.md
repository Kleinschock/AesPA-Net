# AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks

### Official Pytorch Implementation of 'AesPA-Net' (ICCV 2023)
##### (Note that this project is totally powerd by Yonsei University)
![teaser](https://github.com/Kibeom-Hong/AesPA-Net/assets/77425614/8653065b-9554-4481-8673-caa797dab6e2)

> ## AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks
>
>  Paper[CVF] : [Link](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_AesPA-Net_Aesthetic_Pattern-Aware_Style_Transfer_Networks_ICCV_2023_paper.pdf)
>
>  Paper[Arxiv] : [Link](https://arxiv.org/abs/2307.09724)
>
> **Abstract**: *[Original Abstract retained]* ... Through qualitative and quantitative evaluations, we verify the reliability of the proposed pattern repeatability that aligns with human perception, and demonstrate the superiority of the proposed framework.

---

## Prerequisites and Setup (Updated for Modern Systems)

This section details the setup process, reflecting updates needed for compatibility with newer hardware and drivers compared to the original project environment.

### Original Dependencies (May Cause Issues on Newer Systems)

The project was originally developed with the following environment:
- Python==3.7
- CUDA==11.1
- Pytorch==1.7.1
- numpy==1.19.2
- Pillow==8.0.1
- imageio==2.9.0
- scipy==1.5.2

**Important:** Attempting to replicate this exact environment on systems with newer NVIDIA drivers (e.g., those supporting CUDA 12.x) may lead to issues where PyTorch fails to detect the GPU (`torch.cuda.is_available()` returns `False`), even if the correct packages seem installed. This is due to incompatibilities between the old PyTorch 1.7.1 (built for CUDA 11.1) and modern drivers.

### Recommended Setup (Tested Environment)

Based on recent testing and troubleshooting, the following environment is recommended for compatibility with modern GPUs and drivers (like NVIDIA RTX 30xx/40xx series):

- **Python:** 3.10 (Python 3.8+ should also work)
- **CUDA:** A version compatible with both your driver and a recent PyTorch version (e.g., CUDA 11.8 was used successfully). Check your driver's supported CUDA version using `nvidia-smi`.
- **PyTorch:** A recent version compatible with your chosen CUDA (e.g., PyTorch 2.0.1+ was used successfully).
- **Other dependencies:** Versions compatible with Python 3.10+ and PyTorch 2.x+ (use the updated `requirements.txt` below).

### Installation Steps (Recommended)

1.  **Install Anaconda or Miniconda:** If you don't have it, download and install it from the official website. This provides the `conda` package manager.

2.  **Create a New Conda Environment:** Using a dedicated environment prevents conflicts. Replace `aespav_new` with your desired name and choose a Python version (3.10 recommended):
    ```bash
    conda create --name aespav_new python=3.10 -y
    ```

3.  **Activate the Environment:**
    ```bash
    conda activate aespav_new
    ```

4.  **Install PyTorch with CUDA Support:** This is the most critical step. Use `conda` to install PyTorch, as it handles the complex CUDA dependencies better than `pip`. Choose versions compatible with your system. The following command worked successfully with drivers supporting CUDA 12.x:
    ```bash
    # Installs PyTorch 2.0.1, Torchvision 0.15.2, Torchaudio 2.0.2 with CUDA 11.8 support
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ```
    *   **Verify:** After installation, run `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"`. Ensure `torch.cuda.is_available()` returns `True`. If not, double-check the command, CUDA driver compatibility, and consult PyTorch installation guides.

5.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kibeom-Hong/AesPA-Net.git
    cd AesPA-Net
    ```

6.  **Prepare `requirements.txt`:** Create a file named `requirements.txt` in the `AesPA-Net` directory with the following content. This version removes pinned older versions that cause build issues with newer Python/PyTorch.
    ```text
    # AesPA-Net Requirements

    # IMPORTANT: Install PyTorch + CUDA FIRST using Conda (see Step 4 above)
    # before running `pip install -r requirements.txt`.

    # --- Do not uncomment the lines below ---
    # torch
    # torchvision
    # torchaudio
    # --- PyTorch should be installed via Conda as shown above ---

    # Dependencies to be installed by pip AFTER PyTorch is installed via Conda:

    # Unpinned versions - let pip find compatible versions for Python 3.10+ and PyTorch 2.x+
    numpy
    Pillow
    imageio
    scipy

    # Other dependencies
    matplotlib
    scikit-video
    wandb
    torchfile
    opencv-python
    natsort
    ```

7.  **Install Remaining Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on pip warnings:** You might see warnings from pip about dependency conflicts related to packages *not* listed in this file (e.g., `tensorflow`, `jupyter`). If the packages listed *above* installed successfully, these warnings can usually be ignored for running AesPA-Net, as they relate to other potentially installed packages in your environment.

8.  **Install `ffmpeg` (for `skvideo`):** The `scikit-video` library often requires `ffmpeg` for video processing. The easiest way to install it within the conda environment is:
    ```bash
    conda install ffmpeg -c conda-forge -y
    ```

---

## Usage

### 1. Set Pretrained Weights

*   **VGG Encoder:** The VGG-19 encoder weights are needed.
    *   Download `vgg_normalised_conv5_1.t7` from [here](https://drive.google.com/drive/folders/1HsJNskEMC5HUimq6ixkSZk7W_hgFNp7J?usp=sharing).
    *   Place the file in the `./baseline_checkpoints/` directory (create the directory if it doesn't exist).
    *   *Note:* The original `.t7` format requires the `torchfile` package, which should be installed via `requirements.txt`.

*   **AesPA-Net Decoder & Transformer:** Download the specific weights for AesPA-Net.
    *   Decoder: [dec_model.pth](https://drive.google.com/file/d/1nb7dQwj7RcQpi8_cURvErSwA-BxyZTT5/view?usp=sharing)
    *   Transformer: [transformer_model.pth](https://drive.google.com/file/d/1YII45EfR3mVbyvqQlzvfiYFIoTCgGG_R/view?usp=sharing)

*   **Place AesPA-Net Weights:** These weights need to be placed in a specific directory structure related to the `--comment` argument you use when running the script. Create the directories if they don't exist.
    *   Move `transformer_model.pth` to `./train_results/<comment>/log/transformer_model.pth`
    *   Move `dec_model.pth` to `./train_results/<comment>/log/dec_model.pth`
    *   Replace `<comment>` with the actual comment string you will use (e.g., `aepapa_run1` if your command is `python main.py --comment aepapa_run1 ...`).

### 2. Code Modification Note (Already Applied)

*   The original code had an issue loading the VGG weights because it expected a model object instead of a state dictionary (`AttributeError: 'collections.OrderedDict' object has no attribute 'modules'`).
*   **The code in this repository has been updated** with the necessary fix in the `VGGEncoder` class within `aespanet_models.py`. You do *not* need to apply this fix manually.

### 3. Inference

*   **Activate your conda environment:** `conda activate aespav_new`
*   **Run using the script (modify paths inside):**
    ```bash
    bash scripts/test_styleaware_v2.sh
    ```
*   **Run using python command:** Replace placeholders (`#batch_size`, `<comment>`, paths) as needed. Ensure `<comment>` matches the directory where you placed the AesPA-Net weights.
    ```bash
    python main.py --type test --batch_size 1 --comment aepapa_run1 --content_dir ./content --style_dir ./style --num_workers 4 --test_result_dir ./test_results
    ```
    *   `--content_dir`: Path to your content images.
    *   `--style_dir`: Path to your style images.
    *   `--comment`: Used to locate the pre-trained decoder/transformer weights (must match the directory name under `./train_results/`).
    *   `--test_result_dir`: Where stylized output images will be saved.

### 4. Training

*   **Activate your conda environment:** `conda activate aespav_new`
*   Prepare your content and style datasets according to the paths specified in the training script or command-line arguments.
*   Run using the script (modify paths and parameters inside):
    ```bash
    bash scripts/train_styleaware_v2.sh
    ```
*   Or use the python command directly (adjust arguments):
    ```bash
    python main.py --type train --comment <your_training_run_name> --content_dir <path_to_coco> --style_dir <path_to_wikiart> --batch_size 4 --max_iter 100000 --lr 1e-4 --imsize 512 --cropsize 256 --num_workers 8
    ```

## Troubleshooting / Notes

*   **`AssertionError: Torch not compiled with CUDA enabled` or `torch.cuda.is_available() == False`:** This means PyTorch cannot detect your GPU. This usually indicates either a CPU-only PyTorch installation or an incompatibility between your PyTorch build's CUDA version and your installed NVIDIA driver version. Ensure you follow the **Recommended Setup** steps above, installing PyTorch+CUDA via `conda` using versions known to be compatible.
*   **`FileNotFoundError: ... dec_model_.pth` or `... transformer_model_.pth`:** The script cannot find the pretrained AesPA-Net weights. Make sure you have downloaded them and placed them in the correct `./train_results/<comment>/log/` directory, where `<comment>` exactly matches the `--comment` argument you provided to `main.py`.
*   **`ModuleNotFoundError: No module named 'X'`:** A required package is missing. Ensure you have activated the correct conda environment (`aespav_new`) and successfully run `pip install -r requirements.txt` after installing PyTorch via conda. If the error is for `skvideo`, ensure `ffmpeg` is installed (`conda install ffmpeg -c conda-forge`).
*   **Build Errors (e.g., during `pip install numpy` or others):** Often caused by trying to install very old package versions (pinned in the original `requirements.txt`) on a newer Python version. Use the updated `requirements.txt` provided in the setup steps, which unpins these versions. Ensure you have necessary build tools (like a C++ compiler, often included with Visual Studio Build Tools on Windows or `build-essential` on Linux) if pip needs to compile packages from source.
*   **PIP Dependency Warnings:** Warnings during `pip install -r requirements.txt` about packages like `tensorflow`, `jupyter`, `nn-framework` can usually be ignored if the packages *listed* in `requirements.txt` installed correctly. These warnings relate to other packages potentially present in your environment but not used by AesPA-Net.

---


#### Evaluation
Available soon


## Citation
If you find this work useful for your research, please cite:
```
@InProceedings{Hong_2023_ICCV,
    author    = {Hong, Kibeom and Jeon, Seogkyu and Lee, Junsoo and Ahn, Namhyuk and Kim, Kunhee and Lee, Pilhyeon and Kim, Daesik and Uh, Youngjung and Byun, Hyeran},
    title     = {AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22758-22767}
}
```

```
@article{Hong2023AesPANetAP,
  title={AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks},
  author={Kibeom Hong and Seogkyu Jeon and Junsoo Lee and Namhyuk Ahn and Kunhee Kim and Pilhyeon Lee and Daesik Kim and Youngjung Uh and Hyeran Byun},
  journal={ArXiv},
  year={2023},
  volume={abs/2307.09724},
  url={https://api.semanticscholar.org/CorpusID:259982728}
}
```

## Contact
If you have any question or comment, please contact the first author of this paper - Kibeom Hong

[kibeom9212@gmail.com](kibeom9212@gmail.com)
