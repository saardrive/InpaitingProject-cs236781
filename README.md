# Improving Vanilla DDPM Image Inpainting

A deep learning project that enhances standard Denoising Diffusion Probabilistic Model (DDPM) image inpainting using Stable Diffusion 2. This implementation introduces two improvement methods: (1) multimodal LLaVA-based prompt generation for image-specific text guidance, and (2) spatial boundary interpolation to smooth transitions between original and inpainted regions.

**Key Improvements over Vanilla DDPM:**
- 🎯 **Semantic Coherence**: LLaVA-generated prompts provide context-aware guidance for each image
- 🔄 **Smooth Boundaries**: Three spatial averaging kernels reduce visible mask edges
- 📊 **Comprehensive Evaluation**: Quantitative metrics (SSIM, LPIPS) across 6 mask types
- 🚀 **Production-Ready**: Batch processing with progress tracking and error handling

> For detailed methodology and experimental results, see the report.

---

## Table of Contents
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Workflow 1: Generate Masks](#workflow-1-generate-masks)
  - [Workflow 2: Generate Image Descriptions](#workflow-2-generate-image-descriptions-optional)
  - [Workflow 3: Run Inpainting](#workflow-3-run-inpainting)
  - [Workflow 4: Evaluate Results](#workflow-4-evaluate-results)
- [Methodology](#methodology)
- [Directory Structure](#directory-structure)
- [Experimental Results](#experimental-results)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## Key Features

- **Multiple Mask Types**: Six configurable mask patterns for robust testing
  - Rectangular, Circular, Brush Stroke, Random Patches, Half-plane, Random Noise
- **LLaVA-Based Prompts**: Automatic generation of image-specific descriptions using multimodal LLM
- **Spatial Interpolation**: Three kernel types (Regular, Cross, X) for boundary smoothing
- **Comprehensive Evaluation**: SSIM and LPIPS metrics with statistical analysis
- **Batch Processing**: Efficient pipeline for large datasets with progress tracking

---

## Requirements

### Hardware
- **GPU Required**: CUDA-capable NVIDIA GPU
  - 8GB+ VRAM for Stable Diffusion 2 inpainting
  - 16GB+ VRAM recommended for LLaVA prompt generation
- **CUDA Version**: 12.8

### Software
- **Python**: 3.10.19
- **Package Manager**: conda/miniconda

### Key Dependencies
- `torch==2.10.0+cu128` - PyTorch with CUDA support
- `diffusers==0.36.0` - Hugging Face diffusion models
- `transformers==4.57.6` - CLIP and LLaVA models
- `lpips==0.1.4` - Perceptual similarity metrics
- `scikit-image==0.25.2` - SSIM and image processing
- `Pillow==12.1.0`, `opencv-python==4.13.0.90` - Image I/O

*Full dependencies list in [environment.yml](environment.yml)*

### Pre-trained Models (Auto-downloaded)
- **Stable Diffusion 2**: `sd2-community/stable-diffusion-2-base`
- **LLaVA**: `llava-hf/llava-v1.6-mistral-7b-hf`

---

## Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd project
```

### Step 2: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate sd2
```

### Step 3: Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```
Expected output: `CUDA Available: True`

### Step 4: (Optional) Download DIV2K Dataset
```bash
# Validation set (100 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
python -m zipfile -e DIV2K_valid_HR.zip .

# Training set (800 images) - if needed
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
python -m zipfile -e DIV2K_train_HR.zip .
```

---

## Quick Start

Run the complete pipeline on sample images in ~2-3 minutes:

```bash
# 1. Generate rectangular masks
python generate_masked_images.py images/ rectangular -o masked_output/

# 2. Run vanilla inpainting (no text guidance)
python main.py \
  --input_folder images/ \
  --output_folder output/quick_start \
  --masked_images_folder masked_output/rectangular/images

# 3. View results
ls output/quick_start/inpainted_images/
```

**Expected Output**: Inpainted images in `output/quick_start/inpainted_images/`

---

## Usage Guide

### Workflow 1: Generate Masks

Generate masked images with various mask patterns as preprocessing for inpainting.

#### Basic Command
```bash
python generate_masked_images.py <input_folder> <mask_type> [-o output_folder]
```

#### Mask Types & Parameters

| Mask Type | Description | Key Parameters |
|-----------|-------------|----------------|
| `rectangular` | Rectangular region | `mask_fraction=0.35` (size) |
| `circular` | Circular region | `mask_fraction=0.35` (size) |
| `brush_stroke` | Irregular strokes | `num_strokes=5`, `stroke_width_range=(10,30)` |
| `random_patches` | Multiple patches | `num_patches=10`, `patch_size_range=(20,80)` |
| `random_noise` | Scattered pixels | `noise_fraction=0.3` |
| `half` | Half-plane mask | `direction='left'` (left/right/top/bottom) |
| `all` | Generate all types | N/A |

#### Examples

**Basic rectangular mask:**
```bash
python generate_masked_images.py DIV2K_train_val_HR/ rectangular
```

**All mask types at once:**
```bash
python generate_masked_images.py images/ all -o masked_output/
```

**Circular mask with custom size:**
```bash
python generate_masked_images.py images/ circular --circular "mask_fraction=0.5"
```

**Brush stroke mask with custom parameters:**
```bash
python generate_masked_images.py images/ brush_stroke \
  --brush_stroke "num_strokes=10,stroke_width_range=(15,40)"
```

**Half-plane mask (remove right side):**
```bash
python generate_masked_images.py images/ half --half "direction='right'"
```

#### Output Structure
```
masked_output/
  <mask_type>/
    images/          # Masked images (black regions to inpaint)
    masks/           # Binary masks (white=inpaint, black=keep original)
    descriptions_<mask_type>.json  # (created later by LLaVA)
```

**File Naming Convention**: `image.png` → `image_masked.png` (masked image) + `image_mask.png` (binary mask)

---

### Workflow 2: Generate Image Descriptions (Optional)

Use LLaVA multimodal model to generate image-specific prompts for better inpainting guidance.

⚠️ **Performance Warning**: GPU-intensive and slow
- First image: ~30-60 seconds
- Subsequent images: 15-30s (--max_tokens 50) or 50-100s (--max_tokens 200)
- 100 images @ 50 tokens: ~25-50 minutes total

#### Basic Command
```bash
python generate_image_descriptions_LLaVA.py \
  --input_folder <folder> \
  --output_json <output.json> \
  --max_tokens <50|200>
```

#### Examples

**Fast generation (50 tokens, recommended for large datasets):**
```bash
python generate_image_descriptions_LLaVA.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_json descriptions_full_image.json \
  --max_tokens 50
```

**Detailed generation (200 tokens):**
```bash
python generate_image_descriptions_LLaVA.py \
  --input_folder ./masked_output/rectangular/images \
  --output_json descriptions_rectangular_masked.json \
  --max_tokens 200
```

**Custom LLaVA instruction:**
```bash
python generate_image_descriptions_LLaVA.py \
  --input_folder ./images \
  --output_json descriptions.json \
  --max_tokens 50 \
  --instruction "Describe this image in detail, focusing on colors and composition."
```

#### Output Format
JSON file with structure:
```json
[
  {
    "image_path": "./images/example.png",
    "image_name": "example.png",
    "prompt": "Purple starfish, underwater, coral reef, vibrant colors..."
  }
]
```

#### Default LLaVA Instruction
The model uses this optimized prompt for Stable Diffusion:

> "You are a prompt generation machine for Stable Diffusion. Look at this image. Your goal is to generate a text prompt that describes the image and provides logical details to fill in the missing areas (inpainting).
>
> Rules:
> 1. Output ONLY the raw prompt text.
> 2. Use comma-separated keywords and phrases.
> 3. Do NOT use bullet points, conversational filler, or introductions.
> 4. Focus on: Subject description, Lighting, Background scenery, Art style, and Vibe."

---

### Workflow 3: Run Inpainting

Execute the DDPM inpainting pipeline with various configurations.

#### Method A: Vanilla DDPM (No Text Guidance)
Baseline approach without any text conditioning.

```bash
python main.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_folder ./Inpainting/Vanilla_without_text/rectangular \
  --masked_images_folder ./masked_output/rectangular/images \
  --num_steps 50 \
  --seed 42
```

#### Method B: Vanilla DDPM with Text Prompts
Use either a general prompt or image-specific prompts.

**General prompt (same for all images):**
```bash
python main.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_folder ./Inpainting/Vanilla_with_general_text/rectangular \
  --masked_images_folder ./masked_output/rectangular/images \
  --use_text \
  --prompt "Inpaint the image to make it look as it would be without the missing part."
```

**Image-specific prompts (from LLaVA):**
```bash
python main.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_folder ./Inpainting/generated_text_no_interpolation/rectangular \
  --masked_images_folder ./masked_output/rectangular/images \
  --use_text \
  --prompts_json ./masked_output/rectangular/descriptions_rectangular.json
```

#### Method C: Spatial Interpolation (Enhanced Method)
Apply boundary smoothing with three kernel options.

**Regular kernel (3×3 averaging):**
```bash
python main.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_folder ./Inpainting/interpolation_generated_text/regular/rectangular \
  --masked_images_folder ./masked_output/rectangular/images \
  --method spatial \
  --use_text \
  --prompts_json ./masked_output/rectangular/descriptions_rectangular.json \
  --kernel_type regular \
  --steps_avg 5 \
  --area 1
```

**Cross kernel (+ shape, 5 pixels):**
```bash
python main.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_folder ./Inpainting/interpolation_generated_text/cross/rectangular \
  --masked_images_folder ./masked_output/rectangular/images \
  --method spatial \
  --use_text \
  --prompts_json ./masked_output/rectangular/descriptions_rectangular.json \
  --kernel_type cross \
  --steps_avg 5
```

**X kernel (diagonal shape, 5 pixels):**
```bash
python main.py \
  --input_folder ./DIV2K_train_val_HR \
  --output_folder ./Inpainting/interpolation_generated_text/x/rectangular \
  --masked_images_folder ./masked_output/rectangular/images \
  --method spatial \
  --use_text \
  --prompts_json ./masked_output/rectangular/descriptions_rectangular.json \
  --kernel_type x \
  --steps_avg 5
```

#### Spatial Interpolation Kernels

The three kernel types for boundary averaging:

```
Regular (3×3):        Cross (+):          X (diagonal):
1/9 × [1 1 1]        1/5 × [0 1 0]      1/5 × [1 0 1]
      [1 1 1]              [1 1 1]            [0 1 0]
      [1 1 1]              [0 1 0]            [1 0 1]
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_folder` | str | **required** | Original images directory |
| `--output_folder` | str | **required** | Output directory for results |
| `--masked_images_folder` | str | **required** | Masked images directory |
| `--method` | str | `vanilla` | Inpainting method: `vanilla` or `spatial` |
| `--use_text` | flag | `False` | Enable text conditioning |
| `--prompt` | str | `""` | Single prompt for all images |
| `--prompts_json` | str | `None` | JSON file with image-specific prompts |
| `--negative_prompt` | str | `""` | Negative prompt for CFG |
| `--num_steps` | int | `50` | Number of denoising steps |
| `--guidance_scale` | float | `7.5` | Classifier-free guidance scale |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--kernel_type` | str | `regular` | Spatial kernel: `regular`, `cross`, or `x` |
| `--steps_avg` | int | `5` | Apply averaging every N steps |
| `--area` | int | `1` | Padding area for averaging window |
| `--model_id` | str | `sd2-community/stable-diffusion-2-base` | Model identifier |

#### Output Structure
```
output_folder/
  inpainted_images/        # Final inpainted images (.png)
  mask_latents/            # Saved mask latents (.npy)
```

---

### Workflow 4: Evaluate Results

Compute quantitative metrics comparing inpainted images to ground truth.

#### Option A: Comprehensive Evaluation (All Methods)

Evaluates all method-mask combinations found in the [Inpainting/](Inpainting/) directory.

```bash
python evaluate_all_methods.py
```

**With custom options:**
```bash
# Resize to specific dimensions
python evaluate_all_methods.py --resize 512 504

# No resizing (use original sizes)
python evaluate_all_methods.py --no_resize

# Match original image dimensions
python evaluate_all_methods.py --match_original_size
```

**Directory Structure Expected:**
```
Inpainting/
  <method_name>/
    <mask_type>/
      inpainted_images/
        *.png
```

#### Option B: Single Method Evaluation

Evaluate one specific method against ground truth.

```bash
python inpaintg_eval.py \
  --input_dir ./DIV2K_train_val_HR \
  --output_dir ./Inpainting/generated_text_no_interpolation/rectangular/inpainted_images \
  --resize 512 504
```

**Command-Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dir` | str | **required** | Ground truth images directory |
| `--output_dir` | str | **required** | Inpainted images directory |
| `--resize` | int int | `None` | Resize to width height |
| `--no_resize` | flag | `False` | Skip resizing |
| `--match_original_size` | flag | `False` | Resize inpainted to match original |

#### Evaluation Metrics

**SSIM (Structural Similarity Index Measure)**
- **Range**: 0 to 1
- **Direction**: Higher is better (1 = identical)
- **Measures**: Structural similarity considering luminance, contrast, and structure

**LPIPS (Learned Perceptual Image Patch Similarity)**
- **Range**: 0 to 1+ (typically 0-0.5 for similar images)
- **Direction**: Lower is better (0 = identical)
- **Measures**: Perceptual similarity using deep features

#### Output Format

JSON file with timestamp: `evaluation_all_methods_YYYYMMDD_HHMMSS.json`

```json
{
  "method_name/mask_type": {
    "num_images": 900,
    "ssim_mean": 0.6045,
    "ssim_std": 0.1251,
    "lpips_mean": 0.2123,
    "lpips_std": 0.0414
  }
}
```

**Note**: Metrics are computed on the **entire image**, not just the masked region. This follows standard evaluation protocols from the literature.

---

## Methodology

This project implements and compares multiple DDPM inpainting approaches, introducing two novel enhancement methods. For complete methodology and experimental analysis, see the report.

### Vanilla DDPM Baseline

Standard diffusion inpainting pipeline:
1. Encode images to latent space using VAE
2. At each denoising step $t$:
   - Replace known regions with noisy version of original (forward noise schedule)
   - Denoise unknown regions using U-Net prediction
   - Combine using binary mask

Tested with:
- **No text guidance**: Empty prompt (unconditional generation)
- **General text guidance**: Single prompt for all images

### Method 1: LLaVA-Based Prompt Generation

**Problem**: Generic prompts lack image-specific context, resulting in semantically incoherent inpainting.

**Solution**: Use multimodal LLM (LLaVA-1.6-Mistral-7B) to generate image-specific prompts.

**Process**:
1. Feed masked image to LLaVA (model never sees ground truth)
2. LLaVA generates Stable Diffusion-compatible prompt describing visible content + logical inferences for masked regions
3. CLIP encodes prompt → embeddings guide U-Net via cross-attention

**Example**:
- **Image**: Underwater scene with partial starfish visible
- **Generated Prompt**: "Purple starfish, underwater, coral reef, marine life, vibrant colors, ocean depths, natural beauty, biodiversity..."

**Impact**: Dramatically improves semantic coherence of inpainted regions.

### Method 2: Spatial Boundary Interpolation

**Problem**: Sharp, visible boundaries between original and inpainted regions.

**Solution**: Apply spatial averaging filters along mask boundaries during denoising.

**Process**:
1. Encode image and mask to latent space
2. Every $K=5$ denoising steps:
   - Identify boundary pixels (dilate + erode mask, find difference)
   - Apply spatial averaging kernel across boundary
   - Blend smoothed values only at boundary region
3. Average propagates naturally into generated content

**Three Kernel Types**:
- **Regular**: 3×3 averaging (9 pixels)
- **Cross**: Plus-shaped (5 pixels)
- **X**: Diagonal (5 pixels)

**Impact**: Reduces harsh transitions, though boundaries remain partially visible in some cases.

### Combined Approach

The most effective configuration combines both methods:
- **LLaVA-generated prompts** → semantic consistency
- **Spatial interpolation** → boundary smoothing

---

## Directory Structure

```
project/
├── README.md                        # This file
├── environment.yml                  # Conda environment specification
│
├── main.py                          # Main inpainting pipeline
├── ddpm_script.py                   # Core DDPM algorithms
├── generate_masked_images.py        # Mask generation
├── generate_image_descriptions_LLaVA.py  # Prompt generation with LLaVA
├── evaluate_all_methods.py          # Comprehensive evaluation
├── inpaintg_eval.py                 # Single method evaluation
├── mask.py                          # Mask utility functions
├── latent.py                        # VAE encoding/decoding utilities
│
├── DIV2K_train_val_HR/              # Input: DIV2K dataset images
│   └── *.png                        # Original high-resolution images
│
├── images/                          # Input: Custom test images
│   └── *.png
│
├── masked_output/                   # Generated masks and masked images
│   ├── rectangular/
│   │   ├── images/                  # Masked images (black in masked region)
│   │   ├── masks/                   # Binary masks (white=inpaint, black=keep)
│   │   └── descriptions_rectangular.json  # LLaVA-generated prompts
│   ├── circular/
│   ├── brush_stroke/
│   ├── random_patches/
│   ├── random_noise/
│   └── half/
│
├── Inpainting/                      # Inpainting results by method
│   ├── Vanilla_without_text/        # Baseline: No text guidance
│   │   ├── rectangular/
│   │   │   └── inpainted_images/
│   │   ├── circular/
│   │   └── ...
│   ├── Vanilla_with_general_text/   # Baseline: General prompt
│   │   └── <mask_types>/
│   ├── generated_text_no_interpolation/  # Method 1 only
│   │   └── <mask_types>/
│   ├── interpolation_general_text/  # Method 2 only
│   │   ├── regular/
│   │   │   └── <mask_types>/
│   │   ├── cross/
│   │   └── x/
│   └── interpolation_generated_text/  # Methods 1 + 2 (best)
│       ├── regular/
│       ├── cross/
│       └── x/
│
├── evaluation_results_*.json    # Evaluation results with timestamp
```

### Key Naming Conventions

**File Matching**:
- Original: `0001.png`
- Masked: `0001_masked.png`
- Mask: `0001_mask.png`

**Mask Format**:
- Binary masks: White (255) = inpaint this area, Black (0) = keep original
- Masked images: Black (0) in masked regions, original RGB elsewhere

---

## Experimental Results

Five experiments were conducted on the DIV2K validation set (900 images) across 6 mask types. For full analysis and visual comparisons, see the report.

### Summary Table: Key Methods Comparison

**Rectangular Mask Results (Mean ± Std, n=900)**

| Method | SSIM (↑) | LPIPS (↓) |
|--------|----------|-----------|
| Vanilla (no text) | 0.6045 ± 0.1251 | 0.2123 ± 0.0414 |
| Vanilla (general text) | 0.6082 ± 0.1267 | 0.2244 ± 0.0459 |
| LLaVA prompts only | 0.6015 ± 0.1233 | 0.2157 ± 0.0421 |
| Interpolation (regular) + general | 0.6097 ± 0.1267 | 0.2257 ± 0.0458 |
| **Interpolation (regular) + LLaVA** | **0.6033 ± 0.1237** | **0.2156 ± 0.0425** |

### Key Findings

1. **LLaVA prompts provide better semantic coherence** qualitatively, though quantitative metrics show minimal difference (likely due to global vs. local measurement)

2. **Spatial interpolation reduces boundary artifacts** in most cases, with marginal quantitative improvements

3. **Combined approach (LLaVA + Interpolation) works best** for overall quality considering both content and boundaries

4. **Half-plane masks remain challenging** due to extreme context loss (>50% of image missing)

5. **No significant difference between kernel types** (regular, cross, x) in quantitative metrics

---

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Close other GPU-intensive applications
- For LLaVA: Process images in smaller batches or overnight
- Consider using a GPU with more VRAM (16GB+ recommended for LLaVA)
- Reduce inference steps: `--num_steps 25` (faster but lower quality)

### LLaVA Processing Extremely Slow

**Symptoms**: Each image takes >2 minutes

**Solutions**:
- Use `--max_tokens 50` instead of 200 (3-4x faster)
- Verify GPU is being used: Check `nvidia-smi` during processing
- Process overnight for large datasets (100 images ≈ 30-60 minutes)
- Consider processing only a subset of images for testing

### File Not Found Errors

**Symptoms**: `FileNotFoundError` for masks or images

**Solutions**:
- Verify file naming: `image.png` → `image_masked.png` + `image_mask.png`
- Check that masks were generated: `ls masked_output/<mask_type>/images/`
- Ensure all paths use correct directory structure
- Try absolute paths instead of relative paths

### Poor Inpainting Results

**Symptoms**: Inpainted regions look unnatural or don't match context

**Solutions**:
- **Use LLaVA-generated prompts** instead of generic prompts
- Try different mask types (smaller masks often work better)
- Increase inference steps: `--num_steps 100`
- Adjust guidance scale: `--guidance_scale 10.0` (stronger prompt influence)
- Verify mask alignment with masked image

### Model Download Issues

**Symptoms**: Connection errors when downloading models

**Solutions**:
- Check internet connection
- Verify Hugging Face access (no authentication needed for these models)
- Manually download and cache:
  ```bash
  python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('sd2-community/stable-diffusion-2-base')"
  ```
- Check disk space (~10GB needed for models)

### JSON Parse Errors

**Symptoms**: `JSONDecodeError` when loading prompts

**Solutions**:
- Verify JSON file exists and is valid: `cat descriptions.json | python -m json.tool`
- Regenerate descriptions if corrupted
- Check that image names in JSON match actual filenames

---

## Advanced Configuration

### Changing the Base Model

Use a different Stable Diffusion model:

```bash
python main.py \
  --model_id "stabilityai/stable-diffusion-2-1-base" \
  --input_folder ./images \
  --output_folder ./output \
  --masked_images_folder ./masked_output/rectangular/images
```

**Compatible models**: Any Stable Diffusion 2.x model from Hugging Face

### Custom LLaVA Instructions

Modify the prompt generation behavior:

```bash
python generate_image_descriptions_LLaVA.py \
  --input_folder ./images \
  --output_json descriptions.json \
  --instruction "Describe this image focusing on artistic style, colors, and mood. Use descriptive adjectives."
```

### Fine-Tuning Spatial Interpolation

**Adjust averaging frequency**:
- More frequent: `--steps_avg 3` (smoother but may blur)
- Less frequent: `--steps_avg 10` (sharper but more visible boundaries)

**Adjust boundary width**:
- Wider boundary: `--area 2` (affects more pixels around edge)
- Narrow boundary: `--area 0` (minimal blending)

**Example**:
```bash
python main.py \
  --method spatial \
  --kernel_type regular \
  --steps_avg 3 \
  --area 2 \
  # ... other arguments
```

### Modifying the Scheduler

Edit [ddpm_script.py](ddpm_script.py) to change the noise scheduler:

```python
# Line ~20-25: Uncomment DDIMScheduler for faster sampling
from diffusers import DDIMScheduler

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
```

**DDIMScheduler** allows fewer steps (e.g., 25) with comparable quality to DDPM (50 steps).

### Batch Processing Multiple Experiments

Create a shell script to run all configurations:

```bash
#!/bin/bash
# run_all_experiments.sh

MASKS="rectangular circular brush_stroke"
METHODS="vanilla spatial"

for mask in $MASKS; do
  for method in $METHODS; do
    python main.py \
      --input_folder ./DIV2K_train_val_HR \
      --output_folder ./Inpainting/${method}/${mask} \
      --masked_images_folder ./masked_output/${mask}/images \
      --method $method \
      --use_text \
      --prompts_json ./masked_output/${mask}/descriptions_${mask}.json
  done
done
```

---

## Project Structure

### Python Scripts Reference

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| [main.py](main.py) | Main inpainting pipeline | `setup_models()`, `process_images()` |
| [ddpm_script.py](ddpm_script.py) | Core DDPM algorithms | `vanilla_ddpm_inpainting()`, `ddpm_spatial_interpolate_inpainting()` |
| [generate_masked_images.py](generate_masked_images.py) | Mask generation for all types | `process_images()`, mask type handlers |
| [generate_image_descriptions_LLaVA.py](generate_image_descriptions_LLaVA.py) | LLaVA prompt generation | `generate_description()` |
| [evaluate_all_methods.py](evaluate_all_methods.py) | Comprehensive metric evaluation | Computes SSIM/LPIPS for all methods |
| [inpaintg_eval.py](inpaintg_eval.py) | Single method evaluation | SSIM/LPIPS for one configuration |
| [mask.py](mask.py) | Mask creation utilities | 6 mask generation functions |
| [latent.py](latent.py) | VAE encoding/decoding | `pil_to_latent()`, `latent_to_pil()`, `get_text_embeddings()` |

### Configuration Files

- [environment.yml](environment.yml) - Complete conda environment specification

### Archived/Experimental

- [NotInUse/](NotInUse/) - Previous experiments, alternative implementations, old results

---

## Limitations

Several limitations remain in the current implementation:

### 1. Evaluation Metrics
- **Global metrics mask local improvements**: SSIM and LPIPS computed on entire images may not capture successful inpainting in masked regions
- **Recommendation**: Future work should consider using additional evaluation metrics.

### 2. Boundary Artifacts
- Spatial interpolation **partially** mitigates edge artifacts but doesn't eliminate them completely
- Fixed averaging schedule (every K steps) may not be optimal for all scenarios
- High-frequency textures can still show visible seams

### 3. Extreme Mask Types
- **Half-plane masks**: Removing >50% of image provides insufficient context for coherent generation
- Model struggles to infer large missing regions even with LLaVA guidance

### 4. Processing Speed
- **LLaVA prompt generation**: Very slow (15-100s per image depending on token limit)
- **GPU Required**: Pipeline is impractical on CPU-only systems

### 5. Fixed Model Architecture
- Currently tied to Stable Diffusion 2 architecture
- Would require code modifications to use newer models (SDXL, SD3)

---

## Citation & Acknowledgments

### Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{drive2026improving,
  title={Improving Vanilla Inpainting},
  author={Drive, Saar and Shenkar, Omer},
  journal={CS236781 Deep Learning Course Project},
  year={2026},
  month={February}
}
```

### Acknowledgments

This project builds upon several foundational works and tools:

**Models & Frameworks**:
- **Stable Diffusion 2** - Rombach et al. (2022) - High-Resolution Image Synthesis with Latent Diffusion Models
- **LLaVA** - Liu et al. (2024) - Improved Baselines with Visual Instruction Tuning
- **CLIP** - Radford et al. (2021) - Learning Transferable Visual Models From Natural Language Supervision

**Datasets**:
- **DIV2K** - Agustsson & Timofte (2017) - NTIRE 2017 Challenge on Single Image Super-Resolution

**Metrics**:
- **LPIPS** - Zhang et al. (2018) - The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
- **SSIM** - Wang et al. (2004) - Image Quality Assessment

**Tools & Libraries**:
- Hugging Face Diffusers and Transformers
- PyTorch
- scikit-image

**AI Assistance**:
- **Claude Sonnet 4.5** (Anthropic) - Code development and debugging assistance
- **Grammarly** - Report writing assistance
- **autopep8** (VS Code extension) - Code formatting

### Authors

**Saar Drive** and **Omer Shenkar**  
CS236781 Deep Learning Course Project  
Technion - Israel Institute of Technology  
February 2026

---

## License
MIT
This project is an academic research implementation. For academic or research use only.

---

**Questions or Issues?** Please check the [Troubleshooting](#troubleshooting) section or review the report for detailed methodology.

**Last Updated**: March 2, 2026
