"""
Comprehensive Inpainting Evaluation Script

Evaluates all inpainting methods across all mask types.
Computes mean and standard deviation of SSIM and LPIPS metrics for each method-mask combination.
"""

import os
import argparse
import json
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import lpips
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
from pathlib import Path


def load_image_for_metrics(path, resize_to=(256, 256)):
    """
    Load and preprocess image for metric computation.

    Args:
        path: Path to the image file
        resize_to: Tuple (height, width) to resize images to.
                   Set to None to keep original size.

    Returns:
        np_img: Numpy array (H, W, C) in range [0, 255]
        torch_tensor: PyTorch tensor (1, 3, H, W) in range [-1, 1] for LPIPS
    """
    pil_img = Image.open(path).convert('RGB')

    if resize_to is not None:
        pil_img = pil_img.resize((resize_to[1], resize_to[0]), Image.BILINEAR)

    np_img = np.array(pil_img) # SSIM needs npy

    # lpips 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    torch_tensor = transform(pil_img).unsqueeze(0)

    return np_img, torch_tensor


def compute_ssim(img1_np, img2_np):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Range [0, 1], higher is better.
    """
    return ssim(img1_np, img2_np, channel_axis=2, data_range=255)


def compute_lpips(img1_tensor, img2_tensor, loss_fn, device):
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS) between two images.
    Range [0, 1+], lower is better.
    """
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    with torch.no_grad():
        distance = loss_fn(img1_tensor, img2_tensor)

    return distance.item()


def find_image_pairs(input_dir, output_dir, valid_extensions=None):
    """
    Find matching pairs of input and output images.
    Input image X.png corresponds to output image X_inpainted.png
    """
    if valid_extensions is None:
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    pairs = []

    input_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    for input_file in sorted(input_files):
        name, ext = os.path.splitext(input_file)
        if ext.lower() not in valid_extensions:
            continue

        output_file = f"{name}_inpainted{ext}"
        output_path = os.path.join(output_dir, output_file)

        if os.path.exists(output_path):
            pairs.append((input_file, output_file, name))

    return pairs


def evaluate_method_mask(input_dir, output_dir, loss_fn, device, resize_to=(256, 256), match_original_size=False):
    """
    Evaluate a single method-mask combination.

    Args:
        match_original_size: If True, resize inpainted output to match original input size

    Returns:
        dict with 'ssim_mean', 'ssim_std', 'lpips_mean', 'lpips_std', 'num_images'
    """
    pairs = find_image_pairs(input_dir, output_dir)

    if len(pairs) == 0:
        return None

    ssim_values = []
    lpips_values = []

    for input_file, output_file, base_name in pairs:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)

        try:
            if match_original_size:
                # Load original at native size, resize inpainted to match
                input_np, input_tensor = load_image_for_metrics(
                    input_path, resize_to=None)
                original_size = (
                    input_np.shape[0], input_np.shape[1])  # (H, W)
                output_np, output_tensor = load_image_for_metrics(
                    output_path, resize_to=original_size)
            else:
                # Load both with specified resize
                input_np, input_tensor = load_image_for_metrics(
                    input_path, resize_to)
                output_np, output_tensor = load_image_for_metrics(
                    output_path, resize_to)

            # Compute metrics
            ssim_value = compute_ssim(input_np, output_np)
            lpips_value = compute_lpips(
                input_tensor, output_tensor, loss_fn, device)

            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)
        except Exception as e:
            print(f"    Warning: Error processing {input_file}: {e}")
            continue

    if len(ssim_values) == 0:
        return None

    return {
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'lpips_mean': float(np.mean(lpips_values)),
        'lpips_std': float(np.std(lpips_values)),
        'num_images': len(ssim_values)
    }


def evaluate_all_methods(base_dir, input_dir, resize_to=(256, 256), match_original_size=False):
    """
    Evaluate all inpainting methods across all mask types.

    Args:
        base_dir: Base directory containing Inpainting folder
        input_dir: Directory containing original input images
        resize_to: Size to resize images to for consistent comparison
        match_original_size: If True, resize inpainted to match original input size

    Returns:
        results: Nested dictionary with results
    """
    print("Initializing LPIPS model...")
    loss_fn = lpips.LPIPS(net='alex')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)
    print(f"Using device: {device}\n")

    methods = {
        'vanilla_no_text': {
            'name': 'Vanilla (No Text)',
            'base_path': 'Inpainting/Vanilla_without_text'
        },
        'vanilla_general_text': {
            'name': 'Vanilla (General Text)',
            'base_path': 'Inpainting/Vanilla_with_general_text'
        },
        'vanilla_generated_text': {
            'name': 'Vanilla (Generated Text)',
            'base_path': 'Inpainting/generated_text_no_interpolation'
        },
        'spatial_regular_general': {
            'name': 'Spatial Regular (General Text)',
            'base_path': 'Inpainting/interpolation_general_text/regular'
        },
        'spatial_regular_generated': {
            'name': 'Spatial Regular (Generated Text)',
            'base_path': 'Inpainting/interpolation_generated_text/regular'
        },
        'spatial_cross_general': {
            'name': 'Spatial Cross (General Text)',
            'base_path': 'Inpainting/interpolation_general_text/cross'
        },
        'spatial_cross_generated': {
            'name': 'Spatial Cross (Generated Text)',
            'base_path': 'Inpainting/interpolation_generated_text/cross'
        },
        'spatial_x_general': {
            'name': 'Spatial X (General Text)',
            'base_path': 'Inpainting/interpolation_general_text/x'
        },
        'spatial_x_generated': {
            'name': 'Spatial X (Generated Text)',
            'base_path': 'Inpainting/interpolation_generated_text/x'
        }
    }

    mask_types = ['rectangular', 'circular', 'random_patches',
                  'random_noise', 'brush_stroke', 'half']

    results = {}

    print("=" * 100)
    print("EVALUATING ALL METHODS")
    print("=" * 100)

    for method_key, method_info in methods.items():
        print(f"\n### {method_info['name']} ###")
        results[method_key] = {
            'name': method_info['name'],
            'masks': {}
        }

        for mask_type in mask_types:
            output_dir = os.path.join(
                base_dir, method_info['base_path'], mask_type, 'inpainted_images')

            if not os.path.exists(output_dir):
                print(
                    f"  {mask_type:20} - SKIPPED (directory not found: {output_dir})")
                continue

            result = evaluate_method_mask(
                input_dir, output_dir, loss_fn, device, resize_to, match_original_size)

            if result is None:
                print(f"  {mask_type:20} - SKIPPED (no image pairs found)")
                continue

            results[method_key]['masks'][mask_type] = result

            print(f"  {mask_type:20} - "
                  f"SSIM: {result['ssim_mean']:.4f} ± {result['ssim_std']:.4f}, "
                  f"LPIPS: {result['lpips_mean']:.4f} ± {result['lpips_std']:.4f} "
                  f"({result['num_images']} images)")

    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)

    return results


def print_summary_table(results):
    """Print a nicely formatted summary table."""
    print("\n" + "=" * 120)
    print("SUMMARY TABLE - MEAN ± STD")
    print("=" * 120)

    # Get all mask types from results
    mask_types = set()
    for method_data in results.values():
        mask_types.update(method_data['masks'].keys())
    mask_types = sorted(mask_types)

    for mask_type in mask_types:
        print(f"\n### {mask_type.upper()} ###")
        print(f"{'Method':<40} {'SSIM':^25} {'LPIPS':^25} {'N Images':>10}")
        print("-" * 120)

        for method_key, method_data in results.items():
            if mask_type in method_data['masks']:
                m = method_data['masks'][mask_type]
                print(f"{method_data['name']:<40} "
                      f"{m['ssim_mean']:>6.4f} ± {m['ssim_std']:<6.4f}       "
                      f"{m['lpips_mean']:>6.4f} ± {m['lpips_std']:<6.4f}       "
                      f"{m['num_images']:>10}")
            else:
                print(
                    f"{method_data['name']:<40} {'N/A':^25} {'N/A':^25} {'N/A':>10}")

    print("\n" + "=" * 120)


def save_results(results, base_dir):
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_all_methods_{timestamp}.json"
    output_path = os.path.join(base_dir, output_filename)

    # Prepare output data
    output_data = {
        "metadata": {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "evaluation_date": datetime.now().strftime('%Y-%m-%d'),
        },
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all inpainting methods across all mask types'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/home/saar.drive/projects/DeepLearning/cs236781-HomeWork/project/project',
        help='Base directory containing Inpainting folder'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='DIV2K_train_val_HR',
        help='Directory containing original input images (relative to base_dir or absolute)'
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        default=[512, 504],
        help='Resize images to [height, width] for evaluation (default: 512 504)'
    )
    parser.add_argument(
        '--no_resize',
        action='store_true',
        help='Do not resize images (use original sizes)'
    )
    parser.add_argument(
        '--match_original_size',
        action='store_true',
        help='Resize inpainted output to match original input size (evaluate at native resolution)'
    )

    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.abspath(args.base_dir)

    # Check if input_dir is absolute or relative
    if os.path.isabs(args.input_dir):
        input_dir = args.input_dir
    else:
        input_dir = os.path.join(base_dir, args.input_dir)

    # Handle resize options
    match_original_size = args.match_original_size
    if match_original_size:
        resize_to = None  # Will be determined per-image
        resize_msg = "inpainted resized to match original"
    else:
        resize_to = None if args.no_resize else tuple(args.resize)
        resize_msg = str(resize_to) if resize_to else "no resizing"

    print(f"Base directory: {base_dir}")
    print(f"Input directory: {input_dir}")
    print(f"Resize mode: {resize_msg}\n")

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    results = evaluate_all_methods(
        base_dir, input_dir, resize_to, match_original_size)

    print_summary_table(results)

    save_results(results, base_dir)


if __name__ == '__main__':
    main()
