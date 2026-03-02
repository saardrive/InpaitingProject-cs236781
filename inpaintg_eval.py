"""
Image Inpainting Evaluation Script

Evaluates pairs of images from input and output folders using SSIM and LPIPS metrics.
Input image X.png corresponds to output image X_inpainted.png
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

    # Convert to numpy for SSIM
    np_img = np.array(pil_img)

    # Convert to torch tensor for LPIPS (range [-1, 1])
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

    Args:
        input_dir: Directory containing input images
        output_dir: Directory containing inpainted output images
        valid_extensions: List of valid image extensions (e.g., ['.png', '.jpg'])

    Returns:
        pairs: List of tuples (input_filename, output_filename, base_name)
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

        # Look for corresponding output file: X.ext -> X_inpainted.ext
        output_file = f"{name}_inpainted{ext}"
        output_path = os.path.join(output_dir, output_file)

        if os.path.exists(output_path):
            pairs.append((input_file, output_file, name))
        else:
            print(f"Warning: No matching output file found for {input_file}")

    return pairs


def evaluate_inpainting(input_dir, output_dir, resize_to=(256, 256)):
    """
    Evaluate all image pairs using SSIM and LPIPS metrics.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory containing inpainted output images
        resize_to: Size to resize images to for consistent comparison

    Returns:
        results: List of dictionaries containing metrics for each image pair
    """
    print("Initializing LPIPS model...")
    loss_fn = lpips.LPIPS(net='alex')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)
    print(f"Using device: {device}\n")

    pairs = find_image_pairs(input_dir, output_dir)
    print(f"Found {len(pairs)} image pairs to evaluate\n")

    if len(pairs) == 0:
        print("No image pairs found. Exiting.")
        return []

    # Evaluate each pair
    results = []
    print("=" * 80)

    for input_file, output_file, base_name in pairs:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)

        input_np, input_tensor = load_image_for_metrics(input_path, resize_to)
        output_np, output_tensor = load_image_for_metrics(
            output_path, resize_to)

        ssim_value = compute_ssim(input_np, output_np)
        lpips_value = compute_lpips(
            input_tensor, output_tensor, loss_fn, device)

        result = {
            'input': input_file,
            'output': output_file,
            'base_name': base_name,
            'ssim': float(ssim_value),
            'lpips': float(lpips_value)
        }
        results.append(result)

        print(
            f"Image: {base_name} | SSIM: {ssim_value:.4f} | LPIPS: {lpips_value:.4f}")

    # Compute summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    ssim_values = [r['ssim'] for r in results]
    lpips_values = [r['lpips'] for r in results]

    avg_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    avg_lpips = np.mean(lpips_values)
    std_lpips = np.std(lpips_values)

    print(f"SSIM:  Mean = {avg_ssim:.4f}, Std = {std_ssim:.4f}")
    print(f"LPIPS: Mean = {avg_lpips:.4f}, Std = {std_lpips:.4f}")
    print("=" * 80)

    # Save results to JSON file
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f"evaluation_results_{timestamp}.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    output_data = {
        "metadata": {
            "date": datetime.now().strftime('%m-%d %H:%M'),
            "input_directory": input_dir,
            "output_directory": output_dir,
            "resize_to": list(resize_to) if resize_to is not None else None
        },
        "individual_results": results,
        "summary_statistics": {
            "ssim": {
                "mean": float(avg_ssim),
                "std": float(std_ssim)
            },
            "lpips": {
                "mean": float(avg_lpips),
                "std": float(std_lpips)
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate inpainting results')
    parser.add_argument('--input_dir', type=str, default='/home/saardrive/projects/DeepLearning/cs236781-HomeWork/project/project/DIV2K_valid_HR',
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='/home/saardrive/projects/DeepLearning/cs236781-HomeWork/project/project/output_same_prompt/inpainted_images',
                        help='Directory containing inpainted output images')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 504],
                        help='Resize images to [height, width] (default: 512 504)')
    parser.add_argument('--no_resize', action='store_true',
                        help='Do not resize images (use original sizes)')

    args = parser.parse_args()

    resize_to = None if args.no_resize else tuple(args.resize)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resize to: {resize_to}\n")

    results = evaluate_inpainting(args.input_dir, args.output_dir, resize_to)

    return results


if __name__ == '__main__':
    main()
