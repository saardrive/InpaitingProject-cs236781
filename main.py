"""
Main script for batch image inpainting with vanilla DDPM using pre-generated masks.

Requires masks to be pre-generated using generate_masked_images.py

Usage:
    python main.py --input_folder ./images --output_folder ./output --masked_images_folder ./masked_output/rectangular/images
    
    With JSON prompts:
    python main.py --input_folder ./images --output_folder ./output --masked_images_folder ./masked_output/rectangular/images --prompts_json descriptions.json
"""
import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from latent import pil_to_latent, mask_to_latent, latent_to_pil, get_text_embeddings
from ddpm_script import vanilla_ddpm_inpainting, ddpm_spatial_interpolate_inpainting


def setup_models(model_id, device, dtype):
    """Load all necessary models."""
    print("Loading models...")

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("Models loaded successfully!")
    return vae, unet, tokenizer, text_encoder, scheduler


def process_images(args):
    """Main processing function."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    vae, unet, tokenizer, text_encoder, scheduler = setup_models(
        args.model_id, device, dtype
    )

    output_folder = Path(args.output_folder)
    inpainted_folder = output_folder / "inpainted_images"
    mask_latent_folder = output_folder / "mask_latents"

    for folder in [inpainted_folder, mask_latent_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    prompts_dict = {}
    if args.prompts_json:
        print(f"Loading prompts from: {args.prompts_json}")
        with open(args.prompts_json, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        # Create a dictionary mapping image paths/names to prompts
        for item in prompts_data:
            image_path = Path(item['image_path'])
            prompts_dict[image_path.name] = item['prompt']
            prompts_dict[str(image_path)] = item['prompt']
            prompts_dict[str(image_path.absolute())] = item['prompt']

            # Also add entry without "_masked" suffix to match original filenames
            if '_masked' in image_path.name:
                original_name = image_path.name.replace('_masked', '')
                prompts_dict[original_name] = item['prompt']
        print(f"Loaded {len(prompts_data)} prompts from JSON")

    # Get text embeddings if using text conditioning with a single prompt
    text_embeddings = None
    if args.use_text and args.prompt and not args.prompts_json:
        print(f"Using single prompt: '{args.prompt}'")
        text_embeddings = get_text_embeddings(
            args.prompt, tokenizer, text_encoder, device, args.negative_prompt
        )

    input_folder = Path(args.input_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in input_folder.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            base_name = img_path.stem

            masked_images_folder = Path(args.masked_images_folder)
            masks_folder = masked_images_folder.parent / "masks"

            mask_path = None
            masked_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = masks_folder / f"{base_name}_mask{ext}"
                potential_masked = masked_images_folder / \
                    f"{base_name}_masked{ext}"
                if potential_mask.exists() and potential_masked.exists():
                    mask_path = potential_mask
                    masked_path = potential_masked
                    break

            if not mask_path or not masked_path:
                print(
                    f"Warning: Pre-generated mask/masked image not found for {img_path.name} in {masked_images_folder} and {masks_folder}, skipping...")
                continue

            mask_image = Image.open(mask_path).convert('L')
            masked_img_pil = Image.open(masked_path).convert('RGB')

            # Convert to latent space (use masked image, not original)
            masked_latent = pil_to_latent(masked_img_pil, vae, device, dtype)

            # Create mask in latent space
            mask_latent = mask_to_latent(
                mask_image, masked_latent.shape, device, dtype)

            # Get image-specific prompt if using JSON
            current_text_embeddings = text_embeddings
            if args.prompts_json and prompts_dict:
                img_prompt = None
                for key in [img_path.name, str(img_path), str(img_path.absolute())]:
                    if key in prompts_dict:
                        img_prompt = prompts_dict[key]
                        break

                if img_prompt:
                    current_text_embeddings = get_text_embeddings(
                        img_prompt, tokenizer, text_encoder, device, args.negative_prompt
                    )
                else:
                    print(
                        f"Warning: No prompt found for {img_path.name}, skipping text conditioning")

            # If not using text conditioning, create empty embeddings
            if current_text_embeddings is None:
                current_text_embeddings = get_text_embeddings(
                    "", tokenizer, text_encoder, device)

            # Perform inpainting
            if args.method == "spatial":
                inpainted_latent = ddpm_spatial_interpolate_inpainting(
                    masked_latent,
                    mask_latent,
                    current_text_embeddings,
                    unet,
                    scheduler,
                    device,
                    dtype,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    steps_avg=args.steps_avg,
                    area=args.area,
                    kernel_type=args.kernel_type,
                )
            else:
                inpainted_latent = vanilla_ddpm_inpainting(
                    masked_latent,
                    mask_latent,
                    current_text_embeddings,
                    unet,
                    scheduler,
                    device,
                    dtype,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                )

            inpainted_image = latent_to_pil(inpainted_latent, vae)

            inpainted_image.save(inpainted_folder /
                                 f"{base_name}_inpainted.png")

            mask_latent_np = mask_latent.cpu().numpy()
            np.save(mask_latent_folder /
                    f"{base_name}_mask_latent.npy", mask_latent_np)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"\nProcessing complete! Results saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch image inpainting with vanilla DDPM using pre-generated masks"
    )

    # Input/Output
    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="Folder containing input images"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="Folder to save output images"
    )
    parser.add_argument(
        "--masked_images_folder", type=str, required=True,
        help="Folder containing pre-generated masked images (e.g., ./masked_output/rectangular/images)"
    )

    # Model settings
    parser.add_argument(
        "--model_id", type=str, default="sd2-community/stable-diffusion-2-base",
        help="Stable Diffusion model ID"
    )

    # Inpainting settings
    parser.add_argument(
        "--method", type=str, default="vanilla", choices=["vanilla", "spatial"],
        help="Inpainting method: 'vanilla' for standard DDPM, 'spatial' for spatial interpolate"
    )
    parser.add_argument(
        "--use_text", action="store_true",
        help="Use text-conditioned inpainting"
    )
    parser.add_argument(
        "--steps_avg", type=int, default=5,
        help="How often to apply spatial averaging (every N steps) - only for spatial method"
    )
    parser.add_argument(
        "--area", type=int, default=1,
        help="Padding for averaging window (1 creates a 3x3 window) - only for spatial method"
    )
    parser.add_argument(
        "--kernel_type", type=str, default="regular", choices=["regular", "cross", "x"],
        help="Kernel type for spatial averaging: 'regular' (3x3, 9 pixels), 'cross' (5 pixels in + shape), 'x' (5 pixels diagonal)"
    )
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Text prompt for inpainting (if using text conditioning and not using JSON)"
    )
    parser.add_argument(
        "--prompts_json", type=str, default=None,
        help="JSON file with image-specific prompts (generated by generate_image_descriptions.py)"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="",
        help="Negative prompt for inpainting"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale (only for text-conditioned)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    process_images(args)


if __name__ == "__main__":
    main()
