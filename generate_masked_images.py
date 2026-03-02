"""
Script to generate masked images using various mask types.

This script applies different types of masks to input images and saves the results.
"""
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
from mask import create_mask

MASK_TYPES = ['rectangular', 'circular', 'random_patches',
              'random_noise', 'brush_stroke', 'half']


def parse_mask_options(options_str):
    """
    Parse mask options from a string.
    Format: key1=value1,key2=value2

    Example: "mask_fraction=0.4,num_strokes=10"
    """
    if not options_str:
        return {}

    options = {}
    for pair in options_str.split(','):
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        try:
            # Check if it's a tuple
            if value.startswith('(') and value.endswith(')'):
                tuple_values = value[1:-1].split(',')
                options[key] = tuple(int(v.strip()) for v in tuple_values)
            
            elif '.' not in value:
                options[key] = int(value)
            
            else:
                options[key] = float(value)
        except ValueError:
            # Keep as string
            options[key] = value

    return options


def apply_mask_to_image(image, mask):
    """
    Apply mask to image. Masked regions are set to black (or you can customize).

    Args:
        image: PIL Image
        mask: PIL Image (white=inpaint, black=keep)

    Returns:
        Masked PIL Image
    """
    img_array = np.array(image)
    mask_array = np.array(mask)

    # Create masked image (set masked regions to black)
    # mask_array is 255 where we want to inpaint, 0 where we keep
    # We want to set inpaint regions to black
    mask_3channel = np.stack(
        [mask_array] * 3, axis=-1) if len(img_array.shape) == 3 else mask_array

    # Invert mask: where mask=255 (inpaint), we set to 0 (black)
    masked_img_array = img_array.copy()
    masked_img_array[mask_3channel == 255] = 0

    return Image.fromarray(masked_img_array)


def process_images(input_folder, mask_types, output_folder, mask_options):
    """
    Process all images in the input folder with specified mask types.

    Args:
        input_folder: Path to input images
        mask_types: List of mask types to apply (or ['all'])
        output_folder: Base output folder
        mask_options: Dictionary of options for each mask type
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process")

    # Determine which mask types to use
    if 'all' in mask_types:
        types_to_process = MASK_TYPES
    else:
        types_to_process = mask_types

    for mask_type in types_to_process:
        if mask_type not in MASK_TYPES:
            print(f"Warning: Unknown mask type '{mask_type}', skipping...")
            continue

        print(f"\nProcessing with mask type: {mask_type}")

        # Create output folders for this mask type
        output_path = Path(output_folder) / mask_type
        images_folder = output_path / "images"
        masks_folder = output_path / "masks"
        images_folder.mkdir(parents=True, exist_ok=True)
        masks_folder.mkdir(parents=True, exist_ok=True)

        options = mask_options.get(mask_type, {})

        # Process each image
        for img_file in image_files:
            try:
                image = Image.open(img_file).convert('RGB')
                width, height = image.size

                mask = create_mask(width, height, mask_type, **options)

                masked_image = apply_mask_to_image(image, mask)

                output_file = images_folder / \
                    f"{img_file.stem}_masked{img_file.suffix}"
                masked_image.save(output_file)

                mask_file = masks_folder / \
                    f"{img_file.stem}_mask{img_file.suffix}"
                mask.save(mask_file)

                print(f"  Processed: {img_file.name} -> {output_file.name}")

            except Exception as e:
                print(f"  Error processing {img_file.name}: {e}")
                continue

    print(f"\nDone! Results saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate masked images using various mask types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available mask types: {', '.join(MASK_TYPES)}, all

Mask options by type:
  rectangular:     mask_fraction (default: 0.35)
  circular:        mask_fraction (default: 0.35)
  random_patches:  num_patches (default: 10), patch_size_range (default: (20, 80))
  random_noise:    noise_fraction (default: 0.3)
  brush_stroke:    num_strokes (default: 5), stroke_width_range (default: (10, 30))
  half:            direction (default: 'left', options: 'left', 'right', 'top', 'bottom')

Examples:
  # Generate rectangular masks with default options
  python generate_masked_images.py images/ rectangular
  
  # Generate all mask types
  python generate_masked_images.py images/ all -o results/
  
  # Generate circular mask with custom fraction
  python generate_masked_images.py images/ circular --rectangular "mask_fraction=0.5"
  
  # Generate multiple mask types with custom options
  python generate_masked_images.py images/ rectangular circular -o output/ \\
      --rectangular "mask_fraction=0.4" --circular "mask_fraction=0.3"
  
  # Generate brush strokes with custom parameters
  python generate_masked_images.py images/ brush_stroke \\
      --brush_stroke "num_strokes=10,stroke_width_range=(15,40)"
        """
    )

    parser.add_argument('input_folder', type=str,
                        help='Folder containing input images')
    parser.add_argument('mask_types', nargs='+', type=str,
                        help=f'Mask type(s) to create: {", ".join(MASK_TYPES)}, or "all"')
    parser.add_argument('-o', '--output', type=str, default='masked_output',
                        help='Output folder (default: masked_output)')

    # Add options for each mask type
    parser.add_argument('--rectangular', type=str, default='',
                        help='Options for rectangular mask (e.g., "mask_fraction=0.4")')
    parser.add_argument('--circular', type=str, default='',
                        help='Options for circular mask (e.g., "mask_fraction=0.4")')
    parser.add_argument('--random_patches', type=str, default='',
                        help='Options for random_patches mask (e.g., "num_patches=15,patch_size_range=(30,90)")')
    parser.add_argument('--random_noise', type=str, default='',
                        help='Options for random_noise mask (e.g., "noise_fraction=0.4")')
    parser.add_argument('--brush_stroke', type=str, default='',
                        help='Options for brush_stroke mask (e.g., "num_strokes=10,stroke_width_range=(15,40)")')
    parser.add_argument('--half', type=str, default='',
                        help='Options for half mask (e.g., "direction=right")')

    args = parser.parse_args()

    # Parse mask options for each type
    mask_options = {
        'rectangular': parse_mask_options(args.rectangular),
        'circular': parse_mask_options(args.circular),
        'random_patches': parse_mask_options(args.random_patches),
        'random_noise': parse_mask_options(args.random_noise),
        'brush_stroke': parse_mask_options(args.brush_stroke),
        'half': parse_mask_options(args.half),
    }

    process_images(args.input_folder, args.mask_types,
                   args.output, mask_options)


if __name__ == '__main__':
    main()
