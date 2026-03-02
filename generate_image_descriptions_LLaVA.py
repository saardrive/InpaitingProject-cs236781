"""
Script to generate description prompts for images using LLaVA model.

Usage:
    python generate_image_descriptions_LLaVA.py --input_folder ./images --output_json descriptions.json
"""
import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm


def load_llava_model(model_id="llava-hf/llava-v1.6-mistral-7b-hf"):
    """
    Load LLaVA model and processor.

    Args:
        model_id: Hugging Face model ID for LLaVA

    Returns:
        processor, model
    """
    print(f"Loading LLaVA model: {model_id}...")
    print("This may take a few minutes on first run...")

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model loaded successfully!")
    return processor, model


def generate_description(processor, model, image, instruction_text, max_new_tokens=200):
    """
    Generate a description prompt for a single image using LLaVA.

    Args:
        processor: LLaVA processor
        model: LLaVA model
        image: PIL Image
        instruction_text: Text instruction for generating the description
        max_new_tokens: Maximum number of tokens to generate (default: 200)

    Returns:
        Generated description string (extracted after [/INST] tag)
    """
    # Conversation format for LLaVA  
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to(
        "cuda", torch.float16)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full_response = processor.decode(output[0], skip_special_tokens=True)

    # Extract only the generated description after [/INST]
    # LLaVA outputs in format: [INST] ... [/INST] actual_description
    if "[/INST]" in full_response:
        description = full_response.split("[/INST]", 1)[1].strip()
    else:
        # Fallback if format is different
        description = full_response.strip()

    return description


def process_folder(args):
    """Process all images in a folder and generate descriptions."""

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. LLaVA requires GPU to run efficiently.")
        print("This will likely fail or be extremely slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    processor, model = load_llava_model(args.model_id)

    instruction_text = args.instruction or (
        "You are a prompt generation machine for Stable Diffusion. "
        "Look at this image. Your goal is to generate a text prompt that "
        "describes the image and provides logical details to fill in the missing "
        "areas (inpainting).\n\n"
        "Rules:\n"
        "1. Output ONLY the raw prompt text.\n"
        "2. Use comma-separated keywords and phrases.\n"
        "3. Do NOT use bullet points, conversational filler, or introductions like 'Here is the prompt'.\n"
        "4. Focus on: Subject description, Lighting, Background scenery, Art style, and Vibe.\n\n"
    )

    # Get all image files
    input_folder = Path(args.input_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Using instruction: {instruction_text[:100]}...")
    print(f"Max tokens per description: {args.max_tokens}")
    print("\nNote: First image may take 30-60 seconds. Please be patient...\n")

    # Process images and results
    results = []
    import time
    for idx, img_path in enumerate(tqdm(image_files, desc="Generating descriptions")):
        try:
            start_time = time.time()

            image = Image.open(img_path).convert("RGB")

            description = generate_description(
                processor, model, image, instruction_text, args.max_tokens)

            elapsed = time.time() - start_time

            results.append({
                "image_path": str(img_path.absolute()),
                "image_name": img_path.name,
                "prompt": description
            })

            if args.verbose or idx < 3:
                print(
                    f"\n[{idx+1}/{len(image_files)}] {img_path.name} ({elapsed:.1f}s):")
                print(f"  {description}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    output_path = Path(args.output_json)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete! Descriptions saved to: {output_path}")
    print(f"Successfully processed {len(results)}/{len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Generate description prompts for images using LLaVA model"
    )

    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="Folder containing input images"
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Output JSON file path"
    )

    parser.add_argument(
        "--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="LLaVA model ID from Hugging Face (default: llava-v1.6-mistral-7b-hf)"
    )
    parser.add_argument(
        "--instruction", type=str, default=None,
        help="Custom instruction text for generating descriptions (optional)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200,
        help="Maximum tokens to generate per image (default: 200, use 50 for faster results)"
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Print descriptions as they are generated"
    )

    args = parser.parse_args()

    process_folder(args)


if __name__ == "__main__":
    main()
