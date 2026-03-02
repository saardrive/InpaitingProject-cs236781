"""
Latent space conversion functions and text embeddings.
"""
import torch
import numpy as np
from PIL import Image


def pil_to_latent(image_pil, vae, device, dtype):
    """
    Convert PIL image to latent representation using VAE encoder.

    Args:
        image_pil: PIL Image
        vae: VAE model
        device: torch device
        dtype: torch dtype

    Returns:
        Latent tensor
    """
    # Convert to tensor and normalize to [-1, 1]
    img_array = np.array(image_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(
        2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    img_tensor = (img_tensor * 2.0) - 1.0  # Normalize to [-1, 1]
    img_tensor = img_tensor.to(device=device, dtype=dtype)

    # Encode to latent space
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor  # Scale the latent

    return latent


def mask_to_latent(mask_pil, latent_shape, device, dtype):
    """
    Convert PIL mask to latent space mask.
    Downsamples the mask to match the latent dimensions.

    Args:
        mask_pil: PIL Image mask
        latent_shape: Shape of the latent tensor
        device: torch device
        dtype: torch dtype

    Returns:
        Latent mask tensor
    """
    # Convert mask to tensor
    mask_array = np.array(mask_pil).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(
        0).unsqueeze(0)  # (1, 1, H, W)
    mask_tensor = mask_tensor.to(device=device, dtype=dtype)

    # Resize mask to match latent dimensions
    _, _, latent_h, latent_w = latent_shape
    mask_latent = torch.nn.functional.interpolate(
        mask_tensor,
        size=(latent_h, latent_w),
        mode='nearest'
    )

    return mask_latent


def latent_to_pil(latent, vae):
    """
    Decode latent representation back to PIL image using VAE decoder.

    Args:
        latent: Latent tensor
        vae: VAE model

    Returns:
        PIL Image
    """
    # Unscale the latent
    latent = latent / vae.config.scaling_factor

    # Decode
    with torch.no_grad():
        image = vae.decode(latent).sample

    # Convert from [-1, 1] to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)

    # Convert to PIL
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)
    image_pil = Image.fromarray(image)

    return image_pil


def get_text_embeddings(prompt, tokenizer, text_encoder, device, negative_prompt=""):
    """
    Get text embeddings for the prompt and negative prompt.

    Args:
        prompt: Text prompt
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        device: torch device
        negative_prompt: Negative prompt (default: empty)

    Returns:
        Text embeddings tensor (concatenated unconditional and conditional)
    """
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Get unconditional embeddings for classifier-free guidance
    uncond_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings
