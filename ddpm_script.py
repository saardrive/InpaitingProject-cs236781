"""
Vanilla DDPM inpainting functions.
"""
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F_nn


def vanilla_ddpm_inpainting(
    original_latent,
    mask_latent,
    text_embeddings,
    unet,
    scheduler,
    device,
    dtype,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
):
    """
    Vanilla DDPM inpainting in latent space with text conditioning.

    At each step t:
    1. Denoise the current latent z_t using the UNet
    2. Add noise to the original latent to get x_t (noisy version at step t)
    3. Combine: z_t = mask * z_t_denoised + (1 - mask) * x_t_noisy

    This ensures known regions stay close to the original while unknown regions are generated.

    Args:
        original_latent: Original image latent (from masked image)
        mask_latent: Mask in latent space
        text_embeddings: Text conditioning embeddings
        unet: UNet model
        scheduler: Diffusion scheduler
        device: torch device
        dtype: torch dtype
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed

    Returns:
        Inpainted latent tensor
    """
    scheduler.set_timesteps(num_inference_steps)

    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        original_latent.shape,
        generator=generator,
        device=device,
        dtype=dtype
    )
    latent = latent * scheduler.init_noise_sigma

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Inpainting")):
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        latent = scheduler.step(noise_pred, t, latent,
                                generator=generator).prev_sample

        # replace known regions with noisy original
        if i < len(scheduler.timesteps) - 1:
            noise = torch.randn(original_latent.shape,
                                generator=generator, device=device, dtype=dtype)
            noisy_original = scheduler.add_noise(
                original_latent, noise, t.unsqueeze(0))

            # mask_latent: 1 = inpaint (unknown), 0 = keep (known)
            latent = mask_latent * latent + (1 - mask_latent) * noisy_original

    return latent


def ddpm_spatial_interpolate_inpainting(
    masked_latent,
    mask_latent,
    text_embeddings,
    unet,
    scheduler,
    device,
    dtype,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    steps_avg=5,  
    area=1,  # 1 padding creates a 3x3 averaging window
    # "regular" (9-pixel) \ "cross" (5-pixel plus) \ "x" (5-pixel diagonal)
    kernel_type="regular"
):
    """
    DDPM inpainting with spatial boundary averaging between adjacent pixels.

    Args:
        masked_latent: Masked image latent
        mask_latent: Mask in latent space
        text_embeddings: Text conditioning embeddings
        unet: UNet model
        scheduler: Diffusion scheduler
        device: torch device
        dtype: torch dtype
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed
        steps_avg: How often to apply spatial averaging (every N steps)
        area: Padding for averaging window (1 creates a 3x3 window)

    Returns:
        Inpainted latent tensor
    """
    scheduler.set_timesteps(num_inference_steps)

    pad = int(area)
    mask_kernel_size = (pad * 2) + 1

    dilated_mask = F_nn.max_pool2d(
        mask_latent, kernel_size=mask_kernel_size, stride=1, padding=pad)
    eroded_mask = 1.0 - \
        F_nn.max_pool2d(1.0 - mask_latent,
                        kernel_size=mask_kernel_size, stride=1, padding=pad)
    boundary_mask = dilated_mask - eroded_mask

    channels = masked_latent.shape[1]

    if kernel_type == "cross":
        custom_kernel_tensor = torch.tensor([
            [0.0, 0.2, 0.0],
            [0.2, 0.2, 0.2],
            [0.0, 0.2, 0.0]
        ], device=masked_latent.device, dtype=masked_latent.dtype)
        # reshape for depth
        custom_kernel_tensor = custom_kernel_tensor.view(
            1, 1, 3, 3).repeat(channels, 1, 1, 1)
        
    elif kernel_type == "x":
        custom_kernel_tensor = torch.tensor([
            [0.2, 0.0, 0.2],
            [0.0, 0.2, 0.0],
            [0.2, 0.0, 0.2]
        ], device=masked_latent.device, dtype=masked_latent.dtype)
        custom_kernel_tensor = custom_kernel_tensor.view(
            1, 1, 3, 3).repeat(channels, 1, 1, 1)

    # Initialize with random noise
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        masked_latent.shape,
        generator=generator,
        device=device,
        dtype=dtype
    )
    latent = latent * scheduler.init_noise_sigma

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t,
                              encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        latent = scheduler.step(noise_pred, t, latent,
                                generator=generator).prev_sample

        if i < len(scheduler.timesteps) - 1:
            noise = torch.randn(masked_latent.shape,
                                generator=generator, device=device, dtype=dtype)
            noisy_original = scheduler.add_noise(
                masked_latent, noise, t.unsqueeze(0))

            # spatial averaging

            # create hard stitch
            hard_stitched = mask_latent * latent + \
                (1 - mask_latent) * noisy_original

            # average
            if i % steps_avg == 0:

                # avg_pool_2d for regular 
                if kernel_type == "regular":
                    spatially_averaged = F_nn.avg_pool2d(
                        hard_stitched, kernel_size=3, stride=1, padding=1)
                    
                # 2d conv for the other options
                elif kernel_type == "cross":
                    spatially_averaged = F_nn.conv2d(
                        hard_stitched, custom_kernel_tensor, padding=1, groups=channels)
                elif kernel_type == "x":
                    spatially_averaged = F_nn.conv2d(
                        hard_stitched, custom_kernel_tensor, padding=1, groups=channels)
                else:
                    raise ValueError(
                        "kernel_type must be either 'regular', 'cross', or 'x'")

                # apply only to the boundaries
                latent = boundary_mask * spatially_averaged + \
                    (1 - boundary_mask) * hard_stitched
            else:
                latent = hard_stitched

    return latent
