"""
Mask generation functions for image inpainting.
"""
import numpy as np
from PIL import Image


def create_rectangular_mask(width, height, mask_fraction=0.35):
    """
    Create a centered rectangular mask.
    Returns a PIL Image with white (255) for inpaint region, black (0) for keep region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    rect_w = int(width * mask_fraction)
    rect_h = int(height * mask_fraction)

    x0 = (width - rect_w) // 2
    y0 = (height - rect_h) // 2

    mask[y0:y0 + rect_h, x0:x0 + rect_w] = 255

    return Image.fromarray(mask)


def create_circular_mask(width, height, mask_fraction=0.35):
    """
    Create a centered circular mask.
    Returns a PIL Image with white (255) for inpaint region, black (0) for keep region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    radius = int(min(width, height) * mask_fraction / 2)

    center_x = width // 2
    center_y = height // 2

    y, x = np.ogrid[:height, :width]
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask[circle_mask] = 255

    return Image.fromarray(mask)


def create_random_patches_mask(width, height, num_patches=10, patch_size_range=(20, 30)):
    """
    Create a mask with random rectangular patches.
    Returns a PIL Image with white (255) for inpaint region, black (0) for keep region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_patches):
        patch_w = np.random.randint(patch_size_range[0], patch_size_range[1])
        patch_h = np.random.randint(patch_size_range[0], patch_size_range[1])

        x0 = np.random.randint(0, max(1, width - patch_w))
        y0 = np.random.randint(0, max(1, height - patch_h))

        mask[y0:y0 + patch_h, x0:x0 + patch_w] = 255

    return Image.fromarray(mask)


def create_random_noise_mask(width, height, noise_fraction=0.3):
    """
    Create a mask with random pixel removal (salt and pepper style).
    Returns a PIL Image with white (255) for inpaint region, black (0) for keep region.

    Note- wasn't used in the final pipelines
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Randomly select pixels to mask
    num_pixels = int(width * height * noise_fraction)
    random_coords = np.random.choice(width * height, num_pixels, replace=False)
    mask.flat[random_coords] = 255

    return Image.fromarray(mask)


def create_brush_stroke_mask(width, height, num_strokes=5, stroke_width_range=(10, 20)):
    """
    Create a mask simulating brush strokes (useful for realistic inpainting scenarios).
    Returns a PIL Image with white (255) for inpaint region, black (0) for keep region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_strokes):
        stroke_width = np.random.randint(
            stroke_width_range[0], stroke_width_range[1])
        num_points = np.random.randint(5, 15)

        points_x = np.random.randint(0, width, num_points)
        points_y = np.random.randint(0, height, num_points)

        # Draw stroke by connecting points
        for i in range(len(points_x) - 1):
            x1, y1 = points_x[i], points_y[i]
            x2, y2 = points_x[i + 1], points_y[i + 1]

            # Draw line segment
            length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            if length == 0:
                continue

            for t in np.linspace(0, 1, length * 2):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))

                # Draw circle around point to create stroke width
                y_min = max(0, y - stroke_width // 2)
                y_max = min(height, y + stroke_width // 2)
                x_min = max(0, x - stroke_width // 2)
                x_max = min(width, x + stroke_width // 2)

                mask[y_min:y_max, x_min:x_max] = 255

    return Image.fromarray(mask)


def create_half_mask(width, height, direction='left'):
    """
    Create a mask covering half of the image.
    direction: 'left', 'right', 'top', 'bottom'
    Returns a PIL Image with white (255) for inpaint region, black (0) for keep region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if direction == 'left':
        mask[:, :width//2] = 255
    elif direction == 'right':
        mask[:, width//2:] = 255
    elif direction == 'top':
        mask[:height//2, :] = 255
    elif direction == 'bottom':
        mask[height//2:, :] = 255

    return Image.fromarray(mask)


def create_mask(width, height, mask_type='rectangular', **kwargs):
    """
    Create a mask based on the specified type.

    Args:
        width: Image width
        height: Image height
        mask_type: Type of mask ('rectangular', 'circular', 'random_patches', 
                   'random_noise', 'brush_stroke', 'half')
        **kwargs: Additional parameters for specific mask types

    Returns:
        PIL Image with mask
    """
    if mask_type == 'rectangular':
        mask_fraction = kwargs.get('mask_fraction', 0.35)
        return create_rectangular_mask(width, height, mask_fraction)

    elif mask_type == 'circular':
        mask_fraction = kwargs.get('mask_fraction', 0.35)
        return create_circular_mask(width, height, mask_fraction)

    elif mask_type == 'random_patches':
        num_patches = kwargs.get('num_patches', 10)
        patch_size_range = kwargs.get('patch_size_range', (20, 80))
        return create_random_patches_mask(width, height, num_patches, patch_size_range)

    elif mask_type == 'random_noise':
        noise_fraction = kwargs.get('noise_fraction', 0.3)
        return create_random_noise_mask(width, height, noise_fraction)

    elif mask_type == 'brush_stroke':
        num_strokes = kwargs.get('num_strokes', 5)
        stroke_width_range = kwargs.get('stroke_width_range', (10, 20))
        return create_brush_stroke_mask(width, height, num_strokes, stroke_width_range)

    elif mask_type == 'half':
        direction = kwargs.get('direction', 'left')
        return create_half_mask(width, height, direction)

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
