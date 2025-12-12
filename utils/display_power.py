
import torch


def srgb_to_linear(image):                                                                                                                                           
    """Convert an sRGB image to linear color space."""
    threshold = 0.04045
    linear_image = torch.where(image <= threshold,
                               image / 12.92,
                               ((image + 0.055) / 1.055) ** 2.4)
    return linear_image




def display_power_metric_boost(image, scale=3):
    """
    Estimate display power consumption from an input sRGB image.

    Args:
        image (Tensor): Input image in linear sRGB color space, shape (3, H, W),
                        with channels ordered as (R, G, B).
        scale (float): A scaling factor to approximate real display luminance boosting.
                       Recommended value: scale = 3 to match real-world power consumption.
                       Real AR/VR displays require *higher luminance* than standard monitors
                       to achieve better visual quality/contrast, or compensate for ambient light/optics losses.

    Returns:
        float: Estimated display power (W).
    """

    B_coeff=567.6371534
    G_coeff=205.6587915
    R_coeff=228.5245898
    # Convert the image from sRGB to linear color space
    linear_image = srgb_to_linear(image)

    # Calculate power model loss
    power = (
        B_coeff * linear_image[2, :, :].abs().mean() +
        G_coeff * linear_image[1, :, :].abs().mean() +
        R_coeff * linear_image[0, :, :].abs().mean()
    ) / 1000

    power = power * scale
    
    return power

