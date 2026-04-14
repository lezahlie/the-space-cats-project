"""
Utilities for masked loss functions
"""
from common import pt
F=pt.nn.functional
from torchmetrics.functional import structural_similarity_index_measure

# ==================================================
# CONTRIBUTION START: Masked Reconstruction Loss
# Contributor: Leslie Horace
# ==================================================

def _get_square_region(image, masked_map, i):
    mask_2d = masked_map[i, 0] > 0.5

    rows = mask_2d.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
    cols = mask_2d.any(dim=0).nonzero(as_tuple=False).squeeze(-1)

    if rows.numel() == 0 or cols.numel() == 0:
        raise ValueError(f"No masked pixels found for sample {i}")

    top = int(rows[0].item())
    bottom = int(rows[-1].item()) + 1
    left = int(cols[0].item())
    right = int(cols[-1].item()) + 1

    return image[i:i+1, :, top:bottom, left:right]


def masked_reconstruction_loss(pred_image, target_image, masked_map, ssim_weight=0.25):
    """
    weighted reconstruction loss

    - if a sample has masked pixels: 
        loss is computed only on the masked square
    - otherwise, a sample has no masked pixels: 
        loss is computed on the full image

    The total loss is:
        ell_total = (1 - ssim_weight) * SmoothL1 + ssim_weight * (1 - SSIM)
    """
    if not (0.0 <= ssim_weight <= 1.0):
        raise ValueError(f"ssim_weight must be between [0, 1], not {ssim_weight}")

    smooth_l1_losses = []
    ssim_losses = []

    # sadly we cannot batch losses because the masks are randomly placed
    # however, this makes computing loss less memory intensive 
    for i in range(pred_image.shape[0]):

        # check if we used as mask and get only the masked region
        if pt.any(masked_map[i] == 1.0):
            pred_region = _get_square_region(pred_image, masked_map, i)
            target_region = _get_square_region(target_image, masked_map, i)
        else:  # otherwise no mask, so get the whole image instead
            pred_region = pred_image[i:i+1]
            target_region = target_image[i:i+1]

        # get the reconstruction loss 
        smooth_l1_i = F.smooth_l1_loss(pred_region, target_region, reduction="mean")

        # SSIM uses data_range = max_element - min_element
        # Since images are normalized to [0.0, 1.0], data_range = 1.0.
        ssim_i = structural_similarity_index_measure(
            pred_region,
            target_region,
            data_range=1.0,     
        )

        # SSIM = higher is better, so (1.0 - SSIM) 
        # flips it to lower is better for loss convergence
        ssim_loss_i = 1.0 - ssim_i      

        ssim_losses.append(ssim_loss_i)
        smooth_l1_losses.append(smooth_l1_i)

    smooth_l1 = pt.stack(smooth_l1_losses).mean()
    ssim_loss = pt.stack(ssim_losses).mean()
    total_loss = (1.0 - ssim_weight) * smooth_l1 + ssim_weight * ssim_loss

    return total_loss, {
        "smooth_l1": smooth_l1.detach(),
        "ssim_loss": ssim_loss.detach(),
        "ssim_weight": ssim_weight
    }

# ==================================================
# CONTRIBUTION End: Masked Reconstruction Loss
# ==================================================