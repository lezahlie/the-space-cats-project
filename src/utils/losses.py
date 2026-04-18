from src.utils.common import pt, Optional
F=pt.nn.functional
from torchmetrics.functional import structural_similarity_index_measure

# ==================================================
# CONTRIBUTION START: Masked Reconstruction Loss
# Contributor: Leslie Horace
# ==================================================

def _get_coords_from_mask(masked_region_map: pt.Tensor):
    if masked_region_map.ndim != 4 or masked_region_map.shape[1] != 1:
        raise ValueError(f"Expected masked_region_map shape (B, 1, H, W), not {tuple(masked_region_map.shape)}")

    mask_2d = masked_region_map[:, 0] > 0.0
    row_has_mask = mask_2d.any(dim=2)   # (B, H)
    col_has_mask = mask_2d.any(dim=1)   # (B, W)

    top = row_has_mask.to(pt.int64).argmax(dim=1)    # (B,)
    left = col_has_mask.to(pt.int64).argmax(dim=1)   # (B,)

    row_counts = row_has_mask.sum(dim=1)             # (B,)
    col_counts = col_has_mask.sum(dim=1)             # (B,)


    size = int(row_counts[0].item())
    return top, left, size


def _crop_batch_squares(image: pt.Tensor, top: pt.Tensor, left: pt.Tensor, size: int):
    
    B, C, H, W = image.shape
    device = image.device

    row_offsets = pt.arange(size, device=device)
    col_offsets = pt.arange(size, device=device)

    row_idx = top[:, None] + row_offsets[None, :]    # (B, size)
    col_idx = left[:, None] + col_offsets[None, :]   # (B, size)

    cropped = image.gather(
        dim=2,
        index=row_idx[:, None, :, None].expand(B, C, size, W)
    )
    cropped = cropped.gather(
        dim=3,
        index=col_idx[:, None, None, :].expand(B, C, size, size)
    )

    return cropped


def _smooth_l1_term(recon_image: pt.Tensor, target_image: pt.Tensor, reduction: str) -> pt.Tensor:
    smooth_l1_map = F.smooth_l1_loss(
        recon_image,
        target_image,
        reduction="none"
    )
    per_sample = smooth_l1_map.flatten(start_dim=1)

    if reduction == "sum":
        return per_sample.sum(dim=1).mean()

    return per_sample.mean(dim=1).mean()


def _ssim_loss_term(recon_image: pt.Tensor, target_image: pt.Tensor, **kwargs) -> pt.Tensor:
    ssim_score = structural_similarity_index_measure(
        recon_image,
        target_image,
        data_range=1.0,
        reduction="none",
    )
    ssim_loss = 1.0 - ssim_score

    return ssim_loss.mean()

def masked_reconstruction_loss(recon_image: pt.Tensor, target_image: pt.Tensor, masked_region_map: Optional[pt.Tensor] = None, ssim_weight: float = 0.0, reduction: Optional[str] = "mean", return_objective_only: bool = False):
    """
    Args:
        recon_image (pt.Tensor): reconstructed image x_image_recon from the decoder (with masking if mask_ratio > 0.0)
        target_image (pt.Tensor): original target image y_image from the dataset (without masking)
        masked_region_map (pt.Tensor, optional): masked region binary mask with shape (N, 1, 64, 64).
            Note: If none, loss is computed over the full image.
        ssim_weight (float, optional): normalized weight between [0.0, 1.0] for SSIM loss term. Defaults to 0.0.
            a) ssim_weight == 0.0: Only the Smooth_L1 loss term is used
            b) 0.0 < ssim_weight < 1.0: Remaining weight (1.0 - ssim_weight) is applied to Smooth_L1
            c) ssim_weight == 1.0: Only SSIM loss term is used
        reduction (str, Optional): elementwise per-sample loss reduction "mean" or "sum", if None or invalid is set to the default. Defaults to "mean".
            Note: batch reduction is always mean.
        return_objective_only (bool, optional): Only return the overall loss used for back propagation. Defaults to False.
            Note: The partial loss terms are very helpful for plotting, so it is recommended to leave this as default during tuning and training.

    Returns:
        tensor or tuple(tensor, dict): objective_loss or (objective_loss, partial_losses_dict)
    """
    if not (0.0 <= ssim_weight <= 1.0):
        raise ValueError(f"ssim_weight must be between [0, 1], not {ssim_weight}")

    if reduction not in {"mean", "sum"}:
        reduction = "mean"

    zero = recon_image.sum() * 0.0

    recon_for_loss = recon_image
    target_for_loss = target_image

    skip_mask = masked_region_map is None or pt.all(masked_region_map == 0.0)
    if not skip_mask:
        top, left, size = _get_coords_from_mask(masked_region_map)
        recon_for_loss = _crop_batch_squares(recon_image, top, left, size)
        target_for_loss = _crop_batch_squares(target_image, top, left, size)

    smooth_l1 = zero
    if ssim_weight < 1.0:
        smooth_l1 = _smooth_l1_term(recon_for_loss, target_for_loss, reduction)

    ssim_loss = zero
    if ssim_weight > 0.0:
        ssim_loss = _ssim_loss_term(recon_for_loss, target_for_loss)

    objective_loss = (1.0 - ssim_weight) * smooth_l1 + ssim_weight * ssim_loss

    if return_objective_only:
        return objective_loss

    return objective_loss, smooth_l1.detach(), ssim_loss.detach()

# ==================================================
# CONTRIBUTION End: Masked Reconstruction Loss
# ==================================================


def test_main():
    from src.utils.device import SetupDevice
    from src.utils.logger import set_logger_level, get_logger

    set_logger_level(10)
    logger = get_logger()

    SetupDevice.setup_generators(42)

    B, C, H, W = 32, 5, 64, 64
    image_shape = (B, C, H, W)
    mask_shape = (B, 1, H, W)

    target = pt.zeros(image_shape, dtype=pt.float32)
    recon_exact = target.clone()

    mask = pt.zeros(mask_shape, dtype=pt.float32)
    mask_ratio = 0.5
    mask_size = int(H * mask_ratio)

    top_A = left_A = (H - mask_size) // 2
    top_B = top_A + mask_size
    left_B = left_A + mask_size
    mask[:, :, top_A:top_B, left_A:left_B] = 1.0

    logger.debug(f"target shape: {target.shape}")
    logger.debug(f"recon_exact shape: {recon_exact.shape}")
    logger.debug(f"mask shape: {mask.shape}")
    logger.debug(f"mask box: top_A={top_A}, top_B={top_B}, left_A={left_A}, left_B={left_B}")

    # exact reconstruction
    recon_approx = recon_exact.clone()
    recon_approx[:, :, top_A:top_B, left_A:left_B] = 0.25

    # poor reconstruction
    recon_poor = recon_exact.clone()
    recon_poor[:, :, top_A:top_B, left_A:left_B] = 1.0

    # exact opposite case
    recon_opposite = pt.ones_like(target)

    ssim_weight_list = [0.0, 0.5, 1.0]
    mask_cases = [
        ("no_mask", None),
        ("with_mask", mask),
    ]
    recon_cases = [
        ("recon_exact", recon_exact, "zero"),
        ("recon_approx", recon_approx, "positive"),
        ("recon_poor", recon_poor, "positive"),
        ("recon_opposite", recon_opposite, "positive"),
    ]

    for mask_name, masked_region_map in mask_cases:
        for recon_name, recon_image, expectation in recon_cases:
            for ssim_weight in ssim_weight_list:
                total_loss, smoothl1_loss, ssim_loss = masked_reconstruction_loss(
                    recon_image=recon_image,
                    target_image=target,
                    masked_region_map=masked_region_map,
                    ssim_weight=ssim_weight,
                    reduction="mean",
                )

                logger.debug(f"{recon_name} ({mask_name}, ssim_weight={ssim_weight}):\ntotal_loss={total_loss.item():.8f}, smoothl1_loss={smoothl1_loss}, ssim_loss={ssim_loss}")

                if expectation == "zero":
                    assert pt.isclose(total_loss, pt.tensor(0.0), atol=1e-7), (
                        f"{recon_name} ({mask_name}, ssim_weight={ssim_weight}) should be 0"
                    )
                else:
                    if mask_name == "with_mask" and recon_name == "recon_opposite":
                        assert total_loss > 0.0, (
                            f"{recon_name} ({mask_name}, ssim_weight={ssim_weight}) should be > 0"
                        )
                    else:
                        assert total_loss > 0.0, (
                            f"{recon_name} ({mask_name}, ssim_weight={ssim_weight}) should be > 0"
                        )

    # masked case: difference outside masked region only
    recon_outside = target.clone()
    recon_outside[:, :, 0:4, 0:4] = 1.0

    total_loss, smoothl1_loss, ssim_loss = masked_reconstruction_loss(
        recon_image=recon_outside,
        target_image=target,
        masked_region_map=mask,
        ssim_weight=0.0,
        reduction="mean",
    )
    logger.debug(f"recon_outside (with_mask, ssim_weight=0.0): loss={total_loss.item():.8f}, smoothl1_loss={smoothl1_loss}, ssim_loss={ssim_loss}")
    assert pt.isclose(total_loss, pt.tensor(0.0), atol=1e-7), "outside-mask case should be 0 for smooth_l1-only"

    # objective only
    loss_only = masked_reconstruction_loss(
        recon_image=recon_poor,
        target_image=target,
        masked_region_map=mask,
        ssim_weight=0.25,
        reduction="mean",
        return_objective_only=True,
    )
    logger.debug(f"objective_only={loss_only.item():.8f}")
    assert isinstance(loss_only, pt.Tensor), "objective_only should return a tensor"

    # backward pass
    recon_grad = recon_poor.clone().requires_grad_(True)
    loss_only = masked_reconstruction_loss(
        recon_image=recon_grad,
        target_image=target,
        masked_region_map=mask,
        ssim_weight=0.25,
        reduction="mean",
        return_objective_only=True,
    )
    loss_only.backward()
    logger.debug(f"grad is valid? {recon_grad.grad is not None}")
    assert recon_grad.grad is not None, "backwards pass failed"

    logger.info("All masked reconstruction loss tests passed.")


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger

    init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    test_main()