"""Batched collaborative filters for transformed BM3D patch groups.

Public functions consume dense group buckets with shape ``[M, G, K, K]``:
``M`` groups, ``G`` similar patches per group, and ``K x K`` coefficients per
patch. Group sizes are powers of two so a cached Hadamard matrix can act on
the ``G`` axis.
"""

from __future__ import annotations

import math

import torch


_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_RADIAL_FREQUENCY_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

# Pooled geometric means from the full-resolution DIV2K 0801 coefficient
# oracle were 0.2147 at spatial DC and 1.4779 at the highest 2D-DCT
# frequency. Their ratio defines the first fixed-slope approximation to the
# oracle while retaining only one searched value per matched group.
ORACLE_ENDPOINT_HF_DC_RATIO = 6.884
ORACLE_MULTIPLIER_MIN = 1e-4
ORACLE_MULTIPLIER_MAX = 10.0


def hadamard_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an unnormalized Sylvester Hadamard matrix with shape ``[G, G]``."""
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    _HADAMARD_CACHE[key] = matrix
    return matrix


def radial_frequency_coordinates(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized 2D-DCT radii with DC=0 and corner frequency=1."""
    if size <= 0:
        raise ValueError("DCT size must be positive")
    key = (size, str(device), dtype)
    cached = _RADIAL_FREQUENCY_CACHE.get(key)
    if cached is not None:
        return cached

    if size == 1:
        radius = torch.zeros((1, 1), device=device, dtype=dtype)
    else:
        frequencies = torch.arange(size, device=device, dtype=dtype)
        vertical, horizontal = torch.meshgrid(
            frequencies,
            frequencies,
            indexing="ij",
        )
        radius = torch.sqrt(vertical.square() + horizontal.square())
        radius = radius / radius[-1, -1]
    _RADIAL_FREQUENCY_CACHE[key] = radius
    return radius


def frequency_interpolated_multiplier(
    endpoint: torch.Tensor,
    patch_size: int,
    *,
    interpolate_variance: bool = False,
    endpoint_at_dc: bool = False,
) -> torch.Tensor:
    """Expand scalar or ``[M]`` endpoints into radial ``[...,K,K]`` surfaces.

    Endpoints at or above one remain spatially uniform, exactly matching
    scalar Wiener filtering. Below one, ``endpoint_at_dc=False`` retains the
    original one-at-DC to endpoint-at-high-frequency profile, while true uses
    the corrected endpoint-at-DC to one-at-high-frequency direction.
    """
    if endpoint.ndim not in (0, 1):
        raise ValueError("frequency endpoint must be scalar or one value per group")
    radius = radial_frequency_coordinates(
        patch_size,
        device=endpoint.device,
        dtype=endpoint.dtype,
    )
    base = torch.maximum(endpoint, torch.ones_like(endpoint))
    start = endpoint if endpoint_at_dc else base
    end = base if endpoint_at_dc else endpoint
    if endpoint.ndim == 0:
        if interpolate_variance:
            interpolated = torch.sqrt(
                start.square() + radius * (end.square() - start.square())
            )
        else:
            interpolated = start + radius * (end - start)
        return torch.where(endpoint >= 1, endpoint.expand_as(radius), interpolated)

    if interpolate_variance:
        interpolated = torch.sqrt(
            start[:, None, None].square()
            + radius[None, :, :]
            * (end.square() - start.square())[:, None, None]
        )
    else:
        interpolated = (
            start[:, None, None]
            + radius[None, :, :] * (end - start)[:, None, None]
        )
    return torch.where(
        (endpoint >= 1)[:, None, None],
        endpoint[:, None, None].expand_as(interpolated),
        interpolated,
    )


def log_frequency_slope_multiplier(
    dc_multiplier: torch.Tensor,
    patch_size: int,
    *,
    hf_dc_ratio: float = ORACLE_ENDPOINT_HF_DC_RATIO,
    minimum: float = ORACLE_MULTIPLIER_MIN,
    maximum: float = ORACLE_MULTIPLIER_MAX,
) -> torch.Tensor:
    """Expand searched DC values with a fixed radial slope in log space.

    ``hf_dc_ratio`` is the multiplier ratio between the highest radial
    frequency and DC. Intermediate values follow
    ``dc_multiplier * hf_dc_ratio ** radius`` and are clipped to the range
    used by the coefficient oracle.
    """
    if dc_multiplier.ndim not in (0, 1):
        raise ValueError("DC multiplier must be scalar or one value per group")
    if not math.isfinite(hf_dc_ratio) or hf_dc_ratio <= 0:
        raise ValueError("hf_dc_ratio must be finite and positive")
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("multiplier bounds must be finite")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("multiplier bounds must satisfy 0 < minimum <= maximum")

    radius = radial_frequency_coordinates(
        patch_size,
        device=dc_multiplier.device,
        dtype=dc_multiplier.dtype,
    )
    radial_ratio = torch.pow(
        torch.as_tensor(
            hf_dc_ratio,
            device=dc_multiplier.device,
            dtype=dc_multiplier.dtype,
        ),
        radius,
    )
    if dc_multiplier.ndim == 0:
        surface = dc_multiplier * radial_ratio
    else:
        surface = dc_multiplier[:, None, None] * radial_ratio[None, :, :]
    return surface.clamp(min=minimum, max=maximum)


def sd_weight(groups: torch.Tensor) -> torch.Tensor:
    """Return the legacy inverse sample-SD weight with shape ``[M]``.

    The statistic includes all ``G*K*K`` transformed coefficients in a group,
    matching ``utils.sd_weighting`` in the reference implementation.
    """
    flat = groups.flatten(1)
    count = flat.shape[1]
    if count <= 1:
        return torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype)
    total = flat.sum(dim=1)
    square_total = flat.square().sum(dim=1)
    variance = (square_total - total.square() / count) / (count - 1)
    positive = variance > 0
    # Masking ``variance.rsqrt()`` only in the final torch.where is not enough:
    # rsqrt still runs at zero/negative entries and its masked backward branch
    # can yield NaN through 0 * inf. Keep that hidden branch in-domain too.
    safe_variance = torch.where(
        positive,
        variance.clamp_min(torch.finfo(variance.dtype).eps),
        torch.ones_like(variance),
    )
    return torch.where(
        positive,
        safe_variance.rsqrt(),
        torch.ones_like(variance),
    )


def hard_threshold_groups(
    groups: torch.Tensor,
    sigma: torch.Tensor | float,
    threshold_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-threshold ``[M, G, K, K]`` groups and return groups plus ``[M]`` weights."""
    group_size = groups.shape[1]
    matrix = hadamard_matrix(group_size, device=groups.device, dtype=groups.dtype)
    # [M,G,K,K] -> [M,K,K,G]: Hadamard acts across similar patches,
    # independently for every 2D-transform coefficient.
    values = groups.permute(0, 2, 3, 1)
    transformed = torch.matmul(values, matrix)
    sigma_tensor = torch.as_tensor(sigma, device=groups.device, dtype=groups.dtype)
    threshold = threshold_multiplier * sigma_tensor * math.sqrt(group_size)
    mask = transformed.abs() > threshold
    nonzero = mask.flatten(1).sum(dim=1)
    filtered = torch.where(mask, transformed, torch.zeros_like(transformed))
    filtered = torch.matmul(filtered, matrix) / group_size
    filtered = filtered.permute(0, 3, 1, 2)
    denominator = sigma_tensor.square() * nonzero.to(groups.dtype)
    weight = torch.where(nonzero > 0, denominator.reciprocal(), torch.ones_like(denominator))
    return filtered, weight


def wiener_filter_groups(
    noisy_groups: torch.Tensor,
    basic_groups: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    sigma_multiplier: torch.Tensor | float = 1.0,
    frequency_interpolation: bool | str = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wiener-filter two ``[M, G, K, K]`` banks and return ``[M]`` weights.

    ``noisy_groups`` provides the values being filtered. ``basic_groups``
    estimates their signal power and therefore determines the Wiener gain.
    """
    if noisy_groups.shape != basic_groups.shape:
        raise ValueError("noisy_groups and basic_groups must have identical shapes")
    group_size = noisy_groups.shape[1]
    matrix = hadamard_matrix(
        group_size, device=noisy_groups.device, dtype=noisy_groups.dtype,
    )
    # Move G last so a single batched matmul transforms every coefficient.
    noisy = noisy_groups.permute(0, 2, 3, 1)
    basic = basic_groups.permute(0, 2, 3, 1)
    noisy_h = torch.matmul(noisy, matrix)
    basic_h = torch.matmul(basic, matrix)

    sigma_tensor = torch.as_tensor(
        sigma, device=noisy_groups.device, dtype=noisy_groups.dtype,
    )
    multiplier = torch.as_tensor(
        sigma_multiplier, device=noisy_groups.device, dtype=noisy_groups.dtype,
    )
    if isinstance(sigma_multiplier, (float, int)) and (
        not math.isfinite(float(sigma_multiplier)) or float(sigma_multiplier) <= 0
    ):
        raise ValueError("sigma_multiplier must be finite and positive")
    scalar_multiplier = multiplier.ndim == 0
    group_multiplier = multiplier.shape == (noisy_groups.shape[0],)
    if not scalar_multiplier and not group_multiplier:
        raise ValueError(
            "sigma_multiplier must be scalar or have one value per group; "
            f"got shape {tuple(multiplier.shape)} for {noisy_groups.shape[0]} groups"
        )
    if not bool(torch.isfinite(multiplier).all()) or bool((multiplier <= 0).any()):
        raise ValueError("sigma_multiplier must be finite and positive")

    if frequency_interpolation not in (
        False,
        True,
        "linear",
        "variance",
        "dc_linear",
        "dc_variance",
        "oracle_slope",
    ):
        raise ValueError(
            "frequency_interpolation must be false, true/'linear', 'variance', "
            "'dc_linear', 'dc_variance', or 'oracle_slope'"
        )
    use_frequency_interpolation = bool(frequency_interpolation)
    if use_frequency_interpolation:
        if frequency_interpolation == "oracle_slope":
            surface = log_frequency_slope_multiplier(
                multiplier,
                noisy_groups.shape[-1],
            )
        else:
            surface = frequency_interpolated_multiplier(
                multiplier,
                noisy_groups.shape[-1],
                interpolate_variance=frequency_interpolation in (
                    "variance",
                    "dc_variance",
                ),
                endpoint_at_dc=frequency_interpolation in (
                    "dc_linear",
                    "dc_variance",
                ),
            )
        if scalar_multiplier:
            coefficient_sigma = sigma_tensor * surface[None, :, :, None]
        else:
            coefficient_sigma = sigma_tensor * surface[:, :, :, None]
        weight_sigma = None
    elif scalar_multiplier:
        effective_sigma = sigma_tensor * multiplier
        coefficient_sigma = effective_sigma
        weight_sigma = effective_sigma
    else:
        # A precomputed/search run supplies one multiplier per reference
        # group. Add singleton coefficient axes so it broadcasts over K,K,G.
        weight_sigma = sigma_tensor * multiplier
        coefficient_sigma = weight_sigma[:, None, None, None]
    power = basic_h.square() / group_size
    gain = power / (power + coefficient_sigma.square())
    filtered_h = noisy_h * gain / group_size
    filtered = torch.matmul(filtered_h, matrix).permute(0, 3, 1, 2)

    gain_sum = gain.flatten(1).sum(dim=1)
    if use_frequency_interpolation:
        denominator = (coefficient_sigma.square() * gain).flatten(1).sum(dim=1)
    else:
        denominator = weight_sigma.square() * gain_sum
    weight = torch.where(
        denominator > 0,
        denominator.reciprocal(),
        torch.ones_like(denominator),
    )
    return filtered, weight
