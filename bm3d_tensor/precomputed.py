"""Reusable grayscale BM3D preparation for per-reference Wiener searches.

The hard stage, Wiener block matching, patch extraction, and 2D transforms are
fixed for one input image.  :class:`BM3DWienerPrecomputed` performs that work
once and then accepts either a scalar Wiener sigma multiplier or one multiplier
per Wiener reference group.  Group-domain error evaluation deliberately stops
before overlap-add, which makes a binary-search condition local to a reference
group and avoids repeating aggregation during every search iteration.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from .config import BM3DConfig, validate_config
from .core import (
    _aggregate_groups,
    _crop_padding,
    _kaiser_window,
    _matching,
    _prepare_image,
    _restore_image,
    _run_stage,
    _validate_spatial_shape,
    symmetric_pad,
)
from .filters import sd_weight, wiener_filter_groups
from .matching import MatchResult
from .transforms import inverse_transform, transformed_patch_bank


ComparisonScope = Literal["all_members", "reference"]
ErrorMetric = Literal["mae", "max_abs"]


class BM3DWienerPrecomputed:
    """Precompute fixed grayscale BM3D work for fast Wiener experiments.

    One multiplier belongs to one Wiener reference group.  A group contains
    ``G`` matched patches, so changing its multiplier jointly changes all
    ``G`` reconstructed members.  ``evaluate_group_errors`` compares those
    estimates with patches extracted from the original clean input, while
    ``run`` performs the normal BM3D overlap-add using the supplied group map.

    Args:
        clean: Clean grayscale ``HW``, ``1HW``, or ``B1HW`` tensor in the
            same 0--255 scale used by :func:`run_bm3d_tensor`.
        sigma: BM3D noise sigma.  Here it controls denoising strength even
            though the image used to construct the content-preservation map
            is clean.
        config: BM3D algorithm/runtime settings.  The hard stage and Wiener
            matching settings become fixed attributes of this object.
        canonical_matches: Optional Wiener-stage match topology computed for
            another realization of the same image. Candidate coordinates and
            group sizes are reused exactly while noisy/basic patch banks still
            come from ``clean``. This supports controlled multi-noise oracle
            experiments with coefficient-aligned groups.
        cache_error_targets: Retain the spatial patch bank needed by
            ``evaluate_group_errors``. Oracle-only rendering can disable this
            sizeable cache; normal binary-search preparation leaves it on.
        use_input_as_basic: Experimental clean-guide mode. When true, bypass
            stage-one hard thresholding and use the input itself as the basic
            estimate for matching and stage-two coefficient preparation. The
            default is false, so production and existing experiment behavior
            is unchanged.
    """

    @torch.no_grad()
    def __init__(
        self,
        clean: torch.Tensor,
        sigma: float,
        config: BM3DConfig | None = None,
        *,
        canonical_matches: MatchResult | None = None,
        cache_error_targets: bool = True,
        use_input_as_basic: bool = False,
    ) -> None:
        image, layout = _prepare_image(clean, channels=1)
        sigma = float(sigma)
        if not torch.isfinite(image.new_tensor(sigma)) or sigma <= 0:
            raise ValueError("sigma must be a finite positive number")
        config = config or BM3DConfig.classic(sigma)
        validate_config(config)
        _validate_spatial_shape(image, config)

        self.config = config
        self.sigma = sigma
        self.device = image.device
        self.dtype = image.dtype
        self._layout = layout
        self._image = image
        self._sigma_tensor = image.new_tensor(sigma)

        # Stage one is independent of the Wiener sigma multiplier and is paid
        # only once.  It fixes both the basic estimate and stage-two matching.
        # Clean direct-gain experiments deliberately bypass it so their map
        # construction has no hidden hard-threshold sigma dependence.
        self.use_input_as_basic = bool(use_input_as_basic)
        if self.use_input_as_basic:
            self._basic_bchw = image
        else:
            hard_radius = config.hard.search_radius
            noisy_hard = symmetric_pad(image, hard_radius)
            basic_padded, _ = _run_stage(
                noisy_hard,
                self._sigma_tensor.reshape(1),
                config,
            )
            self._basic_bchw = _crop_padding(basic_padded, hard_radius)
        self.basic = _restore_image(self._basic_bchw, layout)

        stage = config.wiener
        self._stage = stage
        self._radius = stage.search_radius
        self._clean_padded = symmetric_pad(image, self._radius)
        self._basic_padded = symmetric_pad(self._basic_bchw, self._radius)
        if canonical_matches is None:
            self.matches = _matching(self._basic_padded[:, :1], stage, config)
        else:
            patch_rows = self._basic_padded.shape[-2] - stage.patch_size + 1
            patch_columns = self._basic_padded.shape[-1] - stage.patch_size + 1
            expected_grid = (patch_rows, patch_columns)
            if canonical_matches.patch_grid_shape != expected_grid:
                raise ValueError(
                    "canonical match patch grid does not match the current image: "
                    f"{canonical_matches.patch_grid_shape} != {expected_grid}"
                )
            if canonical_matches.indices.device != image.device:
                raise ValueError("canonical matches must be on the image device")
            if canonical_matches.indices.ndim != 3:
                raise ValueError("canonical match indices must have shape [B,R,Gmax]")
            if canonical_matches.indices.shape[0] != image.shape[0]:
                raise ValueError("canonical matches have a different batch size")
            if canonical_matches.indices.shape[-1] != stage.max_group_size:
                raise ValueError(
                    "canonical matches use a different maximum group size"
                )
            if canonical_matches.group_sizes.shape != canonical_matches.indices.shape[:2]:
                raise ValueError("canonical group sizes do not match canonical indices")
            if canonical_matches.reference_indices.shape != (
                canonical_matches.indices.shape[1],
            ):
                raise ValueError("canonical reference indices have an invalid shape")
            patch_count = patch_rows * patch_columns
            if bool((canonical_matches.indices < 0).any()) or bool(
                (canonical_matches.indices >= patch_count).any()
            ):
                raise ValueError("canonical match index lies outside the patch grid")
            self.matches = canonical_matches

        self.batch_size = image.shape[0]
        self.reference_count = self.matches.indices.shape[1]
        self.patch_size = stage.patch_size
        self.patch_grid_shape = self.matches.patch_grid_shape
        patch_rows, patch_columns = self.patch_grid_shape
        self._patch_columns = patch_columns
        self._patch_count = patch_rows * patch_columns
        self._height, self._width = self._clean_padded.shape[-2:]

        reference_rows = torch.div(
            self.matches.reference_indices,
            patch_columns,
            rounding_mode="floor",
        )
        reference_columns = torch.remainder(
            self.matches.reference_indices,
            patch_columns,
        )
        self.reference_grid_shape = (
            int(torch.unique(reference_rows).numel()),
            int(torch.unique(reference_columns).numel()),
        )
        if self.reference_grid_shape[0] * self.reference_grid_shape[1] != self.reference_count:
            raise RuntimeError("Wiener reference positions do not form a rectangular grid")

        batch_offsets = (
            torch.arange(self.batch_size, device=self.device, dtype=torch.long)[:, None, None]
            * self._patch_count
        )
        self._global_matches = (self.matches.indices + batch_offsets).reshape(
            self.batch_size * self.reference_count,
            stage.max_group_size,
        )
        self._local_matches = self.matches.indices.reshape(
            self.batch_size * self.reference_count,
            stage.max_group_size,
        )
        self._group_sizes = self.matches.group_sizes.reshape(-1)
        self._reference_batches = torch.arange(
            self.batch_size, device=self.device, dtype=torch.long,
        ).repeat_interleave(self.reference_count)
        self._kaiser = _kaiser_window(self.patch_size, image)

        # Cache the two transformed banks used by every candidate multiplier.
        # The spatial clean bank is the target for MAE/max-absolute checks.
        self._noisy_bank = transformed_patch_bank(
            self._clean_padded, self.patch_size, stage.transform,
        )[:, 0].reshape(self.batch_size * self._patch_count, self.patch_size, self.patch_size)
        self._basic_bank = transformed_patch_bank(
            self._basic_padded, self.patch_size, stage.transform,
        )[:, 0].reshape(self.batch_size * self._patch_count, self.patch_size, self.patch_size)
        self._clean_spatial_bank = (
            F.unfold(
                self._clean_padded, kernel_size=self.patch_size, stride=1,
            ).transpose(1, 2).reshape(
                self.batch_size * self._patch_count,
                self.patch_size,
                self.patch_size,
            )
            if cache_error_targets
            else None
        )

        self._selected_by_size: dict[int, torch.Tensor] = {}
        size = 1
        while size <= stage.max_group_size:
            self._selected_by_size[size] = torch.nonzero(
                self._group_sizes == size, as_tuple=False,
            ).flatten()
            size *= 2

    def _prepare_multiplier_map(
        self,
        multiplier: torch.Tensor | float,
    ) -> torch.Tensor:
        """Normalize scalar/reference-grid inputs to contiguous ``[B,R]``."""
        value = torch.as_tensor(multiplier, device=self.device, dtype=self.dtype)
        batch = self.batch_size
        references = self.reference_count
        grid_rows, grid_columns = self.reference_grid_shape

        if value.ndim == 0:
            value = value.expand(batch, references)
        elif value.shape == (batch, references):
            pass
        elif batch == 1 and value.shape == (references,):
            value = value.unsqueeze(0)
        elif batch == 1 and value.shape == (grid_rows, grid_columns):
            value = value.reshape(1, references)
        elif value.shape == (batch, grid_rows, grid_columns):
            value = value.reshape(batch, references)
        else:
            raise ValueError(
                "multiplier must be scalar, [B,R], or a reference-grid map; "
                f"got {tuple(value.shape)}, expected B={batch}, R={references}, "
                f"grid={self.reference_grid_shape}"
            )

        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("all Wiener sigma multipliers must be finite and positive")
        return value.contiguous()

    def multiplier_grid(self, multiplier: torch.Tensor) -> torch.Tensor:
        """View a ``[B,R]`` multiplier map as ``[B,Rh,Rw]``."""
        value = self._prepare_multiplier_map(multiplier)
        return value.reshape(self.batch_size, *self.reference_grid_shape)

    def _filter_chunk(
        self,
        chosen: torch.Tensor,
        group_size: int,
        flat_multipliers: torch.Tensor,
        *,
        frequency_interpolation: bool | str = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter one same-size group chunk and return matches, values, weights."""
        selected_matches = self._global_matches[chosen, :group_size]
        noisy_groups = self._noisy_bank[selected_matches]
        basic_groups = self._basic_bank[selected_matches]
        filtered, filter_weights = wiener_filter_groups(
            noisy_groups,
            basic_groups,
            self._sigma_tensor,
            sigma_multiplier=flat_multipliers[chosen],
            frequency_interpolation=frequency_interpolation,
        )
        return selected_matches, filtered, filter_weights

    @torch.no_grad()
    def evaluate_group_errors(
        self,
        multiplier: torch.Tensor | float,
        *,
        comparison_scope: ComparisonScope = "all_members",
        error_metric: ErrorMetric = "mae",
        frequency_interpolation: bool | str = False,
    ) -> torch.Tensor:
        """Return one clean-content distortion value per reference group.

        ``all_members`` compares all ``G`` jointly reconstructed patches with
        clean patches at their matched locations.  ``reference`` compares only
        member zero.  The result always has shape ``[B,R]``.
        """
        if comparison_scope not in ("all_members", "reference"):
            raise ValueError("comparison_scope must be 'all_members' or 'reference'")
        if error_metric not in ("mae", "max_abs"):
            raise ValueError("error_metric must be 'mae' or 'max_abs'")
        if self._clean_spatial_bank is None:
            raise RuntimeError(
                "group-error targets were not cached during preparation"
            )

        multipliers = self._prepare_multiplier_map(multiplier)
        flat_multipliers = multipliers.reshape(-1)
        errors = torch.empty_like(flat_multipliers)
        chunk_size = self.config.runtime.group_chunk_size

        for group_size, selected in self._selected_by_size.items():
            for start in range(0, selected.numel(), chunk_size):
                chosen = selected[start:start + chunk_size]
                selected_matches, filtered, _ = self._filter_chunk(
                    chosen,
                    group_size,
                    flat_multipliers,
                    frequency_interpolation=frequency_interpolation,
                )
                reconstructed = inverse_transform(filtered, self._stage.transform)
                targets = self._clean_spatial_bank[selected_matches]
                difference = (reconstructed - targets).abs()
                if comparison_scope == "reference":
                    difference = difference[:, 0]
                flattened = difference.flatten(1)
                if error_metric == "mae":
                    chunk_errors = flattened.mean(dim=1)
                else:
                    chunk_errors = flattened.amax(dim=1)
                errors[chosen] = chunk_errors

        return errors.reshape(self.batch_size, self.reference_count)

    def _run_impl(
        self,
        multiplier: torch.Tensor | float,
        *,
        frequency_interpolation: bool | str = False,
    ) -> torch.Tensor:
        """Implement Wiener filtering and overlap-add with optional autograd."""
        multipliers = self._prepare_multiplier_map(multiplier)
        flat_multipliers = multipliers.reshape(-1)
        numerator = torch.zeros_like(self._clean_padded)
        denominator = torch.zeros(
            (self.batch_size, self._height, self._width),
            device=self.device,
            dtype=self.dtype,
        )
        chunk_size = self.config.runtime.group_chunk_size

        for group_size, selected in self._selected_by_size.items():
            for start in range(0, selected.numel(), chunk_size):
                chosen = selected[start:start + chunk_size]
                _, filtered, filter_weights = self._filter_chunk(
                    chosen,
                    group_size,
                    flat_multipliers,
                    frequency_interpolation=frequency_interpolation,
                )
                weights = (
                    sd_weight(filtered)
                    if self._stage.use_sd_weight
                    else filter_weights
                )
                reconstructed = inverse_transform(filtered, self._stage.transform)
                _aggregate_groups(
                    numerator[:, 0],
                    denominator,
                    reconstructed,
                    self._local_matches[chosen, :group_size],
                    self._reference_batches[chosen],
                    weights,
                    self._kaiser,
                    image_width=self._width,
                    patch_columns=self._patch_columns,
                )

        safe_denominator = torch.where(
            denominator > 0,
            denominator,
            torch.ones_like(denominator),
        )
        final_padded = numerator / safe_denominator.unsqueeze(1)
        final = _crop_padding(final_padded, self._radius)
        return _restore_image(final, self._layout)

    @torch.no_grad()
    def run(
        self,
        multiplier: torch.Tensor | float,
        *,
        frequency_interpolation: bool | str = False,
    ) -> torch.Tensor:
        """Run the existing inference-only Wiener reconstruction path."""
        return self._run_impl(
            multiplier,
            frequency_interpolation=frequency_interpolation,
        )

    def run_differentiable(
        self,
        multiplier: torch.Tensor | float,
        *,
        frequency_interpolation: bool | str = False,
    ) -> torch.Tensor:
        """Run Wiener reconstruction while retaining multiplier gradients.

        Preparation, matching, and the basic estimate remain frozen.  This
        method exists for controlled multiplier-oracle experiments; normal
        inference should continue to use :meth:`run`.
        """
        return self._run_impl(
            multiplier,
            frequency_interpolation=frequency_interpolation,
        )
