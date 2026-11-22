"""Experimental 16-value H0/H1+ radial BM3D Wiener oracle."""

from __future__ import annotations

import torch
from torch import nn

from .core import _crop_padding, _restore_image
from .oracle import (
    BM3DMultiplierOracle,
    _aggregate_weighted_groups,
    radial_row_log_surface,
)
from .precomputed import BM3DWienerPrecomputed


class BM3DDualRadialOracle(BM3DMultiplierOracle):
    """Learn independent radial vectors for H0 and the shared H1+ rows.

    Every reference group owns exactly ``2 * patch_size`` log multipliers.
    The first vector controls collaborative/Hadamard component H0.  The
    second is shared by every remaining component H1 through H(G-1).  Each
    vector is expanded into a symmetric ``patch_size x patch_size`` DCT
    surface using Euclidean-radius interpolation in log-multiplier space.

    Parameters are attached to reference locations rather than match slots,
    so the same learned map remains well-defined if a realization changes
    its candidates or group size.
    """

    def __init__(
        self,
        prepared: BM3DWienerPrecomputed,
        *,
        multiplier_min: float = 1e-4,
        multiplier_max: float = 10.0,
        initial_multiplier: float = 1.0,
        checkpoint_chunks: bool = False,
    ) -> None:
        super().__init__(
            prepared,
            mode="per_group",
            multiplier_min=multiplier_min,
            multiplier_max=multiplier_max,
            initial_multiplier=initial_multiplier,
            checkpoint_chunks=checkpoint_chunks,
        )
        initial = self.log_group_multipliers
        assert initial is not None
        self.log_group_multipliers = nn.Parameter(
            initial[..., None, None]
            .expand(
                prepared.batch_size,
                prepared.reference_count,
                2,
                prepared.patch_size,
            )
            .clone()
        )

    def radial_log_vectors(self) -> torch.Tensor:
        """Return the differentiable ``[B,R,2,K]`` radial log map."""
        assert self.log_group_multipliers is not None
        return self.log_group_multipliers

    def radial_multipliers(self) -> torch.Tensor:
        """Return the differentiable ``[B,R,2,K]`` multiplier map."""
        return self.radial_log_vectors().exp()

    def expanded_pair_surfaces(self) -> torch.Tensor:
        """Return the generated ``[B,R,2,K,K]`` H0/H1+ surfaces."""
        return radial_row_log_surface(
            self.radial_log_vectors(), self.prepared.patch_size
        ).exp()

    def expanded_group_multipliers(
        self,
        chosen: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Expand chosen references to ``[M,G,K,K]`` coefficient gains."""
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        flat_logs = self.radial_log_vectors().reshape(
            self.prepared.batch_size * self.prepared.reference_count,
            2,
            self.prepared.patch_size,
        )
        pair = radial_row_log_surface(
            flat_logs[chosen], self.prepared.patch_size
        ).exp()
        if group_size == 1:
            return pair[:, :1]
        return torch.cat(
            (pair[:, :1], pair[:, 1:2].expand(-1, group_size - 1, -1, -1)),
            dim=1,
        )

    def use_reference_aligned_prepared_(
        self,
        prepared: BM3DWienerPrecomputed,
    ) -> None:
        """Switch realizations while allowing candidates/group sizes to vary."""
        current = self.prepared
        for name in (
            "batch_size",
            "reference_count",
            "patch_size",
            "reference_grid_shape",
            "patch_grid_shape",
            "_height",
            "_width",
        ):
            if getattr(prepared, name) != getattr(current, name):
                raise ValueError(f"prepared realization differs in {name}")
        if prepared.device != current.device or prepared.dtype != current.dtype:
            raise ValueError("prepared realization differs in device or dtype")
        self.prepared = prepared

    def _forward_with_basic_power_floor(
        self,
        basic_power_floor_fraction: float,
    ) -> torch.Tensor:
        """Render with an optional expected-power correction in Wiener gains."""
        prepared = self.prepared
        numerator = torch.zeros_like(prepared._clean_padded)
        denominator = torch.zeros(
            (prepared.batch_size, prepared._height, prepared._width),
            device=prepared.device,
            dtype=prepared.dtype,
        )
        chunk_size = prepared.config.runtime.group_chunk_size

        for group_size, selected in prepared._selected_by_size.items():
            for start in range(0, selected.numel(), chunk_size):
                chosen = selected[start:start + chunk_size]
                selected_matches = prepared._global_matches[chosen, :group_size]
                multipliers = self.expanded_group_multipliers(chosen, group_size)
                weighted_patches, denominator_windows = (
                    self._maybe_weighted_reconstruct_chunk(
                        selected_matches,
                        multipliers,
                        coefficient_mode=True,
                        basic_power_floor_fraction=basic_power_floor_fraction,
                    )
                )
                _aggregate_weighted_groups(
                    numerator[:, 0],
                    denominator,
                    weighted_patches,
                    denominator_windows,
                    prepared._local_matches[chosen, :group_size],
                    prepared._reference_batches[chosen],
                    image_width=prepared._width,
                    patch_columns=prepared._patch_columns,
                )

        safe_denominator = torch.where(
            denominator > 0,
            denominator,
            torch.ones_like(denominator),
        )
        final_padded = numerator / safe_denominator.unsqueeze(1)
        final = _crop_padding(final_padded, prepared._radius)
        return _restore_image(final, prepared._layout)

    def forward_with_basic_power_floor(
        self,
        basic_power_floor_fraction: float,
    ) -> torch.Tensor:
        """Render clean data using a calibrated expected-noisy power floor.

        This is an experimental calibration path. Normal oracle inference and
        production BM3D continue to use :meth:`forward`, where the floor is
        exactly zero.
        """
        return self._forward_with_basic_power_floor(basic_power_floor_fraction)

    def forward(self) -> torch.Tensor:
        """Render using one H0 and one shared H1+ radial surface per group."""
        return self._forward_with_basic_power_floor(0.0)
