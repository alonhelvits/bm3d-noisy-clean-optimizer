"""Differentiable oracle parameterizations for the BM3D Wiener stage.

This module is deliberately separate from the production BM3D entry points.
It freezes the hard stage, block matching, transformed patch banks, and noise
realization stored by :class:`BM3DWienerPrecomputed`, then exposes either one
learnable multiplier per reference group or one per transformed 3D
coefficient.  The latter coefficients live after both the 2D patch transform
and the collaborative Hadamard transform.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .core import _crop_padding, _restore_image
from .filters import hadamard_matrix, sd_weight, wiener_filter_groups
from .precomputed import BM3DWienerPrecomputed
from .transforms import inverse_transform


OracleMode = Literal[
    "per_group",
    "per_coefficient",
    "per_radial_row",
    "per_square_shell",
]


def radial_row_log_surface(row_logs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Interpolate ``[...,K]`` log multipliers into symmetric ``[...,K,K]`` maps.

    The learned vector is the first row and, by symmetry, the first column.
    Interior locations use linear interpolation at radius ``sqrt(u**2+v**2)``.
    Radii beyond the learned axis extent are clamped to the final value.
    """
    if row_logs.ndim < 1 or row_logs.shape[-1] != patch_size:
        raise ValueError(
            f"radial row must end in patch size {patch_size}; got {tuple(row_logs.shape)}"
        )
    frequencies = torch.arange(
        patch_size,
        device=row_logs.device,
        dtype=row_logs.dtype,
    )
    vertical, horizontal = torch.meshgrid(
        frequencies,
        frequencies,
        indexing="ij",
    )
    radius = torch.sqrt(vertical.square() + horizontal.square()).clamp_max(
        patch_size - 1,
    )
    lower = radius.floor().to(torch.int64)
    upper = radius.ceil().to(torch.int64)
    fraction = radius - lower.to(radius.dtype)
    lower_values = row_logs[..., lower]
    upper_values = row_logs[..., upper]
    return lower_values + fraction * (upper_values - lower_values)


def square_shell_log_surface(
    shell_logs: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Expand ``[...,K]`` log multipliers into square L-shaped shells.

    A coefficient at DCT location ``(u,v)`` uses shell ``max(u,v)``. Thus
    shell zero contains DC, shell one contains ``(1,0)``, ``(0,1)``, and
    ``(1,1)``, and shell ``k`` contains the ``2*k + 1`` new locations on the
    bottom and right edges of the square surrounding DC.
    """
    if shell_logs.ndim < 1 or shell_logs.shape[-1] != patch_size:
        raise ValueError(
            f"square shells must end in patch size {patch_size}; "
            f"got {tuple(shell_logs.shape)}"
        )
    frequencies = torch.arange(patch_size, device=shell_logs.device)
    vertical, horizontal = torch.meshgrid(
        frequencies,
        frequencies,
        indexing="ij",
    )
    shell = torch.maximum(vertical, horizontal)
    return shell_logs[..., shell]


def _coefficient_wiener_filter_groups(
    noisy_groups: torch.Tensor,
    basic_groups: torch.Tensor,
    sigma: torch.Tensor,
    multipliers: torch.Tensor,
    *,
    basic_power_floor_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wiener-filter with one multiplier per transformed 3D coefficient.

    All group tensors use ``[M,G,K,K]`` layout.  The multiplier at each entry
    controls the aligned coefficient after the Hadamard transform along G;
    consequently it is permuted to ``[M,K,K,G]`` before forming the gain.
    """
    if noisy_groups.shape != basic_groups.shape:
        raise ValueError("noisy_groups and basic_groups must have identical shapes")
    if multipliers.shape != noisy_groups.shape:
        raise ValueError(
            "coefficient multipliers must match the group tensor shape; "
            f"got {tuple(multipliers.shape)} and {tuple(noisy_groups.shape)}"
        )
    if (
        not math.isfinite(float(basic_power_floor_fraction))
        or basic_power_floor_fraction < 0
    ):
        raise ValueError("basic_power_floor_fraction must be finite and nonnegative")

    group_size = noisy_groups.shape[1]
    matrix = hadamard_matrix(
        group_size,
        device=noisy_groups.device,
        dtype=noisy_groups.dtype,
    )
    noisy_h = torch.matmul(noisy_groups.permute(0, 2, 3, 1), matrix)
    basic_h = torch.matmul(basic_groups.permute(0, 2, 3, 1), matrix)
    coefficient_sigma = sigma * multipliers.permute(0, 2, 3, 1)

    power = basic_h.square() / group_size + (
        basic_power_floor_fraction * sigma.square()
    )
    gain = power / (power + coefficient_sigma.square())
    filtered_h = noisy_h * gain / group_size
    filtered = torch.matmul(filtered_h, matrix).permute(0, 3, 1, 2)

    # This is the coefficient-wise generalization of the legacy scalar
    # Wiener aggregation weight.  The classic configuration uses SD weights,
    # but retaining the correct value keeps other configurations well-defined.
    denominator = (coefficient_sigma.square() * gain).flatten(1).sum(dim=1)
    weight = torch.where(
        denominator > 0,
        denominator.reciprocal(),
        torch.ones_like(denominator),
    )
    return filtered, weight


def _aggregate_weighted_groups(
    accumulator: torch.Tensor,
    denominator: torch.Tensor,
    weighted_patches: torch.Tensor,
    denominator_windows: torch.Tensor,
    local_patch_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    image_width: int,
    patch_columns: int,
) -> None:
    """Scatter already weighted patches and group windows into image buffers."""
    group_count, group_size, patch_size, _ = weighted_patches.shape
    member_indices = local_patch_indices.reshape(group_count * group_size)
    member_batches = batch_indices[:, None].expand(-1, group_size).reshape(-1)
    rows = torch.div(member_indices, patch_columns, rounding_mode="floor")
    columns = torch.remainder(member_indices, patch_columns)
    image_area = accumulator.shape[-2] * accumulator.shape[-1]
    bases = member_batches * image_area + rows * image_width + columns
    pixel_rows = torch.arange(
        patch_size,
        device=weighted_patches.device,
        dtype=torch.long,
    )
    pixel_columns = torch.arange(
        patch_size,
        device=weighted_patches.device,
        dtype=torch.long,
    )
    offset_rows, offset_columns = torch.meshgrid(
        pixel_rows,
        pixel_columns,
        indexing="ij",
    )
    pixel_offsets = (offset_rows * image_width + offset_columns).flatten()
    destinations = bases[:, None] + pixel_offsets[None, :]

    accumulator.reshape(-1).scatter_add_(
        0,
        destinations.flatten(),
        weighted_patches.reshape(group_count * group_size, -1).flatten(),
    )
    denominator_values = denominator_windows[:, None].expand(
        -1,
        group_size,
        -1,
        -1,
    )
    denominator.reshape(-1).scatter_add_(
        0,
        destinations.flatten(),
        denominator_values.reshape(group_count * group_size, -1).flatten(),
    )


class BM3DMultiplierOracle(nn.Module):
    """Optimize only bounded Wiener multipliers for one frozen preparation.

    Parameters are stored in log space because the requested interval spans
    five orders of magnitude.  ``project_`` implements projected gradient
    optimization by clamping those log parameters after every optimizer step.
    """

    def __init__(
        self,
        prepared: BM3DWienerPrecomputed,
        *,
        mode: OracleMode,
        multiplier_min: float = 1e-4,
        multiplier_max: float = 10.0,
        initial_multiplier: float = 1.0,
        checkpoint_chunks: bool = False,
    ) -> None:
        super().__init__()
        if mode not in (
            "per_group",
            "per_coefficient",
            "per_radial_row",
            "per_square_shell",
        ):
            raise ValueError(
                "mode must be 'per_group', 'per_coefficient', "
                "'per_radial_row', or 'per_square_shell'"
            )
        bounds = (multiplier_min, multiplier_max, initial_multiplier)
        if not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("multiplier bounds and initial value must be finite")
        if multiplier_min <= 0 or multiplier_max <= multiplier_min:
            raise ValueError("require 0 < multiplier_min < multiplier_max")
        if not multiplier_min <= initial_multiplier <= multiplier_max:
            raise ValueError("initial_multiplier must lie inside the bounds")

        # BM3DWienerPrecomputed is intentionally a plain Python object rather
        # than an nn.Module, so assigning it cannot register its cached tensors
        # as parameters.  Only the log-multiplier tensors below are learnable.
        self.prepared = prepared
        self.mode: OracleMode = mode
        self.multiplier_min = float(multiplier_min)
        self.multiplier_max = float(multiplier_max)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self._log_min = math.log(self.multiplier_min)
        self._log_max = math.log(self.multiplier_max)
        initial_log = math.log(float(initial_multiplier))

        if mode == "per_group":
            self.log_group_multipliers = nn.Parameter(
                torch.full(
                    (prepared.batch_size, prepared.reference_count),
                    initial_log,
                    device=prepared.device,
                    dtype=prepared.dtype,
                )
            )
            self.log_coefficient_multipliers = nn.ParameterDict()
            self.log_radial_row_multipliers = nn.ParameterDict()
            self.log_square_shell_multipliers = nn.ParameterDict()
        elif mode == "per_coefficient":
            self.register_parameter("log_group_multipliers", None)
            self.log_coefficient_multipliers = nn.ParameterDict()
            self.log_radial_row_multipliers = nn.ParameterDict()
            self.log_square_shell_multipliers = nn.ParameterDict()
            patch_size = prepared.patch_size
            for group_size, selected in prepared._selected_by_size.items():
                self.log_coefficient_multipliers[f"g{group_size}"] = nn.Parameter(
                    torch.full(
                        (selected.numel(), group_size, patch_size, patch_size),
                        initial_log,
                        device=prepared.device,
                        dtype=prepared.dtype,
                    )
                )
        elif mode == "per_radial_row":
            self.register_parameter("log_group_multipliers", None)
            self.log_coefficient_multipliers = nn.ParameterDict()
            self.log_radial_row_multipliers = nn.ParameterDict()
            self.log_square_shell_multipliers = nn.ParameterDict()
            patch_size = prepared.patch_size
            for group_size, selected in prepared._selected_by_size.items():
                self.log_radial_row_multipliers[f"g{group_size}"] = nn.Parameter(
                    torch.full(
                        (selected.numel(), group_size, patch_size),
                        initial_log,
                        device=prepared.device,
                        dtype=prepared.dtype,
                    )
                )
        else:
            self.register_parameter("log_group_multipliers", None)
            self.log_coefficient_multipliers = nn.ParameterDict()
            self.log_radial_row_multipliers = nn.ParameterDict()
            self.log_square_shell_multipliers = nn.ParameterDict()
            patch_size = prepared.patch_size
            for group_size, selected in prepared._selected_by_size.items():
                self.log_square_shell_multipliers[f"g{group_size}"] = nn.Parameter(
                    torch.full(
                        (selected.numel(), group_size, patch_size),
                        initial_log,
                        device=prepared.device,
                        dtype=prepared.dtype,
                    )
                )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def use_prepared_(self, prepared: BM3DWienerPrecomputed) -> None:
        """Switch to a coefficient-aligned preparation for another realization.

        Learnable tensors are bucketed by canonical group size and bucket
        position. Requiring identical selected reference indices and local
        matches prevents accidentally applying them to unrelated groups.
        """
        current = self.prepared
        scalar_attributes = (
            "batch_size",
            "reference_count",
            "patch_size",
            "reference_grid_shape",
            "patch_grid_shape",
            "_height",
            "_width",
        )
        for name in scalar_attributes:
            if getattr(prepared, name) != getattr(current, name):
                raise ValueError(f"prepared realization differs in {name}")
        if prepared.device != current.device or prepared.dtype != current.dtype:
            raise ValueError("prepared realization differs in device or dtype")
        if prepared._selected_by_size.keys() != current._selected_by_size.keys():
            raise ValueError("prepared realization has different group-size buckets")
        for group_size, selected in current._selected_by_size.items():
            if not torch.equal(prepared._selected_by_size[group_size], selected):
                raise ValueError(
                    f"prepared realization has different G={group_size} references"
                )
        if not torch.equal(prepared._local_matches, current._local_matches):
            raise ValueError("prepared realization has different candidate matches")
        self.prepared = prepared

    def group_multipliers(self) -> torch.Tensor:
        """Return the differentiable ``[B,R]`` group multiplier map."""
        if self.log_group_multipliers is None:
            raise RuntimeError("group_multipliers is only available in per_group mode")
        return self.log_group_multipliers.exp()

    def coefficient_multipliers(
        self,
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        """Yield ``(G, global group indices, multipliers [M,G,K,K])``."""
        if self.mode != "per_coefficient":
            raise RuntimeError(
                "coefficient_multipliers is only available in per_coefficient mode"
            )
        for group_size, selected in self.prepared._selected_by_size.items():
            yield (
                group_size,
                selected,
                self.log_coefficient_multipliers[f"g{group_size}"].exp(),
            )

    def expanded_radial_row_multipliers(
        self,
        group_size: int,
        start: int | None = None,
        stop: int | None = None,
    ) -> torch.Tensor:
        """Return full ``[M,G,K,K]`` multipliers generated by radial rows."""
        if self.mode != "per_radial_row":
            raise RuntimeError(
                "expanded_radial_row_multipliers requires per_radial_row mode"
            )
        row_logs = self.log_radial_row_multipliers[f"g{group_size}"]
        if start is not None or stop is not None:
            row_logs = row_logs[slice(start, stop)]
        return radial_row_log_surface(row_logs, self.prepared.patch_size).exp()

    def expanded_square_shell_multipliers(
        self,
        group_size: int,
        start: int | None = None,
        stop: int | None = None,
    ) -> torch.Tensor:
        """Return full ``[M,G,K,K]`` multipliers generated by square shells."""
        if self.mode != "per_square_shell":
            raise RuntimeError(
                "expanded_square_shell_multipliers requires per_square_shell mode"
            )
        shell_logs = self.log_square_shell_multipliers[f"g{group_size}"]
        if start is not None or stop is not None:
            shell_logs = shell_logs[slice(start, stop)]
        return square_shell_log_surface(
            shell_logs,
            self.prepared.patch_size,
        ).exp()

    @torch.no_grad()
    def initialize_from_group_map_(self, multiplier: torch.Tensor | float) -> None:
        """Broadcast a group map over every coefficient in its matched tensor."""
        values = self.prepared._prepare_multiplier_map(multiplier).reshape(-1)
        tolerance = torch.finfo(values.dtype).eps * self.multiplier_max * 8
        if bool((values < self.multiplier_min - tolerance).any()) or bool(
            (values > self.multiplier_max + tolerance).any()
        ):
            raise ValueError("initial group map lies outside the oracle bounds")
        values = values.clamp(self.multiplier_min, self.multiplier_max)
        logs = values.log()
        if self.mode == "per_group":
            assert self.log_group_multipliers is not None
            self.log_group_multipliers.copy_(logs.reshape_as(self.log_group_multipliers))
            return

        for group_size, selected in self.prepared._selected_by_size.items():
            if self.mode == "per_coefficient":
                parameter = self.log_coefficient_multipliers[f"g{group_size}"]
                parameter.copy_(
                    logs[selected, None, None, None].expand_as(parameter)
                )
            elif self.mode == "per_radial_row":
                parameter = self.log_radial_row_multipliers[f"g{group_size}"]
                parameter.copy_(logs[selected, None, None].expand_as(parameter))
            else:
                parameter = self.log_square_shell_multipliers[f"g{group_size}"]
                parameter.copy_(logs[selected, None, None].expand_as(parameter))

    @torch.no_grad()
    def project_(self) -> None:
        """Clamp all learned log multipliers to the configured closed interval."""
        for parameter in self.parameters():
            parameter.clamp_(self._log_min, self._log_max)

    def _weighted_reconstruct_chunk(
        self,
        selected_matches: torch.Tensor,
        multipliers: torch.Tensor,
        *,
        coefficient_mode: bool,
        basic_power_floor_fraction: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gathering inside the checkpointed function is intentional.  Passing
        # gathered [M,G,K,K] banks as checkpoint inputs would retain every
        # chunk until backward and defeat checkpointing on full-size images.
        noisy_groups = self.prepared._noisy_bank[selected_matches]
        basic_groups = self.prepared._basic_bank[selected_matches]
        if coefficient_mode:
            filtered, filter_weights = _coefficient_wiener_filter_groups(
                noisy_groups,
                basic_groups,
                self.prepared._sigma_tensor,
                multipliers,
                basic_power_floor_fraction=basic_power_floor_fraction,
            )
        else:
            if basic_power_floor_fraction != 0.0:
                raise ValueError(
                    "a basic-power floor is only defined for coefficient mode"
                )
            filtered, filter_weights = wiener_filter_groups(
                noisy_groups,
                basic_groups,
                self.prepared._sigma_tensor,
                sigma_multiplier=multipliers,
            )
        weights = (
            sd_weight(filtered)
            if self.prepared._stage.use_sd_weight
            else filter_weights
        )
        reconstructed = inverse_transform(filtered, self.prepared._stage.transform)
        window = self.prepared._kaiser[None, None]
        weighted_patches = reconstructed * weights[:, None, None, None] * window
        denominator_windows = weights[:, None, None] * self.prepared._kaiser[None]
        return weighted_patches, denominator_windows

    def _maybe_weighted_reconstruct_chunk(
        self,
        selected_matches: torch.Tensor,
        multipliers: torch.Tensor,
        *,
        coefficient_mode: bool,
        basic_power_floor_fraction: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.checkpoint_chunks or not self.training:
            return self._weighted_reconstruct_chunk(
                selected_matches,
                multipliers,
                coefficient_mode=coefficient_mode,
                basic_power_floor_fraction=basic_power_floor_fraction,
            )

        def reconstruct(
            multiplier_values: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._weighted_reconstruct_chunk(
                selected_matches,
                multiplier_values,
                coefficient_mode=coefficient_mode,
                basic_power_floor_fraction=basic_power_floor_fraction,
            )

        return checkpoint(
            reconstruct,
            multipliers,
            use_reentrant=False,
        )

    def forward(self) -> torch.Tensor:
        """Render the final image while retaining multiplier gradients."""
        prepared = self.prepared
        numerator = torch.zeros_like(prepared._clean_padded)
        denominator = torch.zeros(
            (prepared.batch_size, prepared._height, prepared._width),
            device=prepared.device,
            dtype=prepared.dtype,
        )
        chunk_size = prepared.config.runtime.group_chunk_size
        flat_group_multipliers = (
            self.group_multipliers().reshape(-1)
            if self.mode == "per_group"
            else None
        )

        for group_size, selected in prepared._selected_by_size.items():
            coefficient_logs = (
                self.log_coefficient_multipliers[f"g{group_size}"]
                if self.mode == "per_coefficient"
                else None
            )
            radial_row_logs = (
                self.log_radial_row_multipliers[f"g{group_size}"]
                if self.mode == "per_radial_row"
                else None
            )
            square_shell_logs = (
                self.log_square_shell_multipliers[f"g{group_size}"]
                if self.mode == "per_square_shell"
                else None
            )
            for start in range(0, selected.numel(), chunk_size):
                chosen = selected[start:start + chunk_size]
                selected_matches = prepared._global_matches[chosen, :group_size]
                if coefficient_logs is None:
                    if radial_row_logs is None and square_shell_logs is None:
                        assert flat_group_multipliers is not None
                        multipliers = flat_group_multipliers[chosen]
                        coefficient_mode = False
                    elif radial_row_logs is not None:
                        multipliers = radial_row_log_surface(
                            radial_row_logs[start:start + chosen.numel()],
                            prepared.patch_size,
                        ).exp()
                        coefficient_mode = True
                    else:
                        assert square_shell_logs is not None
                        multipliers = square_shell_log_surface(
                            square_shell_logs[start:start + chosen.numel()],
                            prepared.patch_size,
                        ).exp()
                        coefficient_mode = True
                else:
                    multipliers = coefficient_logs[
                        start:start + chosen.numel()
                    ].exp()
                    coefficient_mode = True

                weighted_patches, denominator_windows = self._maybe_weighted_reconstruct_chunk(
                    selected_matches,
                    multipliers,
                    coefficient_mode=coefficient_mode,
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


class BM3DTemplateGroupOracle(BM3DMultiplierOracle):
    """Learn one group scalar on top of fixed 3D coefficient templates.

    A template has shape ``[G,K,K]`` for each supported group size and lives
    in the same transformed coefficient domain as ``per_coefficient`` oracle
    parameters. The learned scalar multiplies every template coefficient in
    one reference group, preserving one ordered strength control per group.
    """

    def __init__(
        self,
        prepared: BM3DWienerPrecomputed,
        coefficient_templates: dict[int, torch.Tensor],
        *,
        multiplier_min: float = 1e-4,
        multiplier_max: float = 10.0,
        initial_scalar: float = 1.0,
        checkpoint_chunks: bool = False,
    ) -> None:
        super().__init__(
            prepared,
            mode="per_group",
            multiplier_min=multiplier_min,
            multiplier_max=multiplier_max,
            initial_multiplier=initial_scalar,
            checkpoint_chunks=checkpoint_chunks,
        )
        expected_sizes = set(prepared._selected_by_size)
        if set(coefficient_templates) != expected_sizes:
            raise ValueError(
                "coefficient template group sizes must match the preparation"
            )
        self._template_buffer_names: dict[int, str] = {}
        for group_size in sorted(expected_sizes):
            template = torch.as_tensor(
                coefficient_templates[group_size],
                device=prepared.device,
                dtype=prepared.dtype,
            )
            expected_shape = (
                group_size,
                prepared.patch_size,
                prepared.patch_size,
            )
            if template.shape != expected_shape:
                raise ValueError(
                    f"G={group_size} template shape {tuple(template.shape)} "
                    f"does not match {expected_shape}"
                )
            if not bool(torch.isfinite(template).all()) or bool((template <= 0).any()):
                raise ValueError("all template coefficients must be finite and positive")
            name = f"coefficient_template_g{group_size}"
            self.register_buffer(name, template.contiguous())
            self._template_buffer_names[group_size] = name

    def coefficient_template(self, group_size: int) -> torch.Tensor:
        """Return the fixed ``[G,K,K]`` multiplier template."""
        try:
            name = self._template_buffer_names[group_size]
        except KeyError as error:
            raise ValueError(f"no coefficient template for G={group_size}") from error
        return getattr(self, name)

    def forward(self) -> torch.Tensor:
        """Render with bounded template-times-scalar coefficient multipliers."""
        prepared = self.prepared
        numerator = torch.zeros_like(prepared._clean_padded)
        denominator = torch.zeros(
            (prepared.batch_size, prepared._height, prepared._width),
            device=prepared.device,
            dtype=prepared.dtype,
        )
        chunk_size = prepared.config.runtime.group_chunk_size
        flat_scalars = self.group_multipliers().reshape(-1)

        for group_size, selected in prepared._selected_by_size.items():
            template = self.coefficient_template(group_size)
            for start in range(0, selected.numel(), chunk_size):
                chosen = selected[start:start + chunk_size]
                selected_matches = prepared._global_matches[chosen, :group_size]
                multipliers = (
                    flat_scalars[chosen, None, None, None] * template[None]
                ).clamp(self.multiplier_min, self.multiplier_max)
                weighted_patches, denominator_windows = (
                    self._maybe_weighted_reconstruct_chunk(
                        selected_matches,
                        multipliers,
                        coefficient_mode=True,
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
