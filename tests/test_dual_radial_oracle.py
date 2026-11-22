from __future__ import annotations

from dataclasses import replace

import torch

from bm3d_tensor.config import BM3DConfig, RuntimeConfig
from bm3d_tensor.dual_radial_oracle import BM3DDualRadialOracle
from bm3d_tensor.precomputed import BM3DWienerPrecomputed


def prepared_image() -> BM3DWienerPrecomputed:
    image = torch.linspace(0, 255, 48 * 48, dtype=torch.float64).reshape(48, 48)
    config = replace(
        BM3DConfig.classic(20.0),
        runtime=RuntimeConfig(
            reference_chunk_size=64,
            displacement_chunk_size=32,
            group_chunk_size=64,
        ),
    )
    return BM3DWienerPrecomputed(
        image,
        20.0,
        config,
        cache_error_targets=False,
    )


def test_dual_radial_has_exact_budget_matches_x1_and_backpropagates() -> None:
    prepared = prepared_image()
    oracle = BM3DDualRadialOracle(prepared)

    assert oracle.radial_log_vectors().shape == (
        1,
        prepared.reference_count,
        2,
        8,
    )
    assert oracle.parameter_count == 16 * prepared.reference_count
    torch.testing.assert_close(oracle(), prepared.run(1.0), rtol=1e-12, atol=1e-12)

    oracle().square().mean().backward()
    gradient = oracle.radial_log_vectors().grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient[..., 0, :] != 0).any())
    assert bool((gradient[..., 1, :] != 0).any())


def test_dual_radial_expands_h0_and_one_shared_h1_plus_surface() -> None:
    prepared = prepared_image()
    oracle = BM3DDualRadialOracle(prepared)
    group_size, selected = next(
        (size, indices)
        for size, indices in prepared._selected_by_size.items()
        if size > 1 and indices.numel()
    )
    chosen = selected[:1]

    with torch.no_grad():
        logs = oracle.radial_log_vectors().reshape(-1, 2, 8)
        logs[chosen, 0] = torch.linspace(0.0, 0.7, 8, dtype=prepared.dtype)
        logs[chosen, 1] = torch.linspace(0.0, -0.7, 8, dtype=prepared.dtype)
    expanded = oracle.expanded_group_multipliers(chosen, group_size)

    assert expanded.shape == (1, group_size, 8, 8)
    torch.testing.assert_close(
        expanded[:, 1:],
        expanded[:, 1:2].expand_as(expanded[:, 1:]),
    )
    assert not torch.equal(expanded[:, 0], expanded[:, 1])
