from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from optimize_dataset import image_metrics, noisy_realization


ROOT = Path(__file__).resolve().parents[1]


def test_noise_is_reproducible_and_metrics_are_exact_for_identity() -> None:
    clean = np.arange(48 * 48, dtype=np.float32).reshape(48, 48) % 256
    first = noisy_realization(clean, 20.0, 123, True)
    second = noisy_realization(clean, 20.0, 123, True)

    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0
    assert first.max() <= 255
    metrics = image_metrics(clean, clean)
    assert metrics == {"psnr_db": float("inf"), "ssim": 1.0, "mae": 0.0, "mse": 0.0}


def test_cli_dual_radial_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows, columns = np.indices((48, 48))
    clean = ((3 * rows + 5 * columns) % 256).astype(np.uint8)
    assert cv2.imwrite(str(input_dir / "clean.png"), clean)
    output_dir = tmp_path / "results"

    command = [
        sys.executable,
        str(ROOT / "optimize_dataset.py"),
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--train-realizations",
        "1",
        "--validation-realizations",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--validation-every",
        "1",
        "--reference-chunk-size",
        "64",
        "--displacement-chunk-size",
        "32",
        "--group-chunk-size",
        "64",
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    summary_path = output_dir / "clean_png" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["mode"] == "dual_radial"
    assert summary["optimizer"] == "adam"
    assert summary["learning_rate"] == 0.2
    assert summary["parameter_count"] == 16 * summary["reference_count"]
    assert summary["parameterization"]["h0_parameters"] == 8
    assert summary["parameterization"]["h1_plus_parameters"] == 8
    assert len(summary["validation_history"]) == 2
    for filename in (
        "best_checkpoint.pt",
        "optimized_log_radial_profiles_float32.npy",
        "optimized_model.pt",
        "restart_checkpoint.pt",
        "validation_history.csv",
    ):
        assert (summary_path.parent / filename).is_file()
