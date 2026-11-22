#!/usr/bin/env python3
"""Optimize BM3D Wiener multipliers for every clean image in a folder."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import cv2
import numpy as np
import torch

from bm3d_tensor.config import BM3DConfig, RuntimeConfig
from bm3d_tensor.dual_radial_oracle import BM3DDualRadialOracle
from bm3d_tensor.matching import MatchResult
from bm3d_tensor.oracle import BM3DMultiplierOracle
from bm3d_tensor.precomputed import BM3DWienerPrecomputed


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
METRIC_NAMES = ("psnr_db", "ssim", "mae", "mse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("input_dir", type=Path, help="Folder of clean images.")
    data.add_argument("--output-dir", type=Path, default=Path("outputs"))
    data.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument(
        "--extensions",
        nargs="+",
        default=SUPPORTED_EXTENSIONS,
        help="Image suffixes to include (case-insensitive).",
    )
    data.add_argument("--limit", type=int, default=0, help="Zero processes all images.")
    data.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Optional centered square crop; zero uses the full image.",
    )

    noise = parser.add_argument_group("noise realizations")
    noise.add_argument("--sigma", type=float, default=20.0, help="Gaussian noise std in 0-255 units.")
    noise.add_argument("--train-realizations", type=int, default=8)
    noise.add_argument("--validation-realizations", type=int, default=4)
    noise.add_argument("--train-seed-start", type=int, default=0)
    noise.add_argument("--validation-seed-start", type=int, default=1_000)
    noise.add_argument(
        "--image-seed-stride",
        type=int,
        default=10_000,
        help="Added per image so different images do not reuse noise arrays.",
    )
    noise.add_argument(
        "--clip-noisy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip noisy realizations to the valid 0-255 image range.",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument(
        "--mode",
        choices=(
            "dual_radial",
            "per_coefficient",
            "per_radial_row",
            "per_square_shell",
            "per_group",
        ),
        default="dual_radial",
        help=(
            "Multiplier parameterization; dual_radial learns 8 H0 plus 8 shared "
            "H1+ radial values at every reference location."
        ),
    )
    optimization.add_argument("--epochs", type=int, default=12)
    optimization.add_argument("--batch-size", type=int, default=2, help="Gradient accumulation seeds per update.")
    optimization.add_argument(
        "--optimizer",
        choices=("auto", "adam", "sgd"),
        default="auto",
        help="Auto selects Adam for dual_radial and SGD for legacy modes.",
    )
    optimization.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Auto default: 0.2 for dual_radial, 500000 for legacy modes.",
    )
    optimization.add_argument(
        "--validation-every",
        type=int,
        default=2,
        help="Evaluate held-out seeds and update the best checkpoint every N epochs.",
    )
    optimization.add_argument("--multiplier-min", type=float, default=1e-4)
    optimization.add_argument("--multiplier-max", type=float, default=10.0)
    optimization.add_argument("--initial-multiplier", type=float, default=1.0)
    optimization.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save a restart checkpoint every N epochs; zero saves only the final state.",
    )
    optimization.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade extra compute for lower backward-pass memory.",
    )

    runtime = parser.add_argument_group("hardware and memory")
    runtime.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    runtime.add_argument("--reference-chunk-size", type=int, default=1024)
    runtime.add_argument("--displacement-chunk-size", type=int, default=64)
    runtime.add_argument("--group-chunk-size", type=int, default=512)

    control = parser.add_argument_group("run control")
    control.add_argument("--resume", action="store_true", help="Resume each image from its checkpoint when present.")
    control.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip image folders containing a completed summary.",
    )
    control.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_optimization_defaults(args: argparse.Namespace) -> None:
    """Resolve mode-dependent optimizer settings before validation/recording."""
    if args.optimizer == "auto":
        args.optimizer = "adam" if args.mode == "dual_radial" else "sgd"
    if args.learning_rate is None:
        args.learning_rate = 0.2 if args.mode == "dual_radial" else 500_000.0


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("--sigma must be finite and positive")
    if args.train_realizations <= 0 or args.validation_realizations < 0:
        raise ValueError("training realizations must be positive and validation realizations nonnegative")
    if args.epochs <= 0 or args.batch_size <= 0 or args.validation_every <= 0:
        raise ValueError("--epochs, --batch-size, and --validation-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    bounds = (args.multiplier_min, args.initial_multiplier, args.multiplier_max)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("multiplier values must be finite")
    if not 0 < args.multiplier_min <= args.initial_multiplier <= args.multiplier_max:
        raise ValueError("require 0 < min <= initial <= max")
    if args.multiplier_min == args.multiplier_max:
        raise ValueError("multiplier min and max must differ")
    if args.crop_size < 0 or args.limit < 0 or args.checkpoint_every < 0:
        raise ValueError("crop size, limit, and checkpoint interval cannot be negative")
    if args.image_seed_stride < 0:
        raise ValueError("--image-seed-stride cannot be negative")
    chunks = (
        args.reference_chunk_size,
        args.displacement_chunk_size,
        args.group_chunk_size,
    )
    if min(chunks) <= 0:
        raise ValueError("all chunk sizes must be positive")


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch installation")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch installation")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def discover_images(args: argparse.Namespace) -> list[Path]:
    extensions = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in args.extensions
    }
    iterator = args.input_dir.rglob("*") if args.recursive else args.input_dir.glob("*")
    paths = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.relative_to(args.input_dir).as_posix().casefold(),
    )
    return paths[: args.limit or None]


def output_path_for(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    leaf = f"{relative.stem}_{relative.suffix.lstrip('.').lower()}"
    return output_dir.joinpath(*relative.parts[:-1], leaf)


def load_clean_image(path: Path, crop_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    if crop_size:
        height, width = image.shape
        if crop_size > min(height, width):
            raise ValueError(
                f"crop size {crop_size} exceeds {path}'s {height}x{width} dimensions"
            )
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        image = image[top : top + crop_size, left : left + crop_size]
    return image.astype(np.float32)


def noisy_realization(clean: np.ndarray, sigma: float, seed: int, clip: bool) -> np.ndarray:
    noisy = clean + np.random.default_rng(seed).normal(0.0, sigma, size=clean.shape)
    if clip:
        noisy = np.clip(noisy, 0.0, 255.0)
    return noisy.astype(np.float32)


def image_metrics(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    difference = clean.astype(np.float64) - estimate.astype(np.float64)
    mse = float(np.mean(difference**2))
    reference = clean.astype(np.float64)
    candidate = estimate.astype(np.float64)
    window = cv2.getGaussianKernel(11, 1.5)
    window = window @ window.T
    mean_reference = cv2.filter2D(reference, -1, window)
    mean_candidate = cv2.filter2D(candidate, -1, window)
    mean_reference_sq = np.square(mean_reference)
    mean_candidate_sq = np.square(mean_candidate)
    mean_product = mean_reference * mean_candidate
    variance_reference = cv2.filter2D(np.square(reference), -1, window) - mean_reference_sq
    variance_candidate = cv2.filter2D(np.square(candidate), -1, window) - mean_candidate_sq
    covariance = cv2.filter2D(reference * candidate, -1, window) - mean_product
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ssim_map = ((2 * mean_product + c1) * (2 * covariance + c2)) / (
        (mean_reference_sq + mean_candidate_sq + c1)
        * (variance_reference + variance_candidate + c2)
    )
    return {
        "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        "ssim": float(np.mean(ssim_map[5:-5, 5:-5])),
        "mae": float(np.mean(np.abs(difference))),
        "mse": mse,
    }


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return np.clip(image.detach().cpu().numpy(), 0.0, 255.0)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def release_preparation(prepared: BM3DWienerPrecomputed, device: torch.device) -> None:
    for name in (
        "_image",
        "_basic_bchw",
        "basic",
        "_clean_padded",
        "_basic_padded",
        "_noisy_bank",
        "_basic_bank",
        "_clean_spatial_bank",
    ):
        setattr(prepared, name, None)
    del prepared
    gc.collect()
    empty_device_cache(device)


def prepare_seed(
    clean: np.ndarray,
    sigma: float,
    seed: int,
    clip_noisy: bool,
    config: BM3DConfig,
    matches: MatchResult,
    device: torch.device,
) -> tuple[BM3DWienerPrecomputed, float]:
    start = perf_counter()
    noisy = torch.from_numpy(noisy_realization(clean, sigma, seed, clip_noisy)).to(device)
    prepared = BM3DWienerPrecomputed(
        noisy,
        sigma,
        config,
        canonical_matches=matches,
        cache_error_targets=False,
    )
    synchronize(device)
    return prepared, perf_counter() - start


def checkpoint_payload(
    oracle: BM3DMultiplierOracle,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "completed_epoch": completed_epoch,
        "mode": metadata["mode"],
        "model": {name: value.detach().cpu() for name, value in oracle.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
    }


def save_checkpoint(
    path: Path,
    oracle: BM3DMultiplierOracle,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    metadata: dict[str, object],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(oracle, optimizer, completed_epoch, metadata), temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    oracle: BM3DMultiplierOracle,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict[str, object],
) -> int:
    checkpoint = torch.load(path, map_location=next(oracle.parameters()).device, weights_only=False)
    actual = checkpoint.get("metadata", {})
    for key in (
        "source",
        "source_sha256",
        "shape",
        "sigma",
        "mode",
        "multiplier_min",
        "multiplier_max",
        "initial_multiplier",
        "clip_noisy",
        "train_seeds",
        "validation_seeds",
        "batch_size",
        "optimizer",
        "learning_rate",
        "validation_every",
        "config",
    ):
        if actual.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"checkpoint metadata mismatch for {key}: {actual.get(key)!r} "
                f"!= {expected_metadata.get(key)!r}"
            )
    oracle.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["completed_epoch"])


def mean_metrics(rows: list[dict[str, object]], prefix: str) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[f"{prefix}_{name}"]) for row in rows]))
        for name in METRIC_NAMES
    }


def create_oracle(
    prepared: BM3DWienerPrecomputed,
    args: argparse.Namespace,
) -> BM3DMultiplierOracle:
    common = {
        "multiplier_min": args.multiplier_min,
        "multiplier_max": args.multiplier_max,
        "initial_multiplier": args.initial_multiplier,
        "checkpoint_chunks": args.activation_checkpointing,
    }
    if args.mode == "dual_radial":
        return BM3DDualRadialOracle(prepared, **common)
    return BM3DMultiplierOracle(prepared, mode=args.mode, **common)


def create_optimizer(
    oracle: BM3DMultiplierOracle,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(
            oracle.parameters(),
            lr=args.learning_rate,
            foreach=False,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            oracle.parameters(),
            lr=args.learning_rate,
            momentum=0,
            foreach=False,
        )
    raise ValueError(f"unsupported optimizer: {args.optimizer}")


def profile_statistics(
    log_profiles: np.ndarray,
    multiplier_min: float,
    multiplier_max: float,
) -> dict[str, object]:
    """Summarize a ``[B,R,2,8]`` dual-radial log-multiplier map."""
    if log_profiles.ndim != 4 or log_profiles.shape[-2:] != (2, 8):
        raise ValueError(
            "dual-radial profiles must have [B,R,2,8] shape; "
            f"got {log_profiles.shape}"
        )
    multipliers = np.exp(log_profiles.astype(np.float64))
    result: dict[str, object] = {}
    for index, name in enumerate(("h0", "h1_plus")):
        values = multipliers[..., index, :]
        logs = log_profiles[..., index, :].astype(np.float64)
        second = logs[..., 2:] - 2 * logs[..., 1:-1] + logs[..., :-2]
        result[name] = {
            "arithmetic_mean": float(values.mean()),
            "geometric_mean": float(np.exp(logs.mean())),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "fraction_below_1": float(np.mean(values < 1.0)),
            "fraction_above_1": float(np.mean(values > 1.0)),
            "fraction_at_floor": float(
                np.mean(values <= multiplier_min * (1 + 1e-6))
            ),
            "fraction_at_ceiling": float(
                np.mean(values >= multiplier_max * (1 - 1e-6))
            ),
            "log_second_difference_rms": float(
                np.sqrt(np.mean(np.square(second)))
            ),
            "mean_vector": values.mean(axis=(0, 1)).tolist(),
            "median_vector": np.median(values, axis=(0, 1)).tolist(),
        }
    return result


def save_best_checkpoint(
    path: Path,
    oracle: BM3DMultiplierOracle,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "selected_epoch": epoch,
            "metrics": metrics,
            "model": {
                name: value.detach().cpu().clone()
                for name, value in oracle.state_dict().items()
            },
        },
        temporary,
    )
    temporary.replace(path)


def train(
    oracle: BM3DMultiplierOracle,
    optimizer: torch.optim.Optimizer,
    clean: np.ndarray,
    clean_tensor: torch.Tensor,
    config: BM3DConfig,
    canonical_matches: MatchResult,
    train_seeds: list[int],
    args: argparse.Namespace,
    device: torch.device,
    image_dir: Path,
    checkpoint_metadata: dict[str, object],
    start_epoch: int,
    on_epoch_end: Callable[[int], None] | None = None,
) -> list[dict[str, object]]:
    history_path = image_dir / "optimization_history.csv"
    history: list[dict[str, object]] = []
    if start_epoch and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as file:
            history = list(csv.DictReader(file))
    update = int(history[-1]["update"]) if history else 0
    run_start = perf_counter()

    for epoch in range(start_epoch, args.epochs):
        completed_epoch = epoch + 1
        order = np.random.default_rng(
            90_000 + completed_epoch
        ).permutation(train_seeds).tolist()
        for batch_start in range(0, len(order), args.batch_size):
            batch = order[batch_start : batch_start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_mse = 0.0
            preparation_seconds = 0.0
            oracle.train()
            for seed in batch:
                prepared, elapsed = prepare_seed(
                    clean,
                    args.sigma,
                    seed,
                    args.clip_noisy,
                    config,
                    canonical_matches,
                    device,
                )
                preparation_seconds += elapsed
                oracle.use_prepared_(prepared)
                estimate = oracle()
                normalized_error = ((estimate - clean_tensor) / 255.0).square()
                mse = normalized_error.detach().mean()
                if not bool(torch.isfinite(mse)):
                    raise RuntimeError(f"non-finite loss for seed {seed}")
                (normalized_error.sum() / len(batch)).backward()
                batch_mse += float(mse)
                del estimate, normalized_error, mse
                release_preparation(prepared, device)

            invalid_gradients = 0
            gradient_count = 0
            for parameter in oracle.parameters():
                if parameter.grad is None:
                    continue
                finite = torch.isfinite(parameter.grad)
                gradient_count += parameter.grad.numel()
                invalid_gradients += int((~finite).sum())
                if not bool(finite.all()):
                    parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.step()
            oracle.project_()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            update += 1
            row: dict[str, object] = {
                "epoch": epoch + 1,
                "update": update,
                "batch_seeds": " ".join(map(str, batch)),
                "training_psnr_before_update_db": -10.0 * math.log10(batch_mse / len(batch)),
                "preparation_seconds": preparation_seconds,
                "invalid_gradient_count": invalid_gradients,
                "invalid_gradient_fraction": invalid_gradients / max(gradient_count, 1),
                "elapsed_seconds_this_invocation": perf_counter() - run_start,
            }
            history.append(row)
            write_csv(history_path, history)
            print(
                f"    epoch {epoch + 1}/{args.epochs}, update {update}, "
                f"seeds={batch}, PSNR={row['training_psnr_before_update_db']:.4f} dB, "
                f"prep={preparation_seconds:.1f}s, bad_grad={invalid_gradients}",
                flush=True,
            )

        if args.checkpoint_every and completed_epoch % args.checkpoint_every == 0:
            save_checkpoint(
                image_dir / "restart_checkpoint.pt",
                oracle,
                optimizer,
                completed_epoch,
                checkpoint_metadata,
            )
        if on_epoch_end is not None:
            on_epoch_end(completed_epoch)
    return history


@torch.no_grad()
def evaluate(
    oracle: BM3DMultiplierOracle,
    clean: np.ndarray,
    config: BM3DConfig,
    canonical_matches: MatchResult,
    validation_seeds: list[int],
    args: argparse.Namespace,
    device: torch.device,
    image_dir: Path,
    *,
    include_fixed: bool = True,
    metrics_filename: str = "validation_metrics.csv",
    save_previews: bool = False,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, float]]:
    rows: list[dict[str, object]] = []
    oracle.eval()
    for index, seed in enumerate(validation_seeds, start=1):
        print(f"    validation seed {seed} ({index}/{len(validation_seeds)})", flush=True)
        prepared, elapsed = prepare_seed(
            clean,
            args.sigma,
            seed,
            args.clip_noisy,
            config,
            canonical_matches,
            device,
        )
        oracle.use_prepared_(prepared)
        learned = tensor_to_numpy(oracle())
        learned_metrics = image_metrics(clean, learned)
        row: dict[str, object] = {
            "seed": seed,
            "preparation_seconds": elapsed,
            **{f"learned_{key}": value for key, value in learned_metrics.items()},
        }
        fixed: np.ndarray | None = None
        if include_fixed:
            fixed = tensor_to_numpy(prepared.run(1.0))
            fixed_metrics = image_metrics(clean, fixed)
            row.update(
                {f"fixed_{key}": value for key, value in fixed_metrics.items()}
            )
        rows.append(row)
        if index == 1 and save_previews:
            images = {
                "clean.png": clean,
                "noisy.png": noisy_realization(clean, args.sigma, seed, args.clip_noisy),
                "learned.png": learned,
            }
            if fixed is not None:
                images["fixed_x1.png"] = fixed
            for filename, image in images.items():
                ok = cv2.imwrite(str(image_dir / filename), np.rint(np.clip(image, 0, 255)).astype(np.uint8))
                if not ok:
                    raise OSError(f"failed to write {image_dir / filename}")
        release_preparation(prepared, device)
    write_csv(image_dir / metrics_filename, rows)
    if not rows:
        return rows, {}, {}
    fixed_mean = mean_metrics(rows, "fixed") if include_fixed else {}
    return rows, fixed_mean, mean_metrics(rows, "learned")


def process_image(
    image_path: Path,
    image_index: int,
    args: argparse.Namespace,
    config: BM3DConfig,
    device: torch.device,
) -> dict[str, object]:
    relative = image_path.relative_to(args.input_dir).as_posix()
    image_dir = output_path_for(image_path, args.input_dir, args.output_dir)
    summary_path = image_dir / "summary.json"
    if args.skip_completed and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            print(f"[{image_index + 1}] skipping completed {relative}", flush=True)
            return {"source": relative, "status": "skipped_complete", "output": str(image_dir)}
    image_dir.mkdir(parents=True, exist_ok=True)
    clean = load_clean_image(image_path, args.crop_size)
    clean_tensor = torch.from_numpy(clean).to(device)
    seed_offset = image_index * args.image_seed_stride
    train_seeds = list(
        range(
            args.train_seed_start + seed_offset,
            args.train_seed_start + seed_offset + args.train_realizations,
        )
    )
    validation_seeds = list(
        range(
            args.validation_seed_start + seed_offset,
            args.validation_seed_start + seed_offset + args.validation_realizations,
        )
    )
    metadata: dict[str, object] = {
        "source": relative,
        "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "shape": list(clean.shape),
        "sigma": args.sigma,
        "mode": args.mode,
        "multiplier_min": args.multiplier_min,
        "multiplier_max": args.multiplier_max,
        "initial_multiplier": args.initial_multiplier,
        "clip_noisy": args.clip_noisy,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "validation_every": args.validation_every,
        "config": asdict(config),
    }

    print(
        f"[{image_index + 1}] {relative}: shape={clean.shape}, device={device}, "
        f"train={len(train_seeds)}, validation={len(validation_seeds)}",
        flush=True,
    )
    started = perf_counter()
    canonical_start = perf_counter()
    canonical = BM3DWienerPrecomputed(
        clean_tensor,
        args.sigma,
        config,
        cache_error_targets=False,
    )
    synchronize(device)
    canonical_seconds = perf_counter() - canonical_start
    canonical_matches = canonical.matches
    reference_count = canonical.reference_count
    reference_grid_shape = list(canonical.reference_grid_shape)
    group_counts = {
        str(group_size): int(selected.numel())
        for group_size, selected in canonical._selected_by_size.items()
    }
    oracle = create_oracle(canonical, args)
    parameter_count = oracle.parameter_count
    if args.mode == "dual_radial":
        expected_parameter_count = 16 * reference_count
        if parameter_count != expected_parameter_count:
            raise RuntimeError(
                "dual_radial must have exactly 16 parameters per reference; "
                f"got {parameter_count} parameters for {reference_count} "
                "references"
            )
    release_preparation(canonical, device)
    optimizer = create_optimizer(oracle, args)

    checkpoint_path = image_dir / "restart_checkpoint.pt"
    best_checkpoint_path = image_dir / "best_checkpoint.pt"
    validation_history_path = image_dir / "validation_history.csv"
    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        start_epoch = load_checkpoint(checkpoint_path, oracle, optimizer, metadata)
        if start_epoch > args.epochs:
            raise ValueError(
                f"checkpoint has {start_epoch} completed epochs but --epochs is {args.epochs}"
            )
        print(f"    resumed after epoch {start_epoch}", flush=True)

    validation_history: list[dict[str, object]] = []
    if args.resume and validation_history_path.exists():
        with validation_history_path.open(newline="", encoding="utf-8") as file:
            for raw in csv.DictReader(file):
                validation_history.append(
                    {
                        "epoch": int(raw["epoch"]),
                        **{
                            name: float(raw[name])
                            for name in METRIC_NAMES
                        },
                        "delta_psnr_from_fixed_x1": float(
                            raw["delta_psnr_from_fixed_x1"]
                        ),
                    }
                )

    fixed_metrics: dict[str, float] = {}
    initial_metrics: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    best_epoch = start_epoch
    if validation_seeds:
        _, fixed_metrics, current_metrics = evaluate(
            oracle,
            clean,
            config,
            canonical_matches,
            validation_seeds,
            args,
            device,
            image_dir,
            include_fixed=True,
            metrics_filename=(
                "resume_validation_metrics.csv"
                if start_epoch
                else "start_validation_metrics.csv"
            ),
            save_previews=True,
        )
        current_row = {
            "epoch": start_epoch,
            **current_metrics,
            "delta_psnr_from_fixed_x1": (
                current_metrics["psnr_db"] - fixed_metrics["psnr_db"]
            ),
        }
        if not validation_history:
            validation_history.append(current_row)
        elif int(validation_history[-1]["epoch"]) < start_epoch:
            validation_history.append(current_row)
        write_csv(validation_history_path, validation_history)
        initial_metrics = {
            name: float(validation_history[0][name]) for name in METRIC_NAMES
        }

        if args.resume and best_checkpoint_path.exists():
            saved_best = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            best_epoch = int(saved_best["selected_epoch"])
            best_metrics = {
                name: float(saved_best["metrics"][name]) for name in METRIC_NAMES
            }
            del saved_best
            if current_metrics["psnr_db"] > best_metrics["psnr_db"]:
                best_epoch = start_epoch
                best_metrics = current_metrics
                save_best_checkpoint(
                    best_checkpoint_path,
                    oracle,
                    best_epoch,
                    best_metrics,
                )
        else:
            best_epoch = start_epoch
            best_metrics = current_metrics
            save_best_checkpoint(
                best_checkpoint_path,
                oracle,
                best_epoch,
                best_metrics,
            )

    def validate_epoch(completed_epoch: int) -> None:
        nonlocal best_epoch, best_metrics
        if not validation_seeds:
            return
        if (
            completed_epoch % args.validation_every != 0
            and completed_epoch != args.epochs
        ):
            return
        _, _, metrics = evaluate(
            oracle,
            clean,
            config,
            canonical_matches,
            validation_seeds,
            args,
            device,
            image_dir,
            include_fixed=False,
            metrics_filename=f"validation_metrics_epoch_{completed_epoch}.csv",
        )
        row = {
            "epoch": completed_epoch,
            **metrics,
            "delta_psnr_from_fixed_x1": (
                metrics["psnr_db"] - fixed_metrics["psnr_db"]
            ),
        }
        validation_history.append(row)
        write_csv(validation_history_path, validation_history)
        print(
            f"    epoch {completed_epoch}: held-out={metrics['psnr_db']:.5f} dB, "
            f"delta={row['delta_psnr_from_fixed_x1']:+.5f} dB",
            flush=True,
        )
        if not best_metrics or metrics["psnr_db"] > best_metrics["psnr_db"]:
            best_epoch = completed_epoch
            best_metrics = metrics
            save_best_checkpoint(
                best_checkpoint_path,
                oracle,
                best_epoch,
                best_metrics,
            )

    history = train(
        oracle,
        optimizer,
        clean,
        clean_tensor,
        config,
        canonical_matches,
        train_seeds,
        args,
        device,
        image_dir,
        metadata,
        start_epoch,
        validate_epoch,
    )
    save_checkpoint(checkpoint_path, oracle, optimizer, args.epochs, metadata)
    if validation_seeds:
        saved_best = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        oracle.load_state_dict(saved_best["model"])
        best_epoch = int(saved_best["selected_epoch"])
        del saved_best
        _, _, learned_metrics = evaluate(
            oracle,
            clean,
            config,
            canonical_matches,
            validation_seeds,
            args,
            device,
            image_dir,
            include_fixed=False,
            metrics_filename="best_validation_metrics.csv",
            save_previews=True,
        )
        best_metrics = learned_metrics
    else:
        learned_metrics = {}
        best_epoch = args.epochs

    profile_file: str | None = None
    dual_radial_statistics: dict[str, object] | None = None
    if args.mode == "dual_radial":
        if not isinstance(oracle, BM3DDualRadialOracle):
            raise RuntimeError("dual_radial mode constructed the wrong oracle type")
        log_profiles = (
            oracle.radial_log_vectors().detach().cpu().numpy().astype(np.float32)
        )
        profile_file = "optimized_log_radial_profiles_float32.npy"
        np.save(image_dir / profile_file, log_profiles)
        dual_radial_statistics = profile_statistics(
            log_profiles,
            args.multiplier_min,
            args.multiplier_max,
        )
    final_model_path = image_dir / "optimized_model.pt"
    torch.save(
        {
            "format_version": 1,
            "mode": args.mode,
            "model": {name: value.detach().cpu() for name, value in oracle.state_dict().items()},
            "metadata": metadata,
            "config": asdict(config),
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "selected_epoch": best_epoch,
        },
        final_model_path,
    )
    summary = {
        "status": "complete",
        **metadata,
        "device": str(device),
        "clip_noisy": args.clip_noisy,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "validation_every": args.validation_every,
        "parameter_count": parameter_count,
        "reference_count": reference_count,
        "reference_grid_shape": reference_grid_shape,
        "optimizer_updates": len(history),
        "canonical_preparation_seconds": canonical_seconds,
        "total_seconds": perf_counter() - started,
        "canonical_group_counts": group_counts,
        "fixed_x1_validation_mean": fixed_metrics,
        "initial_validation_mean": initial_metrics,
        "selected_epoch": best_epoch,
        "learned_validation_mean": learned_metrics,
        "delta_psnr_db": (
            learned_metrics["psnr_db"] - fixed_metrics["psnr_db"]
            if learned_metrics
            else None
        ),
        "parameterization": (
            {
                "parameters_per_reference": 16,
                "h0_parameters": 8,
                "h1_plus_parameters": 8,
                "h1_plus_shared_across_hadamard_components": True,
                "radial_expansion": (
                    "Euclidean-radius interpolation in log-multiplier space"
                ),
            }
            if args.mode == "dual_radial"
            else {"mode": args.mode}
        ),
        "profile_statistics": dual_radial_statistics,
        "validation_history": validation_history,
        "config": asdict(config),
        "files": {
            "model": final_model_path.name,
            "checkpoint": checkpoint_path.name,
            "history": "optimization_history.csv",
            "best_checkpoint": (
                best_checkpoint_path.name if validation_seeds else None
            ),
            "validation": (
                "best_validation_metrics.csv" if validation_seeds else None
            ),
            "validation_history": (
                validation_history_path.name if validation_seeds else None
            ),
            "radial_profiles": profile_file,
        },
    }
    write_json(summary_path, summary)
    del oracle, optimizer, clean_tensor
    gc.collect()
    empty_device_cache(device)
    return {
        "source": relative,
        "status": "complete",
        "output": str(image_dir),
        "delta_psnr_db": summary["delta_psnr_db"],
        "seconds": summary["total_seconds"],
    }


def main() -> int:
    args = parse_args()
    resolve_optimization_defaults(args)
    validate_args(args)
    images = discover_images(args)
    if not images:
        raise FileNotFoundError(f"no supported images found under {args.input_dir}")
    if args.output_dir.resolve().is_relative_to(args.input_dir.resolve()):
        raise ValueError(
            "--output-dir must be outside --input-dir so generated PNG previews "
            "cannot be mistaken for future clean inputs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    config = replace(
        BM3DConfig.classic(args.sigma, color=False),
        runtime=RuntimeConfig(
            reference_chunk_size=args.reference_chunk_size,
            displacement_chunk_size=args.displacement_chunk_size,
            group_chunk_size=args.group_chunk_size,
        ),
    )
    run_settings = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_device": str(device),
        "image_count": len(images),
        "images": [path.relative_to(args.input_dir).as_posix() for path in images],
        "config": asdict(config),
    }
    write_json(args.output_dir / "run_settings.json", run_settings)
    print(f"Found {len(images)} image(s); using {device}.", flush=True)

    results: list[dict[str, object]] = []
    failures = 0
    for index, image_path in enumerate(images):
        try:
            result = process_image(image_path, index, args, config, device)
        except Exception as error:
            failures += 1
            relative = image_path.relative_to(args.input_dir).as_posix()
            result = {
                "source": relative,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            traceback.print_exc()
            if not args.continue_on_error:
                results.append(result)
                write_json(args.output_dir / "dataset_summary.json", results)
                raise
        results.append(result)
        write_json(args.output_dir / "dataset_summary.json", results)

    print(
        f"Finished: {len(results) - failures} succeeded/skipped, {failures} failed. "
        f"Results: {args.output_dir.resolve()}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
