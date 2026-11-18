# BM3D noisy-clean dataset optimizer

This repository contains only the code needed to optimize BM3D Wiener-stage
multipliers from clean images and synthetic Gaussian-noise realizations. It
processes every supported image in a folder, creates a separate image-specific
model, streams realizations one at a time, validates on unseen seeds, and saves
restartable checkpoints.

The optimization target is the clean image. For each realization the runner
adds Gaussian noise with the requested standard deviation, runs the fixed BM3D
hard stage, reuses a canonical clean-image match topology, and updates the
Wiener multipliers to minimize output MSE. A coefficient map cannot be shared
directly between differently sized/content images because its parameters are
aligned to that image's BM3D reference groups.

## Set up

Python 3.10 or newer is recommended. On a CUDA machine, install the appropriate
PyTorch build for that machine first; then install the remaining requirements.

```bash
git clone https://github.com/alonhelvits/bm3d-noisy-clean-optimizer.git
cd bm3d-noisy-clean-optimizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place clean images in a directory outside this repository, or in `data/`
(which is git-ignored). Outputs must be outside the input directory so preview
PNGs cannot later be interpreted as clean inputs.

## Run a full folder

```bash
python optimize_dataset.py /path/to/clean_images \
  --output-dir /path/to/results \
  --device cuda \
  --sigma 20 \
  --train-realizations 16 \
  --validation-realizations 4 \
  --epochs 10 \
  --batch-size 4
```

The default `--mode per_coefficient` is the most expressive and the most
memory-intensive option. Before committing a long run, use a smoke test:

```bash
python optimize_dataset.py /path/to/clean_images \
  --output-dir /path/to/smoke_results \
  --device cuda \
  --limit 1 \
  --crop-size 96 \
  --train-realizations 1 \
  --validation-realizations 1 \
  --epochs 1 \
  --batch-size 1 \
  --mode per_group
```

Then remove `--crop-size`, `--limit`, and the smoke-test settings for the real
full-resolution run.

## Important controls

- `--sigma`: Gaussian noise standard deviation in 0-255 pixel units.
- `--train-realizations` / `--validation-realizations`: noise draws per image.
- `--train-seed-start`, `--validation-seed-start`, and
  `--image-seed-stride`: make every realization deterministic and keep image
  noise streams separate.
- `--clip-noisy` / `--no-clip-noisy`: whether noisy pixels are clipped to
  0-255 before BM3D.
- `--mode`: `per_coefficient`, `per_radial_row`, `per_square_shell`, or
  `per_group`. The latter three reduce parameter count in that order.
- `--epochs`, `--batch-size`, `--learning-rate`: optimization schedule. The
  default large learning rate matches the summed pixel loss used by the
  coefficient experiment; retune it when changing parameterization or image
  scale substantially.
- `--multiplier-min`, `--multiplier-max`, `--initial-multiplier`: projected
  Wiener multiplier interval and initialization.
- `--reference-chunk-size`, `--displacement-chunk-size`, and
  `--group-chunk-size`: memory/performance controls. Lower them after an
  out-of-memory error; they do not alter BM3D's mathematical configuration.
- `--activation-checkpointing`: enabled by default to reduce training memory.
- `--recursive`, `--extensions`, and `--limit`: dataset discovery controls.

Run `python optimize_dataset.py --help` for every option and its default.

## Resume interrupted work

Each image writes `restart_checkpoint.pt` at the end of every epoch by default.
Repeat the same command with `--resume`:

```bash
python optimize_dataset.py /path/to/clean_images \
  --output-dir /path/to/results \
  --device cuda \
  --sigma 20 \
  --train-realizations 16 \
  --validation-realizations 4 \
  --epochs 10 \
  --batch-size 4 \
  --resume
```

Completed images are skipped by default. Resume validates the source image,
seeds, noise settings, optimizer settings, parameterization, bounds, and BM3D
configuration against the checkpoint rather than silently mixing experiments.
To extend a completed run to more epochs, use `--no-skip-completed --resume`
and set the larger `--epochs` value.

## Outputs

The output tree mirrors the input tree. Each image directory contains:

- `optimized_model.pt`: final model weights plus configuration and seeds.
- `restart_checkpoint.pt`: model and optimizer state for resume.
- `optimization_history.csv`: update-level training diagnostics.
- `validation_metrics.csv`: fixed BM3D versus learned metrics for unseen noise.
- `clean.png`, `noisy.png`, `fixed_x1.png`, and `learned.png`: first validation
  realization previews.
- `summary.json`: settings, timing, parameter counts, and mean validation gain.

At dataset level, `run_settings.json` records the resolved run and
`dataset_summary.json` is updated after every image. Use
`--continue-on-error` when one bad image should not stop the rest of the
folder.

Large inputs can require substantial accelerator memory and runtime. The
runner is intentionally sequential across images and realizations, making
restart behavior predictable and bounding cached image data.
