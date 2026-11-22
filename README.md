# BM3D noisy-clean dataset optimizer

This repository contains only the code needed to optimize BM3D Wiener-stage
multipliers from clean images and synthetic Gaussian-noise realizations. It
processes every supported image in a folder, creates a separate image-specific
model, streams realizations one at a time, validates on unseen seeds, and saves
restartable checkpoints.

The optimization target is the clean image. For each realization the runner
adds Gaussian noise with the requested standard deviation, runs the fixed BM3D
hard stage, reuses a canonical clean-image match topology, and updates the
Wiener multipliers to minimize output MSE. Each image gets its own model because
the parameters are aligned to that image's BM3D reference locations.

The default `dual_radial` model is the project's primary parameterization. Each
reference location owns exactly 16 log multipliers: eight radial DCT values for
Hadamard component H0 and eight radial values shared by every H1+ component.
The eight-value rows expand to symmetric 8x8 surfaces by Euclidean-radius
interpolation. This is 16 values **per reference location**, not 16 values for
the complete image.

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
  --mode dual_radial \
  --sigma 20 \
  --train-realizations 8 \
  --validation-realizations 4 \
  --epochs 12 \
  --batch-size 2 \
  --optimizer adam \
  --learning-rate 0.2 \
  --validation-every 2
```

Those are also the defaults for the optimization-related flags and reproduce
the established 512x512 experiment. If compute permits, a higher-confidence
run can use 16 training realizations, 8 validation realizations, 24 epochs, and
batch size 4. Periodic held-out validation always selects the best epoch, so a
later overfit epoch does not replace a better model.

Before committing a long run, use a smoke test:

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
  --validation-every 1
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
- `--mode`: `dual_radial` (the default 8×H0 + 8×shared-H1+ model),
  `per_coefficient`, `per_radial_row`, `per_square_shell`, or `per_group`.
- `--optimizer`: `auto` chooses Adam for `dual_radial` and SGD for the legacy
  modes. It can be overridden explicitly.
- `--epochs`, `--batch-size`, `--learning-rate`: optimization schedule. The
  tested dual-radial values are Adam, learning rate 0.2, batch size 2, and 12
  epochs.
- `--validation-every`: held-out evaluation interval. The best validation PSNR
  checkpoint is exported as the final model.
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
  --train-realizations 8 \
  --validation-realizations 4 \
  --epochs 12 \
  --batch-size 2 \
  --resume
```

Completed images are skipped by default. Resume validates the source image,
seeds, noise settings, optimizer settings, parameterization, bounds, and BM3D
configuration against the checkpoint rather than silently mixing experiments.
To extend a completed run to more epochs, use `--no-skip-completed --resume`
and set the larger `--epochs` value.

## Outputs

The output tree mirrors the input tree. Each image directory contains:

- `optimized_model.pt`: best-held-out model weights plus configuration and
  seeds.
- `restart_checkpoint.pt`: model and optimizer state for resume.
- `best_checkpoint.pt`: best held-out model and its selected epoch.
- `optimized_log_radial_profiles_float32.npy`: the selected `[1,R,2,8]`
  dual-radial log multipliers.
- `optimization_history.csv`: update-level training diagnostics.
- `validation_history.csv`: epoch-level held-out PSNR, SSIM, MAE, and MSE.
- `best_validation_metrics.csv`: per-seed metrics for the selected model.
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

## Tests and 512x512 regression

Install the development requirements and run the fast suite with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite checks the exact 16-parameter budget, equality to standard BM3D at
initialization, finite gradients in both H0 and H1+, shared H1+ expansion,
deterministic noise, and an end-to-end dataset run. GitHub Actions runs it on
every push and pull request.

The standalone runner was also exercised on the exact archived 512x512 DIV2K
crop. It reproduced `31.94684594 dB` versus the archived `31.94684596 dB`, with
the same selected epoch 12 and approximately `+1.02004 dB` over fixed BM3D.
See [REPRODUCTION.md](REPRODUCTION.md) for the full command and comparison.
