# 512x512 dual-radial reproduction

The standalone dataset runner was compared with the archived single-image
noisy-clean dual-radial experiment. The input was the exact pixel-identical
crop from `DIV2K/0801.png` at rows 844–1355 and columns 544–1055.

## Configuration

```bash
python optimize_dataset.py /path/to/single_crop_folder \
  --output-dir /path/to/reproduction_results \
  --device mps \
  --mode dual_radial \
  --optimizer adam \
  --sigma 20 \
  --train-realizations 8 \
  --validation-realizations 4 \
  --train-seed-start 0 \
  --validation-seed-start 1000 \
  --epochs 12 \
  --batch-size 2 \
  --learning-rate 0.2 \
  --validation-every 2 \
  --multiplier-min 0.0001 \
  --multiplier-max 10 \
  --checkpoint-every 1
```

Both runs used the classic sigma-20 BM3D configuration, clean-image canonical
matching, clipped Gaussian noise, activation checkpointing, and 8x8 Wiener
patches.

## Results

| Measurement | Archived run | Standalone run | Absolute difference |
|---|---:|---:|---:|
| Fixed BM3D PSNR | 30.926806065 dB | 30.926805409 dB | 0.000000655 dB |
| Learned PSNR | 31.946845956 dB | 31.946845935 dB | 0.000000021 dB |
| PSNR gain | +1.020039892 dB | +1.020040526 dB | 0.000000634 dB |
| Selected epoch | 12 | 12 | 0 |
| Reference locations | 28,561 | 28,561 | 0 |
| Parameters | 456,976 | 456,976 | 0 |

The held-out trajectory also agreed at every two-epoch validation point; the
largest PSNR difference was below 0.000006 dB. All 48 optimizer updates had
zero non-finite gradients. The selected profile file has shape
`[1, 28561, 2, 8]`, confirming 8 H0 and 8 shared-H1+ values per reference.

Small per-parameter differences are expected from accelerator-level floating
point ordering, but the reconstructed-image metrics reproduce the archived
result to substantially better than 0.00001 dB.
