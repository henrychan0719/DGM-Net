# DGM-Net

Official PyTorch implementation of

**Breaking the Resource Wall: Geometry-Guided Sequence Modeling for Efficient Semantic Segmentation**

[![arXiv](https://img.shields.io/badge/arXiv-2604.23399-b31b1b.svg)](https://arxiv.org/abs/2604.23399)

## Overview

DGM-Net (Directional Geometric Mamba Network) is an efficient semantic segmentation framework for resource-constrained settings. SSM-based operators such as Mamba offer linear-complexity long-range modeling, but tend to over-smooth during spatial propagation, degrading boundary quality. DGM-Net reformulates spatial modeling as a **geometry-conditioned propagation process**: explicit geometric priors guide feature aggregation in a structure-aware manner, preserving boundaries without sacrificing global context.

## What's in this release (paper → code)

| Paper component | Code |
|---|---|
| **G-Mamba** — geometry-guided Mamba operator | `GMambaBlock` in [`dgm_net/model.py`](dgm_net/model.py) |
| **Cascade GN-SSM** — 2× spatial Mamba + 1× G-Mamba | `SpatialMambaBlock`, `CascadeGN_SSM` |
| **DGM-Module** — geometric priors (potential V / centripetal flow / boundary D / curvature) | `dgm_predictor` head + `compute_dgm_targets_adaptive` in [`dgm_net/dgm_targets.py`](dgm_net/dgm_targets.py) |
| **GOAD** — geometry-aware offset-aligned decoder | `DGMNet.forward` |
| Training objective — OHEM CE + Lovász + geometric terms | `DGMNetLoss_Pro` in [`dgm_net/losses.py`](dgm_net/losses.py) |

Training scripts and the data pipeline are not included in this release.

## Results

### Cityscapes (val)

| Setting | mIoU (%) |
|---|---|
| 28k iterations (SS) | 80.8 |
| 90k iterations (SS) | 81.6 |
| 28k + MST | 82.3 |

### ADE20K

| Setting | mIoU (%) |
|---|---|
| 28k iterations | 45.24 |
| 90k iterations | 46.43 |

All results are obtained under a **single-GPU (16 GB VRAM)** budget, without large-scale segmentation pretraining. Reduced configurations still reach ~77 mIoU on 8 GB GPUs.

## Installation

```bash
pip install -r requirements.txt
```

`mamba-ssm` requires an NVIDIA GPU with CUDA. Installing `causal-conv1d` first is recommended for speed.

## Usage

### Inference

```python
import torch
from dgm_net import DGMNet

device = "cuda"
model = DGMNet(n_classes=19).to(device).eval()

x = torch.randn(1, 3, 768, 768, device=device)
with torch.no_grad():
    out_c, out_dgm = model(x)   # (B, 19, H, W), (B, 5, H, W)
pred = out_c.argmax(1)
```

`out_dgm` channels are `[V, Gx, Gy, D_logits, C_map]`.

### Generating DGM-Module targets for training

```python
import numpy as np
from dgm_net import compute_dgm_targets_adaptive

mask = np.array(...)  # (H, W) int label map, ignore label = 255
dgm_targets = compute_dgm_targets_adaptive(mask)  # (5, H, W) float32
```

### Loss

```python
from dgm_net import DGMNetLoss_Pro

criterion = DGMNetLoss_Pro(ohem_thresh=0.9, ignore_index=255).to(device)
# model in train mode returns (out_c, out_dgm, out_aux)
total, l_seg, l_vg, l_d = criterion(out_c, target_c, out_dgm, dgm_targets, out_aux)
```

## Datasets

Cityscapes and ADE20K — please download from the official sources.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{chan2026breaking,
  title={Breaking the Resource Wall: Geometry-Guided Sequence Modeling for Efficient Semantic Segmentation},
  author={Chan, Sheng-Wei and Pan, Hsin-Jui and Shen, Chun-Po and Lin, Chia-Min and Wang, Yung-Che and Chiang, Jen-Shiun},
  journal={arXiv preprint arXiv:2604.23399},
  year={2026}
}
```
