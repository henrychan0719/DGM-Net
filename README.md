# DGM-Net

**Breaking the Resource Wall: Geometry-Guided Sequence Modeling for Efficient Semantic Segmentation**

---

## Overview

DGM-Net (Directional Geometric Mamba Network) is an efficient semantic segmentation framework designed for resource-constrained environments.

Recent State Space Models (SSMs), such as Mamba, provide efficient long-range modeling with linear complexity, but often suffer from over-smoothing during spatial propagation, which degrades boundary quality.

In this work, we address this issue by reformulating spatial modeling as a **geometry-conditioned propagation process**. By introducing explicit geometric priors, DGM-Net guides feature aggregation in a structure-aware manner, enabling better boundary preservation without sacrificing global context modeling.

---

## Key Ideas

- Introduce **geometry-guided propagation** into SSM-based models  
- Propose **G-Mamba**, a structure-aware variant of Mamba  
- Design the **DGM-Module** to extract geometric priors (flow, boundary, curvature, potential)  
- Develop a **geometry-aware decoder (GOAD)** for feature alignment and refinement  

---

## Results

### Cityscapes
| Setting | mIoU (%) |
|--------|--------|
| 28k iterations (SS) | 80.8 |
| 90k iterations (SS) | 81.6 |
| 28k + MST | 82.3 |


### ADE20K
| Setting | mIoU (%) |
|--------|--------|
| 28k iterations | 45.24 |
| 90k iterations | 46.43 |

The model achieves competitive performance under a **single-GPU (16GB VRAM)** setting, without relying on large-scale pretraining.
---

## Efficiency

- Linear complexity: O(N)  
- Single GPU training (16GB VRAM)  
- Still works under 8GB GPUs (~77 mIoU)  

---

## Status

Code is being cleaned and will be released soon.

---

## Dataset

- Cityscapes  
- ADE20K  

Please download from official sources.

---

## Paper

Available on arXiv:

**Breaking the Resource Wall: Geometry-Guided Sequence Modeling for Efficient Semantic Segmentation**

---

## Citation

```bibtex
@article{chan2026breaking,
  title={Breaking the Resource Wall: Geometry-Guided Sequence Modeling for Efficient Semantic Segmentation},
  author={Chan, Sheng-Wei and Pan, Hsin-Jui and Shen, Chun-Po and Lin, Chia-Min and Wang, Yung-Che and Chiang, Jen-Shiun},
  journal={arXiv preprint arXiv:2604.23399},
  year={2026}
}
