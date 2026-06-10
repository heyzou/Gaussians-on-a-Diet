# Gaussians on a Diet

Official code for:

**Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training**  
Yangming Zhang, Jian Xu, Chaojian Li, Kunxiong Zhu, Wei Niu, Gagan Agrawal, Yang Katie Zhao, Jian Wang, Yingyan Celine Lin, Miao Yin  
arXiv, 2026

Paper: [arXiv:2604.20046](https://arxiv.org/abs/2604.20046)

## Overview

Gaussians on a Diet is a memory-bounded 3D Gaussian Splatting training framework. It alternates between growing, compensation, and pruning to keep training memory usage under a budget while preserving rendering quality.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

## Training

Example scripts are provided:

```bash
bash train.sh
bash train_op.sh
```

Please update dataset paths and output paths in the scripts before running.

## Citation

```bibtex
@article{zhang2026gaussians,
  title={Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training},
  author={Zhang, Yangming and Xu, Jian and Li, Chaojian and Zhu, Kunxiong and Niu, Wei and Agrawal, Gagan and Zhao, Yang Katie and Wang, Jian and Lin, Yingyan Celine and Yin, Miao},
  journal={arXiv preprint arXiv:2604.20046},
  year={2026}
}
```
