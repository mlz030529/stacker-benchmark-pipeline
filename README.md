# STaCker Benchmark Pipeline

This repository provides a **clean and reproducible benchmarking pipeline** for evaluating **STaCker** on spatial transcriptomics (ST) datasets using coordinate-based inputs.


---

## Overview

- **Input**
  - Cell-level spatial coordinates (`center_x`, `center_y`)
  - Gene expression matrices (cells × genes)

- **Output**
  - Aligned spatial coordinates for the moving slice
  - Benchmark-ready CSV files

- **Core method**
  - **STaCker with affine-only spatial registration**

Although STaCker supports both **affine** and **dense (nonlinear)** registration,  
this benchmark intentionally restricts alignment to affine transformations only.
The dense alignment stage is intentionally disabled and not evaluated in this repository.

STaCker’s dense alignment stage relies on patch-wise image deformation using a deep neural network.
In this benchmark, spatial transcriptomic slices are converted from point coordinates into smooth
density raster images for alignment.

Under this setting, the resulting density images are spatially sparse and highly smoothed,
which leads to empty or invalid patch selections during dense registration.
As a result, the dense model produces empty batches and cannot generate meaningful deformation fields.
## Pretrained weights

This repository does **not** include pretrained STaCker weights due to size limits.

Please download the official weights from: https://github.com/regeneron-mpds/stacker




