# CLARIFI3D

**CLARIFI3D** is a GPU-accelerated pipeline for 3D image segmentation and analysis, designed for processing anisotropic confocal volumes of primary human fibroblasts and similar cell types.  
It integrates classical image processing, watershed-based segmentation, and machine learning components to extract single-cell and sub-organelle features from large-scale microscopy datasets.

This repository contains the version of CLARIFI3D used for the analyses reported in:

> Shukla et al., *Nature Communications*, 2025

---

## Status

- **Active development**: This codebase is research software under continuous refinement.  
- **Intended use**: Academic reproducibility of the results in the associated manuscript.  
- **Stability**: APIs and outputs may change in future releases.  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/fdmast-2/CLARIFI3D.git
cd CLARIFI3D
pip install -e .

Python ≥3.9 and CUDA-enabled PyTorch are required for GPU execution.

⸻

Core Modules
	•	io_clarifi3d.py – I/O functions for volumetric imaging datasets
	•	models.py – neural network models (e.g. 3D UNet variants)
	•	normalization.py – preprocessing and intensity normalization
	•	profiling.py – feature extraction and quantitative profiling
	•	seg.py – segmentation pipeline logic
	•	watershed.py – custom GPU watershed implementation
	•	filters.py – image filtering utilities
	•	utils.py – helper functions
	•	cli.py – command-line interface wrapper
	•	setup.py – package configuration

⸻

Versioning

The release corresponding to the Nature Communications manuscript is tagged:

v0.1.0-ncomms

Please cite this version when referring to the code used in the paper.

⸻

License

This project is licensed under the Apache License 2.0 (see LICENSE).

⸻

Citation

If you use CLARIFI3D in your work, please cite:

Shukla et al., TOR and heat shock response pathways regulate peroxisome biogenesis during proteotoxic stress. Nature Communications (2025).


⸻

Contact

For questions, reproducibility requests, or collaborations, please contact:
Fred Mast, PhD
Seattle Children’s Research Institute
Email: Fred.Mast@SeattleChildrens.org

---
