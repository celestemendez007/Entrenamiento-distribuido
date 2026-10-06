# Instructions for Claude Code

Welcome! This repository contains a completed, successful **Distributed Deep Learning project** that trained a **Panoptic Segmentation model (Panoptic FPN with ResNet-50)** across two Windows PCs using PyTorch DDP + Gloo and Detectron2.

> **CRITICAL DIRECTIVE:**
> The model is ALREADY FULLY TRAINED (`output/model_final.pth`, 40.4% Panoptic Quality on COCO).
> **DO NOT initiate any new training or re-training.**
> The current objective is strictly **testing, inference demonstration, visualization, and helping the user record/explain their project.**

## Core Information & Quick Reference
For full technical details, engineering solutions, and architectural background, consult `PROJECT_CONTEXT.md`.

### Weights & Checkpoint
* The final weights file `model_final.pth` (~180 MB) is stored on the user's USB drive (`DENIIIIIIS (D:)`) to comply with GitHub's 100 MB file limit.
* To run inference, place `model_final.pth` in the project root or `./output/`.

### Running Demos and Inference
* Script: `demo_inference.py`
* Basic command:
  ```powershell
  python demo_inference.py --weights model_final.pth --image path/to/image.jpg --output result.jpg
  ```
* Supports CUDA or CPU automatically (`cfg.MODEL.DEVICE`).

### Key Accomplishments & Metrics (For Explanations / Video)
* **Architecture:** Panoptic FPN (ResNet-50 backbone) on COCO 2017.
* **Distributed Setup:** 2 Windows PCs (RTX 3060 12GB each), PyTorch DDP over TCP via GLOO backend.
* **Training Time:** 5 hours, 19 minutes (optimized down from initial ~45 hours via AMP FP16, batch size 12, Ampere Tensor Cores).
* **Final Results:**
  * **PQ (Panoptic Quality):** 40.43% (matches/exceeds official Facebook AI Research baseline of ~40.2%).
  * **SQ (Segmentation Quality):** 77.75% (mask edge precision).
  * **RQ (Recognition Quality):** 49.46% (detection/classification accuracy).
