# WheelEye Development - Future Phases

This document outlines the phases for the upcoming development cycles to upgrade the custom WheelEye model to industry-standard (YOLO-equivalent) performance for the Renault Production Plant.

## Phase 1: Data & Ground Truth Overhaul (Complete)
- [x] Enhance `generate_synthetic_data.py` to produce photorealistic images (metallic textures, reflections, noise, and shadows) simulating factory environments.
- [x] Run the synthetic data generator to create a robust 200+ image dataset.
- [x] Restructure the dataset loader (`yolo_dataset.py`) to support Mosaic augmentations, heavily improving the model's ability to detect small objects like fasteners.

## Phase 2: Loss & Anchor Formulation Upgrade (Complete)
- [x] Implement **CIoU (Complete Intersection over Union) Loss** in `loss.py` to replace Smooth L1 regression, penalizing box overlap, center distance, and aspect ratio simultaneously.
- [x] Migrate the anchor assignment logic in `anchors.py` from Max-IoU Bipartite matching to a dynamic **Task-Aligned Assigner**, preventing static scaling bugs and improving detection convergence.

## Phase 3: Custom Architecture Evolution (Complete)
- [x] Upgrade the detection neck in `detector.py` from a basic FPN to a **PANet (Path Aggregation Network)** for superior multi-scale feature fusion.
- [x] Integrate **SiLU** activations across the convolution blocks for better gradient flow on edge devices.
- [x] Ensure strict mathematical decoupling between the regression and classification branches in the Decoupled Head.

## Phase 4: Production Training Pipeline (Complete)
- [x] Add **Cosine Annealing Learning Rate** schedules to `train_detect.py` to stabilize the loss curve.
- [x] Introduce **EMA (Exponential Moving Average)** weight updates during training to achieve higher peak accuracy and stability.
- [x] Train the upgraded model on the new photorealistic dataset.

## Phase 5: High-Efficiency Export & Deployment (Complete)
- [x] Develop `scripts/export.py` to strip training-specific tensors and export the `WheelEyeDetector` to the **ONNX** format, specifically optimized for NVIDIA Jetson deployment.
- [x] Update `verifier.py` to utilize `onnxruntime` for inference, reducing latency by up to 3x to easily meet the 1.5s Takt time requirement.

## Phase 6: Dynamic Frontend Integration (Complete)
- [x] Update `wheeleye/frontend/src/App.jsx` to replace the hardcoded expected manifest with a dynamic dropdown menu.
- [x] Allow operators to select the incoming wheel SKU (Material, Tier, Size) so the backend can verify live camera feeds against the correct expected configuration.
