# WheelEye Project Status

## Project Overview
**Goal:** A dual-camera AI visual inspection system for the Renault-Nissan assembly line to verify correct wheel installation, tire tier, and fastener counts.
**Architecture:** 
- **Detector:** YOLO-style object detector using MobileNetV3-Small backbone and PANet neck, featuring a custom focal-loss decoupled head.
- **Classifier:** MobileNetV3-Small classifier for determining material/tier.
- **Backend:** FastAPI for asynchronous inference via ONNX Runtime.
- **Frontend:** Next.js React Dashboard for real-time factory monitoring.

## Current Status (Phase 5 Complete, Re-Training Required)
The core infrastructure, API, frontend, and ONNX export logic are 100% complete and fully integrated. The system successfully boots up using Docker.

However, a **critical flaw** was discovered in the Detector's mathematical design, which caused the model to learn absolutely nothing during the Kaggle training run, resulting in a model that outputs `0` bounding boxes.

## The Flaw (And the Fix)
**The Bug:** Inside `wheeleye/models/detector.py`, the network flattened the output tensors in a `(num_anchors, H, W)` order. However, the ground-truth targets (anchors) were generated in a `(H*W, num_anchors)` order. 
**The Consequence:** During the initial Kaggle training, the model's predictions for one part of the image were being compared to ground-truth objects in completely random, unrelated parts of the image. The gradients were scrambled, preventing the loss from dropping and resulting in a randomly initialized "untrained" model being exported.
**The Fix:** This has been resolved by correcting the permutation sequence in `detector.py` to `permute(0, 3, 4, 1, 2)`. I have also removed an incorrect ImageNet normalization step from the inference pipeline that mismatched the training data distribution.

## Next Steps
All fixes have been pushed to the GitHub repository. To complete the project, you must re-run the training on Kaggle so the model can actually learn with the corrected gradient mappings.

### Steps to Execute on Kaggle:
1. Open your Kaggle Notebook.
2. Clone the latest updated codebase from GitHub.
3. Run the training script for both models.
4. Export the ONNX files and download them.
5. Place them in your local `exports/` folder and run `docker-compose up`!
