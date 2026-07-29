# WheelEye: Automated Visual Inspection for Automotive Assembly

![alt text](docs/assets/image.png)

## Overview
**WheelEye** is an end-to-end computer vision platform designed to automate the visual inspection of automotive components on a high-speed factory line. It acts as an intelligent checkpoint that verifies wheel configurations (material, tier, size, and fastener counts) against a manufacturing manifest and detects critical assembly defects like scratches and dents.

This system demonstrates full-stack AI engineering—bridging the gap between raw PyTorch models and a production-ready, deployable microservice architecture with real-time observability.

---

## Key Features

- **Smart Defect Highlighting**: The UI intelligently filters out background noise, drawing bounding boxes *only* when an error occurs. Critical defects like scratches and dents flash with a high-visibility red animation to instantly alert line operators.
- **Real-Time Observability**: Fully instrumented with Prometheus and Grafana. Custom business metrics (Pass Rate, Total Units Inspected, Average Inference Latency) are aggregated and visualized in a live dashboard.
- **High-Speed Inference**: By utilizing ONNX exports and optimized graph execution for the custom PANet WHEELEYE detector, inference runs smoothly in <50ms on CPU, easily meeting standard factory takt times of 1-3 seconds.
- **Robust Data Augmentation**: The synthetic data generator employs OpenCV to simulate factory floor conditions, including high-speed motion blur (conveyor belt simulation) and harsh overhead glare, ensuring the model remains resilient in production.

---

## Architecture

The system is decoupled into three primary tiers:

```mermaid
graph LR
    subgraph Frontend [React + Vite]
        UI[Dashboard UI]
        SIM[Line Simulator]
    end
    
    subgraph Backend [FastAPI]
        API[Inference API]
        CV[WHEELEYE + Classifier]
    end
    
    subgraph Observability [Prometheus & Grafana]
        PROM[Prometheus Scraper]
        GRAF[Grafana Dashboard]
    end

    UI -- HTTP POST /inspect --> API
    SIM -- Fake Takt Time --> UI
    API -- Expose /metrics --> PROM
    PROM -- Query --> GRAF
```

1. **Frontend (React + Vite)**: A sleek, dark-mode terminal interface built to resemble industrial HMIs (Human-Machine Interfaces). It features a manual upload mode and a "Simulate Line" mode that mocks a high-speed factory conveyor belt.
2. **Backend (FastAPI)**: Serves the computer vision pipeline. It accepts image uploads and JSON manifests, runs the image through an ONNX-optimized WHEELEYE detector and a PyTorch classifier, and returns a structured JSON report.
3. **Observability**: The backend is instrumented with Prometheus. Grafana is auto-provisioned via Docker Compose to instantly display real-time metrics.

---

## Quick Start (Docker)

The easiest way to run the entire stack (Frontend, Backend, Prometheus, Grafana) is via Docker Compose.

```bash
# Start the system and build containers
docker-compose up -d --build
```

**Available Services:**
- **Frontend UI**: `http://localhost:80` (or `http://localhost:5173` if running Vite locally)
- **FastAPI Swagger Docs**: `http://localhost:8000/docs`
- **Grafana Dashboard**: `http://localhost:3000` (Login: `admin` / `admin`)
- **Prometheus**: `http://localhost:9090`

---

## Tech Stack
* **Machine Learning**: PyTorch, Custom WHEELEYE Architecture (MobileNetV3 + PANet), ONNX, OpenCV
* **Backend**: Python 3.10, FastAPI, Uvicorn
* **Frontend**: React 18, Vite, Lucide Icons, Vanilla CSS
* **DevOps**: Docker, Docker Compose
* **Telemetry**: Prometheus, Grafana

---

## System Evolution & Debugging Logs

Building a robust CV system for manufacturing is challenging. A critical milestone in this project was resolving silent failure modes during the custom object detector implementation.

**The Tensor Index Contract Violation:**
During Phase 5 integration, the ONNX-exported detector failed to locate any objects despite a normally decreasing training loss. The root cause was a subtle tensor index contract violation: the detector head flattened the tensor using `permute(0, 1, 3, 4, 2)`, while the ground-truth target generator expected `(B, H*W, num_anchors, num_classes)`. This caused the network's predictions for the top-left of the image to be penalized against ground-truth objects in the bottom-right. 

**The Resolution:**
We implemented strict structural safeguards:
1. **Permanent Unit Tests**: A test (`tests/test_tensor_contract.py`) that simulates the exact reshape/permute path on a dummy tensor.
2. **Unified Preprocessing**: Stripped out duplicated transforms and centralized them in `wheeleye/preprocessing.py`.
3. **Overfit Sanity Checks**: A tiny-batch CPU training script (`scripts/sanity_check_overfit.py`) to prove rapid loss collapse before committing to full training runs.

*Takeaway: A decreasing loss curve does not guarantee a model is learning the correct objective. Silent coordinate mapping errors must be verified with explicit index contract tests.*
