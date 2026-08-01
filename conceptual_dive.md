# WheelEye: A Conceptual Dive

This document provides a comprehensive, end-to-end breakdown of the WheelEye system, covering its workflow, computer vision implementation (with code), and the underlying algorithms that power its Machine Learning, Computer Vision, and Neural Network architectures.

---

## 1. The Entire Workflow & Functionality

The system operates in a highly optimized loop designed to fit within a strict 1-3 second factory takt time (the rate at which products need to be finished to meet customer demand).

1. **Capture & Manifest Delivery:** As a component (e.g., a wheel assembly) moves down the conveyor belt, an image is captured. Simultaneously, the expected "manifest" (what the part *should* be: specific material, size, correct number of fasteners) is queried.
2. **API Ingestion:** The React frontend (acting as the Human-Machine Interface) sends the image and the manifest to the FastAPI backend via an HTTP POST request.
3. **Inference Pipeline:**
   - The backend passes the image to the **WheelEye Core Engine**.
   - The image is processed through a highly optimized ONNX model running on the CPU.
   - The model detects all objects (fasteners, wheel size boundaries) and classifies the presence of defects (scratches, dents).
4. **Logic & Verification:** The backend compares the model's physical detections against the required JSON manifest. If a car requires 5 fasteners but only 4 are detected, or a scratch is found, the part is flagged.
5. **Operator Feedback (Dynamic Noise Filtering):** The backend responds to the UI. To prevent operator fatigue, the UI **only** draws bounding boxes around anomalies or missing components, hiding successful matches. 
6. **Telemetry:** Every inference request logs metrics (latency, pass/fail rate) to Prometheus, which is instantly viewable on a live Grafana dashboard for factory managers.

---

## 2. The Computer Vision Pipeline (in Detail)

The Computer Vision (CV) pipeline is built entirely in PyTorch and is explicitly optimized for speed (CPU inference) and high accuracy on small objects (like tiny fasteners on a large wheel). It is divided into three main components: Preprocessing, the Multi-Head Classifier, and the Single-Stage Detector.

### A. Preprocessing (`preprocessing.py`)

Before an image can be fed into a neural network, it needs to be transformed into a standardized tensor. 

```python
import torchvision.transforms as T

# Standard ImageNet normalization for pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size, augment=False):
    transforms = [T.Resize(image_size)]
    
    # Simulating factory lighting conditions
    if augment:
        transforms.extend([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return T.Compose(transforms)
```
**Why this matters:** The `ColorJitter` is crucial here. In a real factory, overhead lights flicker, cameras get bumped, and parts have different glare. By randomly jittering the colors during training, we force the network to look at the *shapes* (fasteners, scratches) rather than relying on exact pixel colors. The ImageNet Normalization is required because we are using pretrained backbones that expect this exact mathematical distribution.

### B. The Multi-Head Classifier (`models/classifier.py`)

This model determines the wheel's configuration (Material, Tier, Size). Notice how we extract the `features` from a pretrained MobileNetV3, and then branch out into three separate linear heads.

```python
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class WheelEyeClassifier(nn.Module):
    def __init__(self, num_material=2, num_tier=3, num_size=3):
        super().__init__()
        
        # 1. Load pretrained backbone
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = mobilenet.features # Output shape: (Batch, 576, H, W)
        
        # 2. Pooling & Flattening
        self.pool = nn.AdaptiveAvgPool2d(1) # Squashes H, W down to 1x1
        self.flatten = nn.Flatten()
        
        # 3. The Three Independent Heads
        in_features = 576
        self.material_head = nn.Linear(in_features, num_material)
        self.tier_head = nn.Linear(in_features, num_tier)
        self.size_head = nn.Linear(in_features, num_size)

    def forward(self, x):
        # Extract features globally once
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        
        # Pass the single feature vector to all three heads
        material_out = self.material_head(x)
        tier_out = self.tier_head(x)
        size_out = self.size_head(x)
        
        return material_out, tier_out, size_out
```

### C. The WheelEye Detector (`models/detector.py`)

This is the most advanced part of the system, determining *where* the objects and defects are. It's built in three stages: **Backbone**, **Neck (PANet)**, and **Decoupled Head**.

#### I. The Backbone (Multi-Scale Extraction)
Unlike the classifier which only wants the *final* feature map, the detector needs features at different scales to find large wheels and tiny scratches. 

```python
from torchvision.models.feature_extraction import create_feature_extractor

class Backbone(nn.Module):
    def __init__(self, freeze_early_layers=True):
        super().__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Tap into the network at 3 different resolutions
        return_nodes = {
            'features.3': 'C3',   # High Res (Stride 8) - Good for tiny scratches
            'features.8': 'C4',   # Medium Res (Stride 16)
            'features.12': 'C5'   # Low Res, High Semantic (Stride 32)
        }
        
        self.body = create_feature_extractor(mobilenet, return_nodes=return_nodes)
```

#### II. The Neck (PANet)
The `C3`, `C4`, and `C5` maps are isolated. The PANet fuses them together so the network has both context (what the object is) and localization (exactly where its edges are).

```python
class PANet(nn.Module):
    # ... init omitted for brevity ...
    
    def forward(self, features):
        c3, c4, c5 = features['C3'], features['C4'], features['C5']

        # --- Top-down (FPN Pathway) ---
        # Bring semantic meaning DOWN from low-res to high-res
        p5 = self.lat_c5(c5)
        
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.td_conv_p4(self.lat_c4(c4) + p5_up)
        
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.td_conv_p3(self.lat_c3(c3) + p4_up)

        # --- Bottom-up (PAN Pathway) ---
        # Bring sharp spatial edges UP from high-res to low-res
        n3 = p3 
        
        n3_down = self.bu_down_n3(n3)
        n4 = self.bu_conv_n4(n3_down + p4)
        
        n4_down = self.bu_down_n4(n4)
        n5 = self.bu_conv_n5(n4_down + p5)

        # We now have 3 perfectly fused feature maps
        return [n3, n4, n5]
```

#### III. The Decoupled Head
Those fused maps (`N3, N4, N5`) are passed to independent branches to predict the class and bounding box coordinates.

```python
class DecoupledHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=4, num_anchors=9):
        super().__init__()
        
        # Branch 1: What is it?
        self.cls_conv = nn.Sequential(
            ConvBnSiLU(in_channels, in_channels),
            ConvBnSiLU(in_channels, in_channels),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Branch 2: Where is it? (dx, dy, dw, dh)
        self.reg_conv = nn.Sequential(
            ConvBnSiLU(in_channels, in_channels),
            ConvBnSiLU(in_channels, in_channels),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        
        # Initialization Hack for Focal Loss (CRITICAL)
        # We tell the network to assume EVERYTHING is background on step 1.
        # This prevents the loss from exploding on the first batch.
        pi = 0.01
        bias_val = -torch.math.log((1 - pi) / pi)
        nn.init.constant_(self.cls_conv[-1].bias, bias_val)

    def forward(self, p):
        # Format the outputs to (Batch, Anchors, Values)
        cls_out = self.cls_conv(p)
        reg_out = self.reg_conv(p)
        return cls_out, reg_out
```

---

## 3. The Algorithm Breakdown

To understand WheelEye, we have to look at the three foundational pillars of Artificial Intelligence it uses. While often used interchangeably, **Machine Learning**, **Computer Vision**, and **Neural Networks** refer to different parts of the algorithm's lifecycle.

### A. The Machine Learning Algorithm: *How it Learns*
Machine Learning is the algorithm of **optimization**. The system isn't programmed with rules; it learns via **Supervised Learning and Gradient Descent**.

*   **The Forward Pass:** An image of a wheel is pushed through the network. The network outputs random guesses for where the fasteners are.
*   **The Loss Function (The Grader):** The ML algorithm compares the network's random guess to the "Ground Truth" (the bounding boxes drawn by a human). It calculates a "Loss" (error) score.
    *   For the Classifier, it uses **Cross-Entropy Loss** (penalizing wrong categories).
    *   For the Bounding Boxes, it likely uses a **Focal Loss** combined with an **IoU (Intersection over Union) Loss** to measure how perfectly the predicted box overlaps the real box.
*   **The Task-Aligned Assigner (The Secret Weapon):** In standard ML algorithms, an anchor box is assigned to a ground truth object if they physically overlap. **Task-Aligned Learning (TAL)** dynamically calculates an "alignment metric" by multiplying the *classification score* and the *localization accuracy*. It forces the network to learn to predict the highest confidence score for the box that has the tightest physical boundaries.
*   **Backpropagation:** The algorithm calculates the calculus derivatives (gradients) of the Loss backward through every single parameter in the network, adjusting the weights slightly so the guess will be better next time.

### B. The Computer Vision Algorithm: *How it "Sees"*
Computer Vision algorithms deal with spatial mathematics.

*   **Convolutions (The Sliding Window):** The core algorithm is the `Conv2d` layer. Imagine a small 3x3 magnifying glass (a matrix of numbers called a kernel) sliding across the image pixel by pixel. At every step, it performs a mathematical dot product. 
*   **Hierarchical Receptive Fields:** 
    *   In the first few layers, the convolutions act as simple math filters, finding **edges and gradients** (light-to-dark transitions).
    *   In the middle layers, it combines those edges to find **shapes and textures** (the curve of a rim, the metallic texture of a bolt).
    *   In the deep layers, it combines shapes to find **semantic concepts** (the entire wheel hub).
*   **The Feature Pyramid:** The CV algorithm explicitly handles scale. By extracting features at **Stride 8 (High Res)**, **Stride 16 (Mid Res)**, and **Stride 32 (Low Res)**, the CV algorithm ensures that tiny scratches are detected by the high-resolution maps, while the overall wheel size is determined by the low-resolution maps.

### C. The Neural Network Algorithms: *The Architecture*
Neural Networks dictate the *flow of information*. 

*   **Depthwise Separable Convolutions (The MobileNetV3 Algorithm):** A standard convolution algorithm is mathematically expensive (it multiplies every input channel with every output channel). The `MobileNetV3` backbone uses an algorithm that splits this math into two steps: **Depthwise** (it applies a single filter to each color channel independently) and **Pointwise** (it uses a 1x1 convolution to combine them). This mathematically reduces the required CPU calculations by roughly 8x, making the <50ms inference time possible.
*   **Path Aggregation Network (The PANet Algorithm):** Information naturally flows one way in a neural network: from raw pixels (shallow) to abstract concepts (deep). The PANet algorithm physically rewires this. It uses **addition** and **interpolation** algorithms. It takes the abstract concepts (like "I think this general area is a wheel hub") and mathematically *adds* them back to the shallow, high-resolution layers (like "Here are the exact sharp edges of the metal"). It creates a bi-directional flow of information.
*   **The Decoupled Head Algorithm:** Predicting *what* something is (Classification) requires focusing on the center of the object to see its features. Predicting *where* something is (Regression) requires focusing on the sharp outer edges of the object. The Decoupled Head algorithm physically splits the network into two separate paths at the very end so these two conflicting mathematical tasks don't interfere with each other.
