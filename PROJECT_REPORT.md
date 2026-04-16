# DeepFake Detection System — Complete Project Report

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [End-to-End Workflow](#3-end-to-end-workflow)
4. [Dataset — FaceForensics++](#4-dataset--faceforensics)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Model Architecture — EfficientNet-B4](#6-model-architecture--efficientnet-b4)
7. [Why EfficientNet-B4?](#7-why-efficientnet-b4)
8. [Transfer Learning — How & Why](#8-transfer-learning--how--why)
9. [Training Strategy](#9-training-strategy)
10. [Data Augmentation](#10-data-augmentation)
11. [Training Results (v2)](#11-training-results-v2)
12. [Explainable AI — Grad-CAM](#12-explainable-ai--grad-cam)
13. [Web Application — Streamlit](#13-web-application--streamlit)
14. [False Positive Mitigation](#14-false-positive-mitigation)
15. [Project Structure](#15-project-structure)
16. [How to Run — Full Pipeline](#16-how-to-run--full-pipeline)
17. [Limitations & Future Work](#17-limitations--future-work)
18. [Common Questions & Answers](#18-common-questions--answers)

---

## 1. Project Overview

| Detail | Value |
|--------|-------|
| **Goal** | Detect deepfake images/videos using deep learning |
| **Model** | EfficientNet-B4 (pretrained on ImageNet, fine-tuned) |
| **Face Detection** | MTCNN (Multi-task Cascaded Convolutional Network) |
| **Explainability** | Grad-CAM heatmaps showing model attention |
| **Frontend** | Streamlit web application with live analysis |
| **Dataset** | FaceForensics++ (C23 compression) |
| **Best Val Accuracy** | 98.45% (v2, 15 epochs, 54K faces) |
| **Language** | Python 3.x + PyTorch |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU (4GB VRAM) |

---

## 2. Problem Statement & Motivation

### What are Deepfakes?
Deepfakes are AI-generated media where a person's face is synthetically replaced with someone else's face, typically using **autoencoders** or **Generative Adversarial Networks (GANs)**. The term comes from combining "deep learning" + "fake."

### Why is Detection Important?
1. **Misinformation** — Fake videos of politicians/public figures spreading false information
2. **Identity theft & fraud** — Impersonating individuals in video calls or financial transactions
3. **Non-consensual content** — Creating fake intimate media of real people
4. **Erosion of trust** — When anyone's face can be faked, video evidence loses credibility

### Our Approach
We use a **Convolutional Neural Network (CNN)** to detect deepfakes by identifying subtle artifacts that deepfake generation algorithms leave behind:
- **Blending boundaries** — Where the swapped face meets the original face
- **Texture inconsistencies** — Unnatural skin texture or lighting patterns
- **Facial anomalies** — Mismatched eye reflections, teeth artifacts, jawline irregularities

The model doesn't just give a verdict — it generates **Grad-CAM heatmaps** showing exactly which facial regions it focused on, making the detection **explainable and trustworthy**.

---

## 3. End-to-End Workflow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE PIPELINE (5 PHASES)                          │
│                                                                          │
│   PHASE 1: PREPROCESSING                                                │
│   ┌──────────────┐     ┌──────────┐     ┌───────────────┐               │
│   │ Raw Videos   │────▶│  MTCNN   │────▶│ Face Crops    │               │
│   │ (FaceFor++)  │     │  Face    │     │ 380×380 PNG   │               │
│   │ ~7000 videos │     │  Detect  │     │ ~54,000 faces │               │
│   └──────────────┘     └──────────┘     └───────┬───────┘               │
│                                                 │                        │
│   PHASE 2: TRAINING                             ▼                        │
│   ┌─────────────────────────────────────────────────────────┐            │
│   │  EfficientNet-B4 (ImageNet Pretrained → Fine-tuned)     │            │
│   │                                                         │            │
│   │  Phase A: Freeze backbone (2 epochs) → 66% accuracy     │            │
│   │  Phase B: Unfreeze + fine-tune (13 epochs) → 98.45%     │            │
│   │                                                         │            │
│   │  Output: best.pth (67.6 MB model weights)               │            │
│   └─────────────────────────────────────────────────────────┘            │
│                                                                          │
│   PHASE 3: TESTING                                                       │
│   ┌──────────────┐     ┌──────────────┐     ┌───────────────┐           │
│   │ Test faces   │────▶│ Inference    │────▶│ Metrics       │           │
│   │ 8,144 images │     │ (best.pth)   │     │ Acc/Prec/Rec  │           │
│   └──────────────┘     └──────────────┘     └───────────────┘           │
│                                                                          │
│   PHASE 4: PREDICTION                                                    │
│   ┌──────────────┐     ┌──────────┐     ┌──────────────┐                │
│   │ New video    │────▶│ MTCNN +  │────▶│ Annotated    │                │
│   │ (phone/web)  │     │ Model    │     │ output video │                │
│   └──────────────┘     └──────────┘     └──────────────┘                │
│                                                                          │
│   PHASE 5: EXPLAINABILITY                                                │
│   ┌──────────────┐     ┌──────────┐     ┌──────────────┐                │
│   │ Face image   │────▶│ Grad-CAM │────▶│ Heatmap      │                │
│   │              │     │          │     │ overlay      │                │
│   └──────────────┘     └──────────┘     └──────────────┘                │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset — FaceForensics++

### What is FaceForensics++?
FaceForensics++ (FF++) is the **largest and most widely-used benchmark** for face manipulation detection, created by researchers at TU Munich. It contains:

| Category | Count | Description |
|----------|-------|-------------|
| **Original (Real)** | 1,000 videos | Unmodified YouTube face videos |
| **Deepfakes** | 1,000 videos | Autoencoder-based face swap |
| **Face2Face** | 1,000 videos | Expression/reenactment transfer |
| **FaceSwap** | 1,000 videos | 3D morphable model face swap |
| **FaceShifter** | 1,000 videos | GAN-based high-fidelity face swap |
| **NeuralTextures** | 1,000 videos | Neural rendering manipulation |
| **DeepFakeDetection** | 1,000 videos | Google's deepfake generation |
| **Total** | ~7,000 videos | |

### Compression Level
We use **C23** (constant rate quantization parameter 23) — light compression that preserves most manipulation artifacts while being realistic to real-world video quality.

### Our Sampling
- **Sample ratio:** 50% of videos from each folder (v2)
- **Frame interval:** Every 20th frame extracted
- **Total faces extracted:** 54,288

### Data Split

| Split | Count | Real | Fake | Ratio |
|-------|-------|------|------|-------|
| Train | 38,001 | 5,509 (14.5%) | 32,492 (85.5%) | 1:6 |
| Validation | 8,143 | 1,246 (15.3%) | 6,897 (84.7%) | 1:6 |
| Test | 8,144 | 1,205 (14.8%) | 6,939 (85.2%) | 1:6 |

### Fake Generation Methods Explained:

1. **Deepfakes** — Uses an autoencoder architecture. Two encoders share weights, two separate decoders learn each person's face. Decoder A reconstructs face B → face swap.

2. **Face2Face** — Real-time facial reenactment. Transfers expressions from a source face to a target face using 3D face reconstruction and dense photometric alignment.

3. **FaceSwap** — Traditional computer vision approach using 3D morphable models. Fits a 3D face model, transfers shape/texture between source and target.

4. **FaceShifter** — Uses a two-stage GAN: first generates a high-fidelity swapped face, then a second network (HEAR) refines anomalous regions for seamless blending.

5. **NeuralTextures** — Learns a neural texture map of the target face, enabling neural rendering for subtle expression modification without full face replacement.

6. **DeepFakeDetection** — Google's method for generating realistic deepfakes, specifically designed to challenge detection systems.

---

## 5. Preprocessing Pipeline

**Script:** `preprocess_dataset.py`

### Step-by-Step Process:

```
Video file (.mp4)
    │
    ▼
Step 1: cv2.VideoCapture() reads the video
    │
    ▼
Step 2: Extract every Nth frame (N=20 for speed, N=10 for quality)
    │
    ▼
Step 3: MTCNN detects face bounding boxes
    │    - P-Net: generates candidate windows (12×12 grid)
    │    - R-Net: refines candidates, removes false positives
    │    - O-Net: outputs precise bounding box + 5 facial landmarks
    │
    ▼
Step 4: Expand bounding box by 40px margin (captures jaw/forehead)
    │
    ▼
Step 5: Crop and resize face to 380×380 pixels
    │
    ▼
Step 6: Save as PNG in: processed_faces/{method}/{video_id}/frame_{N}.png
    │
    ▼
Step 7: Generate train/val/test split text files (70/15/15)
        Each line: <image_path> <label>   (0=REAL, 1=FAKE)
```

### What is MTCNN?
**Multi-task Cascaded Convolutional Network** — a 3-stage face detector:

| Stage | Network | Purpose | Input |
|-------|---------|---------|-------|
| 1 | P-Net (12×12) | Fast candidate generation | Full image at multiple scales |
| 2 | R-Net (24×24) | Refine candidates + reject false positives | Candidate boxes from P-Net |
| 3 | O-Net (48×48) | Final bounding box + 5 landmarks (eyes, nose, mouth) | Refined boxes from R-Net |

We use MTCNN because:
- ✅ High accuracy face detection even in varied conditions
- ✅ GPU-accelerated via PyTorch (facenet-pytorch library)
- ✅ Works on different face sizes and orientations
- ✅ Provides landmark points for identity verification

### Performance Optimizations:
1. **Thread-local MTCNN instances** — Each worker gets its own MTCNN to avoid GPU contention
2. **ThreadPoolExecutor** — 4-8 parallel workers process videos concurrently
3. **Batch frame capture** — Reads all target frames first, then processes in batches of 16
4. **Frame skipping** — Only processes every Nth frame (reduces I/O)

### Command:
```bash
python preprocess_dataset.py -d "/path/to/FF++_C23" -o "./processed_faces" \
    --sample_ratio 0.50 --num_workers 8 --frame_interval 20
```

---

## 6. Model Architecture — EfficientNet-B4

### Architecture Overview

```
INPUT IMAGE (380 × 380 × 3 RGB)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│           EfficientNet-B4 BACKBONE                  │
│   (19.3 million parameters, ImageNet pretrained)    │
│                                                     │
│   ┌─────────────────────────────────────────┐       │
│   │ Stem: Conv2d(3, 48, kernel=3, stride=2) │       │
│   │ + BatchNorm + SiLU activation           │       │
│   └─────────────────┬───────────────────────┘       │
│                     │                               │
│   ┌─────────────────▼───────────────────────┐       │
│   │ Stage 1: MBConv1, 48→24, ×2 blocks     │       │
│   │ Stage 2: MBConv6, 24→32, ×4 blocks     │       │
│   │ Stage 3: MBConv6, 32→56, ×4 blocks     │       │
│   │ Stage 4: MBConv6, 56→112, ×6 blocks    │       │
│   │ Stage 5: MBConv6, 112→160, ×6 blocks   │       │
│   │ Stage 6: MBConv6, 160→272, ×8 blocks   │       │
│   │ Stage 7: MBConv6, 272→448, ×2 blocks   │       │
│   └─────────────────┬───────────────────────┘       │
│                     │                               │
│   ┌─────────────────▼───────────────────────┐       │
│   │ Head: Conv2d(448, 1792, kernel=1)       │       │
│   │ + BatchNorm + SiLU                      │       │
│   └─────────────────┬───────────────────────┘       │
└─────────────────────┼───────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │ AdaptiveAvgPool2d       │
         │ (1792 channels → 1792) │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ Dropout (p=0.5)        │  ← Regularization
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ Linear (1792 → 2)      │  ← Our custom classifier
         └────────────┬────────────┘
                      │
                      ▼
         [P(REAL), P(FAKE)]  ← Softmax probabilities
```

### What are MBConv Blocks?
**Mobile Inverted Bottleneck Convolution** — the core building block of EfficientNet:

```
Input (C channels)
    │
    ▼
Expand: Conv1×1 (C → C×E)          ← Expand channels by factor E (1 or 6)
    │   + BatchNorm + SiLU
    ▼
Depthwise: Conv3×3 or 5×5          ← Spatial filtering (each channel separately)
    │   + BatchNorm + SiLU            Depthwise = way fewer parameters than regular conv
    ▼
Squeeze-and-Excitation (SE)         ← Channel attention
    │   Global avg pool → FC → ReLU → FC → Sigmoid
    │   Learns which channels are important
    ▼
Project: Conv1×1 (C×E → C_out)     ← Compress back to output channels
    │   + BatchNorm
    ▼
Skip Connection (+)                 ← If input size = output size, add residual
    │
Output (C_out channels)
```

**Key advantages:**
- **Depthwise separable convolutions** — 8-9× fewer parameters than standard convolutions
- **Squeeze-and-Excitation** — Learns which feature channels matter (attention mechanism)
- **Skip connections** — Enables gradient flow, prevents vanishing gradients

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Backbone (features) | ~17.5M |
| Head conv (1×1) | ~800K |
| Classifier (Linear 1792→2) | ~3.6K |
| **Total** | **~19.3M** |

---

## 7. Why EfficientNet-B4?

### Compound Scaling
EfficientNet's key innovation is **compound scaling** — it scales width, depth, and resolution **simultaneously** using a compound coefficient φ:

- **Depth** (d = α^φ): More layers = more complex features
- **Width** (w = β^φ): More channels = richer representations
- **Resolution** (r = γ^φ): Larger input = more detail

This is fundamentally different from ResNet (depth-only scaling) or WideResNet (width-only scaling).

### Why B4 Specifically?

| Model | Input Size | Parameters | Top-1 ImageNet Acc |
|-------|-----------|-----------|-------------------|
| ResNet-50 | 224×224 | 23.5M | 76.1% |
| EfficientNet-B0 | 224×224 | 5.3M | 77.1% |
| EfficientNet-B3 | 300×300 | 12M | 81.6% |
| **EfficientNet-B4** | **380×380** | **19.3M** | **82.9%** |
| EfficientNet-B7 | 600×600 | 66M | 84.3% |

We chose B4 because:
1. **380×380 resolution** — High enough to capture subtle blending artifacts at face boundaries
2. **19.3M parameters** — Large enough for nuanced feature extraction, small enough for 4GB VRAM
3. **Best accuracy/compute tradeoff** — B7 would be better but doesn't fit in 4GB VRAM
4. **ImageNet pretraining** — Already understands textures, edges, lighting → perfect starting point

### Comparison with Other Models

| Feature | EfficientNet-B4 | ResNet-50 | XceptionNet |
|---------|-----------------|-----------|-------------|
| Input resolution | 380×380 | 224×224 | 299×299 |
| Parameters | 19.3M | 23.5M | 22.9M |
| ImageNet accuracy | 82.9% | 76.1% | 79.0% |
| Depthwise convs | ✅ Yes | ❌ No | ✅ Yes |
| SE attention | ✅ Yes | ❌ No | ❌ No |
| Compound scaling | ✅ Yes | ❌ No | ❌ No |

---

## 8. Transfer Learning — How & Why

### What is Transfer Learning?
Instead of training a model from scratch (random weights → learn everything), we take a model already trained on a large dataset (ImageNet — 1.3M images, 1000 classes) and **adapt it** for our specific task (deepfake detection).

### Why It Works for Deepfakes
ImageNet-pretrained models have already learned:
- **Low-level features** (edges, textures, gradients) — useful for detecting blending artifacts
- **Mid-level features** (patterns, shapes) — useful for detecting facial structure anomalies
- **High-level features** (objects, faces) — useful for understanding facial context

We only need to teach the model **what makes a face "fake"** — the visual feature extraction is already done.

### Our Two-Phase Approach

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Classifier Warmup (Epochs 1-2)                        │
│                                                                 │
│ Backbone:    🔒 FROZEN (ImageNet weights preserved)             │
│ Classifier:  🔓 Training from random initialization             │
│ LR:          0.001 (full learning rate)                         │
│                                                                 │
│ WHY: The classifier head starts with random weights.            │
│ If we unfreeze the backbone now, random gradients from the      │
│ classifier would DESTROY the good pretrained features.          │
│ So we let the classifier learn meaningful outputs first.        │
│                                                                 │
│ Expected: 60-66% accuracy (classifier learns, backbone frozen)  │
│ Observed: Epoch 1 = 65.58%, Epoch 2 = 66.23%                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    🔓 UNFREEZE BACKBONE
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│ PHASE 2: Full Fine-Tuning (Epochs 3-15)                        │
│                                                                 │
│ Backbone:    🔓 UNFROZEN with discriminative LRs:               │
│              - Backbone features: LR × 0.1 (= 0.0001)          │
│              - Classifier head:   LR × 1.0 (= 0.001)           │
│ Optimizer:   Reset AdamW with parameter groups                  │
│                                                                 │
│ WHY use lower LR for backbone?                                  │
│ The backbone already has good features from ImageNet.           │
│ We want to NUDGE them toward deepfake detection,               │
│ not completely overwrite them. 10x lower LR = gentle nudging.  │
│                                                                 │
│ Observed: Epoch 3 = 89.32% (huge jump!), Epoch 14 = 98.45%    │
└─────────────────────────────────────────────────────────────────┘
```

### What is Discriminative Learning Rate?
Different parts of the model get different learning rates:

| Layer Group | Learning Rate | Rationale |
|------------|---------------|-----------|
| Backbone features | 0.0001 (LR × 0.1) | Don't overwrite good ImageNet features |
| Classifier head | 0.001 (full LR) | Needs to learn fast — started from random |

This ensures the backbone features are gently adapted while the classifier learns aggressively.

---

## 9. Training Strategy

### Optimizer: AdamW
**Adam with Decoupled Weight Decay** — improves upon standard Adam by properly applying L2 regularization:
- **Momentum (β₁=0.9):** Smooths gradient updates
- **RMS scaling (β₂=0.999):** Adapts learning rate per parameter
- **Weight decay (1e-4):** Penalizes large weights → prevents overfitting

### Learning Rate Scheduler: CosineAnnealingWarmRestarts
Instead of a fixed learning rate, the LR follows a **cosine curve** that periodically restarts:

```
LR
 │  ╱╲
 │ ╱  ╲         ╱╲
 │╱    ╲       ╱  ╲
 │      ╲     ╱    ╲
 │       ╲   ╱      ╲
 │        ╲ ╱        ╲
 └──────────────────────── Epochs
    T₀=5     T₀×T_mult=10
```

- **T₀=5:** First cosine cycle is 5 epochs
- **T_mult=2:** Each subsequent cycle is 2× longer
- **Why restarts?** Allows the model to escape local minima by temporarily increasing LR

### Mixed Precision Training (AMP)
**Automatic Mixed Precision** uses FP16 (half precision) instead of FP32 for GPU operations:

| Aspect | FP32 | FP16 |
|--------|------|------|
| Bits per number | 32 | 16 |
| VRAM usage | 100% | ~50% |
| Training speed | 1× | ~2× |
| Precision | Higher | Lower (but sufficient) |

**GradScaler** prevents numerical underflow in FP16 by scaling gradients before backward pass and unscaling before optimizer step.

### Class Imbalance Handling: WeightedRandomSampler
Our dataset has a **1:6 real:fake ratio** (5,509 real vs 32,492 fake in training). Without handling this:
- Model could achieve 85% accuracy by **always predicting FAKE**
- Model wouldn't learn what "real" looks like

**WeightedRandomSampler** solution:
```python
weights = 1.0 / class_counts        # Real gets weight 6×, Fake gets weight 1×
sampler = WeightedRandomSampler(...)  # Each batch has ~50% real, ~50% fake
```

### Early Stopping
Monitors validation accuracy. If no improvement for N epochs (patience), training stops to prevent overfitting:
- **v1:** patience=5
- **v2:** patience=3

---

## 10. Data Augmentation

Data augmentation artificially increases training data diversity by applying random transformations:

### Standard Augmentations

| Augmentation | Parameter | Purpose |
|---|---|---|
| **RandomHorizontalFlip** | p=0.5 | Faces can appear mirrored |
| **RandomRotation** | ±15° | Handles tilted faces |
| **ColorJitter** | brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08 | Different lighting conditions |
| **GaussianBlur** | kernel=3, σ=0.1-2.0 | Simulates out-of-focus camera |
| **RandomGrayscale** | p=0.05 | Color-invariant features |
| **RandomErasing** | p=0.2, scale=2-15% | Handles partial occlusion |

### Real-World Augmentations (v2 — Critical for Phone Video Robustness)

| Augmentation | Parameter | Purpose |
|---|---|---|
| **JPEGCompression** | quality=30-95, p=0.5 | Simulates phone camera, social media compression |
| **RandomDownscale** | scale=0.5-1.0, p=0.3 | Simulates low-resolution captures |

**Why these matter:**
The v1 model was trained only on FF++ (C23/H.264 compression). When it saw phone videos (HEVC/H.265), it confused compression artifacts with deepfake artifacts → false positives on real phone videos.

By training with random JPEG compression and downscaling, the model learns that **compression artifacts ≠ deepfake artifacts**.

### Validation/Test Transforms
Only resize + normalize (no augmentation) — we want consistent, fair evaluation:
```python
Resize(380×380) → ToTensor() → Normalize(ImageNet mean/std)
```

---

## 11. Training Results (v2)

### Per-Epoch Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Phase |
|-------|-----------|-----------|----------|---------|-------|
| 1 | 0.6251 | 63.57% | 0.5782 | 65.58% | 🔒 Frozen |
| 2 | 0.6155 | 64.81% | 0.5677 | 66.23% | 🔒 Frozen |
| 3 | 0.3944 | 81.05% | 0.2480 | **89.32%** | 🔓 Unfroze! |
| 4 | 0.1964 | 92.14% | 0.1031 | 95.95% | Fine-tune |
| 5 | 0.1313 | 94.82% | 0.0896 | 96.50% | Fine-tune |
| 6 | 0.1015 | 96.12% | 0.0846 | 96.92% | Fine-tune |
| 7 | 0.0828 | 96.70% | 0.0764 | 97.02% | Fine-tune |
| 8 | 0.0974 | 96.20% | 0.1763 | 94.69% | LR restart |
| 9 | 0.0850 | 96.82% | 0.1007 | 96.60% | Recovery |
| 10 | 0.0706 | 97.30% | 0.0598 | 97.72% | Fine-tune |
| 11 | 0.0609 | 97.69% | 0.0695 | 97.38% | Fine-tune |
| 12 | 0.0487 | 98.26% | 0.0643 | 97.75% | Fine-tune |
| 13 | 0.0392 | 98.54% | 0.0483 | 98.28% | Fine-tune |
| **14** | **0.0356** | **98.72%** | **0.0482** | **98.45%** | **🏆 Best** |
| 15 | 0.0312 | 98.84% | 0.0611 | 98.07% | Slight overfit |

### Key Observations:
1. **Epochs 1-2 (frozen):** Model learns basic classification with frozen backbone → ~66% accuracy
2. **Epoch 3 (unfreeze):** Massive jump from 66% → 89% as backbone adapts to deepfake features
3. **Epochs 4-7:** Rapid improvement as full network fine-tunes → 97%
4. **Epoch 8 (LR restart):** Cosine annealing restarts LR → temporary accuracy drop (normal!)
5. **Epochs 9-14:** Model recovers and achieves best accuracy → **98.45%**
6. **Epoch 15:** Slight val loss increase → early signs of overfitting

### Training Configuration (v2)
```
Model:          EfficientNet-B4 (ImageNet pretrained)
Dataset:        54,288 faces (50% FF++, frame_interval=20)
Batch size:     8
Epochs:         15
Optimizer:      AdamW (weight_decay=1e-4)
Initial LR:    0.001
Scheduler:     CosineAnnealingWarmRestarts (T₀=5, T_mult=2)
Mixed Precision: FP16 (AMP)
GPU:            NVIDIA RTX 3050 (4GB VRAM)
Training time:  ~6 hours total
```

---

## 12. Explainable AI — Grad-CAM

### What is Grad-CAM?
**Gradient-weighted Class Activation Mapping** — a technique that produces visual explanations for CNN decisions by highlighting which regions of the input image are most important for the prediction.

### How It Works (Step by Step):

```
Step 1: FORWARD PASS
    Image → EfficientNet-B4 → [P(REAL)=0.03, P(FAKE)=0.97]
    
    During forward pass, we capture the ACTIVATIONS
    at the last convolutional layer (A^k, shape: 1792 × 12 × 12)

Step 2: BACKWARD PASS
    Compute gradients of the FAKE score with respect to
    the last conv layer activations (∂y_FAKE/∂A^k)

Step 3: COMPUTE IMPORTANCE WEIGHTS
    α_k = (1/Z) × Σᵢ Σⱼ (∂y_FAKE / ∂A^k_ij)
    
    Global Average Pooling of gradients → one weight per channel
    Channels with large positive gradients = important for prediction

Step 4: WEIGHTED COMBINATION
    L_GradCAM = ReLU( Σ_k  α_k × A^k )
    
    Weighted sum of activation maps → ReLU (keep only positive contributions)
    Result: a heatmap showing WHERE the model is looking

Step 5: OVERLAY
    Resize heatmap to input image size
    Apply colormap (JET: blue=low → red=high attention)
    Blend with original image at 50% opacity
```

### Mathematical Formulation:
```
α_k^c = (1/Z) Σᵢ Σⱼ (∂y^c / ∂A^k_ij)          ← importance of channel k for class c
L^c_GradCAM = ReLU(Σ_k α_k^c × A^k)              ← weighted combination + ReLU
```

### What the Heatmaps Reveal:

**Real faces → Model focuses on:**
- Natural skin texture (nose, cheeks, forehead)
- Consistent lighting/shadow patterns
- Single concentrated attention region
- Uniform texture confidence

**Fake faces → Model focuses on:**
- **Blending boundaries** (jawline, hairline, forehead) — where swapped face meets original
- **Eye region** — deepfakes often fail at realistic eye reflections and catchlights
- **Mouth/teeth** — synthesis errors at lip edges and teeth rendering
- **Multiple scattered attention spots** — indicating multiple artifact sites

### Why Grad-CAM Matters:
1. **Trust** — Shows the model is detecting real artifacts, not memorizing patterns
2. **Debugging** — If attention is on background, model might be using dataset bias
3. **Communication** — Non-technical stakeholders can understand why a prediction was made

---

## 13. Web Application — Streamlit

**Script:** `app.py`

### Features:
- 📸 **Image Analysis** — Upload face image → REAL/FAKE/UNCERTAIN verdict + Grad-CAM heatmap
- 🎬 **Video Analysis** — Upload video → frame-by-frame analysis + confidence chart
- 🎚️ **Adjustable Thresholds** — Sidebar sliders for fake confidence threshold and verdict ratio
- 🧠 **Grad-CAM Panels** — Original / Heatmap / Overlay side-by-side
- 📊 **Metrics Dashboard** — Frames analyzed, real/fake counts, video duration
- 🎨 **Premium Dark Theme** — Gradient accents, glassmorphism, Inter font

### Architecture:
```python
@st.cache_resource
def load_model():    # Loaded once, cached in memory
    model = EfficientNet_B4(pretrained_weights)
    return model

# Image tab: Upload → MTCNN face detect → Model predict → Grad-CAM → Display
# Video tab: Upload → Sample frames → Batch predict → Confidence chart
```

### Launch:
```bash
streamlit run app.py
```

---

## 14. False Positive Mitigation

### The Problem:
Model trained on FF++ (H.264/C23 compression) misclassifies real phone videos as fake because phone videos use different compression (HEVC/H.265), lighting, and quality — the model confuses these differences with deepfake artifacts.

### Solutions Implemented:

| Solution | How It Works | Impact |
|----------|-------------|--------|
| **Confidence Threshold (70%)** | Only flag as FAKE if P(fake) ≥ 0.70. Below → UNCERTAIN | Eliminates borderline false positives |
| **Video Verdict Ratio (60%)** | Require 60%+ of frames to be FAKE for video-level verdict | Prevents one bad frame from ruining the verdict |
| **JPEG Augmentation** | Training with random JPEG quality 30-95 | Model learns compression artifacts ≠ deepfake |
| **Downscale Augmentation** | Training with random 0.5-1.0× resolution | Handles low-res phone cameras |
| **More Training Data** | 54K faces (v2) vs 44K faces (v1) | Greater face diversity |

### Confidence Categories:
```
P(FAKE) ≥ 0.70  →  🚨 FAKE    (high confidence)
P(FAKE) ≤ 0.30  →  ✅ REAL    (high confidence)
0.30 < P(FAKE) < 0.70  →  ⚠️ UNCERTAIN  (borderline)
```

---

## 15. Project Structure

```
DeepFake/
├── network/
│   ├── __init__.py
│   └── models.py                 ← EfficientNet-B4 model definition
├── dataset/
│   ├── __init__.py
│   ├── transform.py              ← Augmentations + normalization
│   └── mydataset.py              ← PyTorch Dataset class
├── preprocess_dataset.py         ← Phase 1: MTCNN face extraction
├── train_CNN.py                  ← Phase 2: Training with AMP + CSV logging
├── test_CNN.py                   ← Phase 3: Metrics (Acc/Prec/Rec/F1)
├── predict.py                    ← Phase 4: Video prediction + threshold
├── explain.py                    ← Phase 5: Grad-CAM visualization
├── app.py                        ← Streamlit web application
├── generate_report.py            ← HTML report generator
├── plot_training.py              ← Training curve plotter (matplotlib)
├── shutdown_after_training.py    ← Auto-shutdown after training
├── requirements.txt              ← Dependencies
├── README.md                     ← GitHub README
├── PROJECT_REPORT.md             ← This file
├── .gitignore                    ← Git exclusions
├── processed_faces/              ← Extracted faces (generated)
├── data_list/                    ← Train/val/test manifests (generated)
├── output/
│   ├── ff_effnet_v1/best.pth    ← v1 model (12% data, no compression aug)
│   └── ff_effnet_v2/            ← v2 model (50% data + compression aug)
│       ├── best.pth              ← Best model weights (98.45% val acc)
│       ├── training_history.csv  ← Per-epoch metrics
│       ├── training_log.txt      ← Complete terminal output
│       ├── loss_curve.png        ← Loss over epochs
│       ├── accuracy_curve.png    ← Accuracy over epochs
│       └── training_summary.png  ← Combined loss + accuracy chart
├── explanations/                 ← Grad-CAM output images
└── predictions/                  ← Annotated prediction videos
```

---

## 16. How to Run — Full Pipeline

### Prerequisites
```bash
git clone https://github.com/OmRajput17/Deep-Fake.git
cd Deep-Fake
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Phase 1: Preprocess Dataset
```bash
python preprocess_dataset.py \
    -d "/path/to/FaceForensics++_C23" \
    -o "./processed_faces" \
    --sample_ratio 0.50 \
    --num_workers 8 \
    --frame_interval 20
```

### Phase 2: Train Model
```bash
python train_CNN.py --name ff_effnet_v2 --batch_size 8 --epoches 15 --lr 0.001 --patience 3
```

### Phase 3: Test Model
```bash
python test_CNN.py --test_list ./data_list/ff_all_test.txt --model_path ./output/ff_effnet_v2/best.pth
```

### Phase 4: Predict on New Video
```bash
python predict.py -i video.mp4 -m ./output/ff_effnet_v2/best.pth --threshold 0.70
```

### Phase 5: Generate Grad-CAM
```bash
python explain.py -i face.png -m ./output/ff_effnet_v2/best.pth
```

### Launch Web App
```bash
streamlit run app.py
```

### Generate Training Plots
```bash
python plot_training.py --history ./output/ff_effnet_v2/training_history.csv
```

---

## 17. Limitations & Future Work

### Current Limitations
1. **Single face per frame** — MTCNN `keep_all=False` processes only the largest face
2. **No temporal modeling** — Each frame is analyzed independently (no inter-frame consistency check)
3. **FF++ bias** — Trained only on FaceForensics++; may not generalize to entirely new deepfake methods
4. **Phone video gap** — Mitigated with augmentation but not fully solved
5. **No real-time processing** — Inference is ~100ms/frame (10 FPS) on RTX 3050

### Potential Improvements
1. **Cross-dataset training** — Add Celeb-DF, DFDC, WildDeepfake datasets
2. **Temporal models** — LSTM/Transformer on frame sequences for temporal inconsistency detection
3. **Multi-face support** — Process all faces in frame simultaneously
4. **ONNX/TensorRT export** — For faster inference and mobile deployment
5. **Audio analysis** — Detect audio-visual desynchronization
6. **Adversarial robustness** — Train against adversarial attacks on deepfakes

---

## 18. Common Questions & Answers

### Q1: Is this a pretrained model or trained from scratch?
**A:** Transfer learning. We use EfficientNet-B4 **pretrained on ImageNet** (1.3M images, 1000 classes) and **fine-tune** it for binary deepfake classification. The backbone already understands visual features (textures, edges, shapes). We add a new 2-class classifier head (Linear 1792→2) and fine-tune the whole network using discriminative learning rates.

### Q2: Why EfficientNet-B4 and not ResNet or VGG?
**A:** EfficientNet uses **compound scaling** (width + depth + resolution simultaneously). B4 at 380×380 resolution with 19.3M params achieves 82.9% ImageNet accuracy — better than ResNet-50 (23.5M params, 76.1% accuracy) while using fewer parameters. The higher resolution is critical for capturing subtle deepfake artifacts.

### Q3: What is MTCNN and why use it?
**A:** Multi-task Cascaded Convolutional Network — a 3-stage face detector (P-Net → R-Net → O-Net). We use it because: (a) high detection accuracy, (b) GPU-accelerated via PyTorch, (c) provides facial landmarks for alignment, (d) handles varied face sizes and orientations.

### Q4: Why two-phase training instead of training all at once?
**A:** Phase 1 (frozen) lets the classifier learn meaningful outputs before backbone updates. If we unfreeze immediately, **random gradients from the untrained classifier would destroy the good pretrained features**. Phase 2 uses 10× lower LR for backbone to gently adapt features rather than overwrite them.

### Q5: What is Mixed Precision Training?
**A:** Uses FP16 (16-bit floats) instead of FP32 for GPU operations. Benefits: ~2× faster training, ~50% less VRAM. GradScaler prevents numerical underflow. This is critical for our RTX 3050 (4GB VRAM) — without AMP, batch size > 4 would cause OOM errors.

### Q6: How does WeightedRandomSampler handle class imbalance?
**A:** With 1:6 real:fake ratio, the sampler assigns 6× higher probability to real samples. Each training batch gets approximately 50% real and 50% fake, even though the dataset is imbalanced. This prevents the model from achieving high accuracy by simply predicting FAKE for everything.

### Q7: What is CosineAnnealingWarmRestarts?
**A:** A LR scheduler where the learning rate follows a cosine curve from max to min, then "restarts" at max. This helps escape local minima by periodically increasing LR. T₀=5 means the first cycle is 5 epochs, T_mult=2 means each subsequent cycle is 2× longer.

### Q8: What does Grad-CAM show mathematically?
**A:** It computes importance weights α_k by global average pooling the gradients of the target class score w.r.t. the last conv layer activations. The weighted sum of activation maps, passed through ReLU: `L = ReLU(Σ_k α_k × A^k)`. ReLU ensures only features that **positively contribute** to the prediction are shown.

### Q9: Why does the model fail on some phone videos?
**A:** Domain gap. FF++ uses H.264/C23 compression. Phone videos use H.265/HEVC with different quality, lighting, and resolution. v2 mitigates this with JPEG compression augmentation during training and a 70% confidence threshold at inference. Remaining failures are from extreme domain differences.

### Q10: How does the confidence threshold reduce false positives?
**A:** Instead of argmax (51% fake = FAKE), we require ≥70% confidence. Frame-level predictions between 30-70% are marked UNCERTAIN. For video-level verdict, 60%+ of frames must be FAKE. This prevents borderline predictions from causing false alarms.

### Q11: What is the SiLU activation function?
**A:** SiLU (Sigmoid Linear Unit) = x × sigmoid(x). Also called "Swish." Unlike ReLU (which kills negative values), SiLU smoothly handles negatives. EfficientNet uses SiLU throughout, which improves gradient flow and training stability.

### Q12: Can this detect deepfakes in real-time?
**A:** At ~100ms per frame on RTX 3050, it achieves ~10 FPS — not true real-time for video but fast enough for per-frame analysis. With TensorRT optimization or ONNX quantization, real-time performance (~30 FPS) is achievable.

### Q13: What if someone asks about the v1 vs v2 model?
**A:** v1 used 12% data without compression augmentation → 99.1% test accuracy on FF++ but high false positives on phone videos. v2 used 50% data + JPEG/downscale augmentation → 98.45% val accuracy but better generalization to real-world videos. v2 trades slight in-distribution accuracy for much better out-of-distribution robustness.

### Q14: What is the model file size and can it be reduced?
**A:** `best.pth` is ~67.6 MB (FP32 weights). It can be reduced to ~34 MB via FP16 quantization or ~17 MB via INT8 quantization with minimal accuracy loss (~0.5%).

### Q15: How would you deploy this to production?
**A:** Options: (a) Export to ONNX → deploy with ONNX Runtime, (b) TensorRT for NVIDIA GPU servers, (c) Streamlit Community Cloud for the web app, (d) Docker container with FastAPI backend, (e) Mobile via TFLite/CoreML conversion.

---

*Report generated for DeepFake Detection Project — EfficientNet-B4 + MTCNN + Grad-CAM*
*Last updated: March 2026*
