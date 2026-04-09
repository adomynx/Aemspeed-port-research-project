# Comparative Analysis of Image-Based, LiDAR-Based, and Multi-Modal Fusion Obstacle Detection for Autonomous Systems

**Author:** Kartik Subhash Wakekar  
**Program:** M.Sc. Electrical Engineering, TH OWL  
**Total GPU-Hours:** ~344

---

## Repository Structure

```
├── README.md
├── lidar-based/
│   ├── pointpillars/
│   │   ├── kitti/          (config, training log)
│   │   └── nuscenes/       (config, training log)
│   └── second/
│       ├── kitti/          (config, training log)
│       └── nuscenes/       (config, training log)
├── fusion/
│   ├── voxel-rcnn-focal/
│   │   └── kitti/          (config, training log)
│   ├── mvxnet/
│   │   └── kitti/          (config, training log)
│   └── bevfusion/
│       └── nuscenes/       (config, training & eval logs)
└── image-based/            (configs & scripts — weights lost)
```

---

## 1. Trained Model Weights

Weights are hosted on Google Drive due to file size constraints:

**[Download Model Weights (1.38 GB)](https://drive.google.com/drive/folders/10lazpY6fCT_dGG5bBgWq7nwmYrXWjsm_?usp=drive_link)**

| Model | Dataset | File | Size | Epochs |
|-------|---------|------|------|--------|
| PointPillars | KITTI | checkpoint_epoch_80.pth | 56 MB | 80 |
| PointPillars | nuScenes | checkpoint_epoch_10.pth | 67 MB | 10 |
| SECOND | KITTI | checkpoint_epoch_80.pth | 62 MB | 80 |
| SECOND | nuScenes | checkpoint_epoch_10.pth | 79 MB | 10 |
| Voxel-RCNN Focal | KITTI | checkpoint_epoch_80.pth | 244 MB | 80 |
| MVXNet | KITTI | epoch_40.pth | 394 MB | 40 |
| BEVFusion | nuScenes | epoch_3.pth | 476 MB | 3 (of 10 planned) |

### Missing Weights

Image-based model weights (YOLOv5m, YOLOv8m, RT-DETR-L) for both COCO and nuScenes were lost due to external storage failure. All results reported in the thesis are reproducible using the training configurations and scripts provided in this repository.

---

## 2. Datasets

| Dataset | Purpose | Download Link |
|---------|---------|---------------|
| MS COCO 2017 | Image-based training & evaluation | [cocodataset.org](https://cocodataset.org/#download) |
| KITTI 3D Object Detection | LiDAR & fusion training/evaluation | [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) |
| nuScenes v1.0 | Cross-modal evaluation (all modalities) | [nuscenes.org](https://www.nuscenes.org/nuscenes#download) |

### Dataset Modifications

- **nuScenes to YOLO format:** Converted nuScenes 2D annotations to YOLO-format bounding boxes for image-based model evaluation. 10 nuScenes classes mapped with normalized coordinates.
- **KITTI train/val split:** Standard Chen et al. split — 3,712 training / 3,769 validation samples.
- **nuScenes CBGS resampling:** Class-Balanced Grouping and Sampling applied for LiDAR training, expanding 28,130 original samples to 112,040 resampled training frames.

---

## 3. Training Configurations

### Image-Based Models (~191 GPU-hours on RTX 3050 6 GB)

| Parameter | COCO | nuScenes |
|-----------|------|----------|
| Models | YOLOv5m, YOLOv8m, RT-DETR-L | YOLOv5m, YOLOv8m, RT-DETR-L |
| Epochs | 10 | 30 |
| Batch Size | 4 | 4 |
| Input Resolution | 640×640 | 768×768 |
| Optimizer (YOLO) | SGD (lr=0.01, momentum=0.937) | SGD (lr=0.01, momentum=0.937) |
| Optimizer (RT-DETR) | AdamW (lr=0.0001) | AdamW (lr=0.0001) |
| Weight Init | ImageNet pretrained | COCO pretrained |
| Augmentation | Mosaic, HSV, flip, scale | Mosaic, HSV, flip, scale |

### LiDAR-Based Models (~81 GPU-hours on A100 80 GB)

| Parameter | PointPillars KITTI | SECOND KITTI | PointPillars nuScenes | SECOND nuScenes |
|-----------|-------------------|--------------|----------------------|-----------------|
| Epochs | 80 | 80 | 10 | 10 |
| Batch Size | 4 | 2 | 16 | 4 |
| Learning Rate | 0.003 | 0.003 | 0.001 | 0.001 |
| Optimizer | AdamW (wd=0.01) | AdamW (wd=0.01) | AdamW (wd=0.01) | AdamW (wd=0.01) |
| Framework | OpenPCDet | OpenPCDet | OpenPCDet | OpenPCDet |

### Fusion Models (~71 GPU-hours on A100 80 GB)

| Parameter | Voxel-RCNN Focal | MVXNet | BEVFusion |
|-----------|-----------------|--------|-----------|
| Dataset | KITTI | KITTI | nuScenes |
| Epochs | 80 | 40 | 3 (of 10 planned) |
| Batch Size | 2 | 8 | 4 |
| Learning Rate | 0.01 | 0.0001 | 0.0002 |
| Optimizer | AdamW (OneCycle) | AdamW | AdamW (Cyclic) |
| Framework | OpenPCDet | mmdetection3d | mmdetection3d |

BEVFusion training was limited to 3 epochs due to computational constraints (~12–13 hours per epoch on A100). Despite this, results were highly significant — see Key Results below.

---

## 4. Data Pipelines

### Image-Based Pipeline
```
Raw Images → Resize (640×640) → Mosaic Augmentation → HSV Jitter →
Random Flip/Scale → Model Input → NMS Post-processing → mAP Evaluation
```

### LiDAR-Based Pipeline (OpenPCDet)
```
Raw Point Cloud (.bin) → Point Range Filter → Voxelization →
Sparse 3D Conv (SECOND) / Pillar Encoding (PointPillars) →
BEV Feature Map → Region Proposal → 3D BBox Prediction → NMS → Evaluation
```

### Multi-Modal Fusion Pipeline (BEVFusion)
```
Camera Images → Swin-T Backbone → FPN → Camera BEV Features ─┐
                                                               ├→ BEV Fusion → Detection Head → 3D BBoxes
LiDAR Points → VoxelNet → Sparse Conv → LiDAR BEV Features ──┘
```

---

## 5. Key Results

| Model | Dataset | Key Metric | Value |
|-------|---------|-----------|-------|
| RT-DETR-L | COCO | mAP@0.5:0.95 | 54.4% |
| YOLOv8m | COCO | mAP@0.5:0.95 | 50.6% |
| PointPillars | KITTI (Car, Moderate) | 3D AP | 82.54% |
| SECOND | KITTI (Car, Moderate) | 3D AP | 79.67% |
| Voxel-RCNN Focal | KITTI (Car, Moderate) | 3D AP | 90.37% |
| MVXNet | KITTI (Car, Moderate) | 3D AP | 82.78% |
| BEVFusion (3 ep) | nuScenes | mAP / NDS | 61.47% / 63.98% |

**Centerpiece finding:** BEVFusion at just 3 epochs detected all 10/10 nuScenes classes vs. LiDAR-only 4/10, with pedestrian AP recovering from 0% to 87.6% — demonstrating the critical safety value of multi-modal fusion for autonomous systems.

---

## 6. Environment Setup

### Image-Based (Local Machine)
- **GPU:** NVIDIA RTX 3050 (6 GB VRAM)
- **Framework:** Ultralytics (YOLOv5, YOLOv8, RT-DETR)
- **Install:** `pip install ultralytics`

### LiDAR & Fusion (DGX Server)
- **GPU:** NVIDIA A100 (80 GB VRAM)
- **Docker Image:** `nvcr.io/nvidia/pytorch:24.01-py3`
- **Frameworks:** OpenPCDet, mmdetection3d

```bash
# OpenPCDet
cd OpenPCDet && pip install -r requirements.txt && python setup.py develop

# mmdetection3d
cd mmdetection3d && pip install -r requirements.txt && pip install -v -e .
```

---

## License

This repository is part of a Research project at TH OWL and is intended for academic evaluation purposes.
