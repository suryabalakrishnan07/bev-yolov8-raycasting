# OmniVision-BEV: Real-Time Geometric Radar

## Project Overview
OmniVision-BEV is a 360-degree Bird's-Eye-View (BEV) perception system designed to provide real-time situational awareness for autonomous vehicles. Developed for the 2026 AI and Computer Vision Challenge by Team Why Not, this project demonstrates that actionable 3D localization can be achieved using standard RGB cameras and geometric algorithms.

Our approach reduces sensor hardware costs by over 90% compared to traditional LiDAR-based systems, making high-level perception accessible on consumer-grade edge hardware while maintaining a high frame rate for safety-critical applications.

## Model Architecture
The system utilizes a modular "Detect-then-Project" pipeline optimized for low-latency inference on the edge:
- 2D Detection Backbone: We utilize YOLOv8-Nano for its industry-leading inference speed. It processes six synchronized camera feeds simultaneously to identify vehicle classes (Car, Truck, Bus).
- Geometric Raycasting Engine: The system employs a pinhole camera model with a ground-plane constraint ($Z=0$) to "lift" 2D detections into 3D ego-coordinates.
- Spatial Multi-View Fusion: Detections from overlapping camera fields are unified into a single coordinate system, providing a cohesive top-down radar map and filtering out mathematically impossible projections.

## Dataset Used
The project is built and evaluated on the nuScenes v1.0-mini dataset. This dataset provides:
- Synchronized 360-degree camera coverage across six high-resolution sensors.
- Full sensor calibration data (intrinsics and extrinsics).
- LiDAR-verified ground truth annotations for 3D object localization.

## Setup and Installation

The system was developed and tested on Pop!_OS / Ubuntu 24.04 using an NVIDIA GeForce RTX 4060 (8GB).

### Prerequisites
- Python 3.10+
- CUDA 12.1+

### Installation Steps
1. Clone the Repository
```
git clone https://github.com/suryabalakrishnan07/bev-yolov8-raycasting.git
cd bev-yolov8-raycasting
```
2. Create and Activate Environment
```
python3 -m venv bev_env
source bev-yolov8-raycasting-env/bin/activate

```
3. Install Dependencies
```
pip install ultralytics nuscenes-devkit pyquaternion tqdm torch torchvision
```
4. Download Weights The trained .pth model weights are hosted in the Releases section. Download the file and place it in the project root directory as yolov8n.pt.

## How to install

NOTE: v1.0-mini dataset should be pre-installed 
To reproduce the evaluation metrics and verify the performance of the geometric radar engine, execute the primary evaluation script:
```
python evaluation.py
```

To generate visualizations and dashboards as seen in the project results, run:
```
python final_dashboard.py
```

## Example Outputs and Results
The system was evaluated using industry-standard metrics for Bird's-Eye-View perception.

### Performance Benchmarks
|Metric|Result|Engineering Significance|
|Throughput|42 FPS|Real-time perfomance on consumer GPUs|
|mAP(Multi-Distance)|0.2787|Consisten detection across 1m, 2m, 4m, and 8m|
|BEV IoU|0.2583|Strong spatial overlap on the top-down map|
|Precision|0.3579|Optimized to minimize false-positive detections|
|Accuracy|13.0%|SOlid baseline for pure-vision geometric projection|

### Visual Results
- fusion_dashboard_final.png: High-resolution output showing the unified 360-degree radar map.
- boston_seaport_perception.mp4: Real-time inference demo in a complex urban environment.

## Declaration

We confirm that the modular 2D-to-3D architecture and the accompanying evaluation suite are original work developed by Team Why Not specifically for the 2026 AI and Computer Vision Challenge.
