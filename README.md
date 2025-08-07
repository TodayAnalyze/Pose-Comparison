# Pose_Comparison

This project provides a direct real-time comparison of three pose estimation frameworks—**MediaPipe**, **YOLOv11**, and **OpenPose**—for fitness applications with guided corrective feedback. Each model has its own pipeline that runs live through a webcam, monitors key joint angles, and provides rep-counting and posture correction feedback for bodyweight exercises like push-ups, sit-ups, and bicep curls.

---

## 📁 Folder Structure

Each model implementation is located in its own subdirectory:

- `/MediaPipe/`
- `/YOLO/`
- `/OpenPose/`

Each folder contains the relevant scripts and supporting files to run that model's live fitness feedback pipeline.

---

## ▶️ How to Run Each System

Ensure you're running from within the correct directory and that all dependencies are installed (see below).

### 🔹 MediaPipe
```bash
python mediapipe_live_action.py
```

### 🔹 YOLOv11
```bash
python yolo_live_action_guided.py
```

### 🔹 OpenPose
```bash
python openpose_live_action_guided.py
```

Due to GitHub’s 100MB file limit, the following model file had to be excluded from this repository:

```swift

OPENPOSE/models/graph/mobilenet_thin/graph_opt.pb

```

You must manually download this file from the original repository: https://github.com/jiajunhua/ildoonet-tf-pose-estimation/blob/master/models/graph/mobilenet_thin/graph.pb

once downloaded please place it in .../mobilenet_thin.

## 🧰 Dependencies

# All systems require:
 - Python 3.8–3.10.

### 🔹 MediaPipe
 - OpenCV
 - NumPy
   
### 🔹 YOLOv11
 - Ultralytics
 - PyTorch
 - CUDA Toolkit (for GPU support)
 - Used Ultralytics original repo as a starting point: https://github.com/ultralytics/ultralytics

### 🔹 OpenPose
 - Tensorflow
 - graph_otp.pb model file (see note above)
 - Used jiajunhua's original repo as a starting point: https://github.com/jiajunhua/ildoonet-tf-pose-estimation
 - CUDA Toolkit (for GPU support)
  ```bash
git clone https://github.com/jiajunhua/ildoonet-tf-pose-estimation
cd tf-pose-estimation
pip install -r requirements.txt
```
IMPORTANT RECOMMENDATION:
I highly advise to use separate conda environments to run these three pose frameworks since they all require different versions of dependencies so to avoid having to reinstall versions of PyTorch or whatever, just set up each conda environment with the correct dependencies. 


```
NOI
