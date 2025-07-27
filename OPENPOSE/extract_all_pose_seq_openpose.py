#!/usr/bin/env python

import argparse, pathlib, cv2, numpy as np, math, logging
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks  import model_wh, get_graph_path

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# COCO-18
LIMBS = 18                        

def extract(video_path, est, resize_ratio=4.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seq = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        humans = est.inference(
            frame, resize_to_default=True, upsample_size=resize_ratio)

        if humans:
            h0   = max(humans,
                       key=lambda h: sum(b.score for b in h.body_parts.values()))
            flat = []
            for idx in range(LIMBS):
                if idx in h0.body_parts:
                    bp = h0.body_parts[idx]
                    flat.extend((bp.x * W, bp.y * H))
                else:
                    flat.extend((0.0, 0.0))
        else:
            flat = [0.0] * (LIMBS * 2)
        seq.append(flat)

    cap.release()
    return np.asarray(seq, dtype=np.float32)


# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True)
ap.add_argument("--out_dir",   default="openpose_vectors")
ap.add_argument("--model",     default="mobilenet_thin")
ap.add_argument("--resize",    default="432x368")
ap.add_argument("--resize_out_ratio", type=float, default=4.0)
args = ap.parse_args()

w, h = model_wh(args.resize)
if w == 0 or h == 0:         
    w, h = 432, 368
est = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

vd = pathlib.Path(args.video_dir)
od = pathlib.Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
mp4s = sorted(vd.glob("*.mp4"))
print(f"[Info] {len(mp4s)} videos")

for v in mp4s:
    npy_path = od / (v.stem + ".npy")
    np.save(npy_path, extract(v, est, args.resize_out_ratio))
    print("  ✓", npy_path.name)

print(f"[Done] saved to {od}")
