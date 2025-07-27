#!/usr/bin/env python

import cv2, time, json, csv, argparse, pathlib
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh

def elbow_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    theta = abs(
        np.arctan2(c[1]-b[1], c[0]-b[0]) -
        np.arctan2(a[1]-b[1], a[0]-b[0])
    ) * 180 / np.pi
    return 360 - theta if theta > 180 else theta

counter = 0       
stage   = None     

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--video",    required=True)
ap.add_argument("--out_kp",   required=True)
ap.add_argument("--out_log",  required=True)
ap.add_argument("--model",    required=True,
                help="path to mobilenet_thin.pb or cmu.pb")
ap.add_argument("--resize",   default="320x176",
                help="WxH for estimator, e.g. 432x368, or 0x0 to keep orig")
args = ap.parse_args()

w, h = model_wh(args.resize)
if w == 0 or h == 0:
    w = h = None                   
estimator = TfPoseEstimator(args.model, target_size=(w, h) if w else (432,368),
                            tf_config=None)  

L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7

# Webcam setup
cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS)
print(f"[Info] video opened at {fps_in:.1f} FPS")

kp_log, time_log = [], []
frame_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.perf_counter()
    humans = estimator.inference(frame, resize_to_default=(w is not None),
                                 upsample_size=4.0)
    infer_ms = (time.perf_counter() - t0) * 1000
    time_log.append(infer_ms)

    # Picks 3 joints from first detected person
    kp_triplet = []
    if humans:
        h0 = humans[0]
        for part_idx in (L_SHOULDER, L_ELBOW, L_WRIST):
            bp = h0.body_parts.get(part_idx, None)
            if bp:
                kp_triplet.append({"x": bp.x * frame.shape[1],
                                   "y": bp.y * frame.shape[0],
                                   "score": bp.score})
            else:                    
                kp_triplet.append({"x": 0, "y": 0, "score": 0.0})

    # Curl counter
    if len(kp_triplet) == 3 and all(k["score"] > 0 for k in kp_triplet):
        sh = (kp_triplet[0]["x"], kp_triplet[0]["y"])
        el = (kp_triplet[1]["x"], kp_triplet[1]["y"])
        wr = (kp_triplet[2]["x"], kp_triplet[2]["y"])
        ang = elbow_angle(sh, el, wr)

        if ang > 150:        
            stage = "down"
        if ang < 35 and stage == "down":
            stage = "up"
            counter += 1

    kp_log.append({"frame": frame_id, "kp": kp_triplet})
    frame_id += 1

print(f"[DONE] frames: {frame_id} | mean {np.mean(time_log):.1f} ms "
      f"| reps counted = {counter}")

pathlib.Path(args.out_kp).parent.mkdir(parents=True, exist_ok=True)
json.dump(kp_log, open(args.out_kp, "w"))
with open(args.out_log, "w", newline="") as cf:
    wr = csv.writer(cf); wr.writerow(["frame", "infer_ms"])
    for i, ms in enumerate(time_log):
        wr.writerow([i, f"{ms:.3f}"])
