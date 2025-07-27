#!/usr/bin/env python

import argparse, pathlib, numpy as np, cv2
from ultralytics import YOLO

DEVICE = "cuda:0"                   
yolo = YOLO("yolo11m-pose.pt")      
yolo.to(DEVICE)                     
yolo.fuse()                         

def extract(video_path):
    cap = cv2.VideoCapture(str(video_path))
    seq = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        res = yolo.predict(f, imgsz=640, conf=0.25,
                           device=DEVICE, verbose=False)
        if res and res[0].boxes.shape[0]:
            kp = res[0].keypoints.xy[0].cpu().numpy() 
            flat = kp.flatten().tolist()              
        else:
            flat = [0.0]*34
        seq.append(flat)
    cap.release()
    return np.asarray(seq, np.float32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir",   default="yolo_vectors")
    args = ap.parse_args()

    vd = pathlib.Path(args.video_dir)
    od = pathlib.Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(vd.glob("*.mp4"))
    print(f"[INFO] {len(mp4s)} videos")
    for v in mp4s:
        np.save(od / (v.stem+".npy"), extract(v))
        print(f"  ✓ {v.stem}.npy")

    print("[Done]")

