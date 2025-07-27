#!/usr/bin/env python

import argparse, pathlib, cv2, numpy as np, mediapipe as mp

mp_pose = mp.solutions.pose

def extract(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            kp = res.pose_landmarks.landmark
            flat = [kp[i].x * W for i in range(17)] + \
                   [kp[i].y * H for i in range(17)]
        else:
            flat = [0.0] * 34
        seq.append(flat)
    cap.release()
    pose.close()
    return np.asarray(seq, dtype=np.float32)

# CLI

p = argparse.ArgumentParser()
p.add_argument("--video_dir", required=True, help="folder with *.mp4 clips")
p.add_argument("--out_dir",   default="mp_extracted_vectors",
               help="where to write *.npy files")
args = p.parse_args()

video_dir = pathlib.Path(args.video_dir)
out_dir   = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

mp4s = sorted(video_dir.glob("*.mp4"))
print(f"[INFO] found {len(mp4s)} videos in {video_dir}")

for vid in mp4s:
    out_path = out_dir / (vid.stem + ".npy")
    arr = extract(vid)
    np.save(out_path, arr)
    print(f"[✓] {out_path.name}  shape={arr.shape}")

print(f"\n[Done] all pose vectors saved to {out_dir}")
