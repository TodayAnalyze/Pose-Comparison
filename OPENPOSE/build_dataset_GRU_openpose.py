#!/usr/bin/env python

# window = 60 frames  (≈2 s at 30 FPS).  Change --win if needed.
# stride  = 15 frames (75 % overlap).   Change --stride.
# curl_*.npy = 0
# pushup_*.npy = 1
# situp_*.npy = 2

import argparse, pathlib, re, numpy as np

LABEL = {'curl': 0, 'pushup': 1, 'situp': 2}

ap = argparse.ArgumentParser()
ap.add_argument("--np_dir",  required=True,
                help="folder containing *.npy pose-sequence files")
ap.add_argument("--win",     type=int, default=60,
                help="window length (frames)")
ap.add_argument("--stride",  type=int, default=15,
                help="stride between windows")
ap.add_argument("--out",     default="dataset_openpose",
                help="output folder for X.npy / y.npy")
args = ap.parse_args()

X, y       = [], []
first_feat = None                  

for file in sorted(pathlib.Path(args.np_dir).glob("*.npy")):
    cls = next((LABEL[k] for k in LABEL if k in file.stem.lower()), None)
    if cls is None:
        print(f"[skip] {file.name} (label not recognised)")
        continue

    seq = np.load(file)                 
    if first_feat is None:
        first_feat = seq.shape[1]
        print(f"[Info] feature dimension detected: {first_feat}")

    if seq.shape[1] != first_feat:
        print(f"[Warn] {file.name} has {seq.shape[1]} features;"
              f" expected {first_feat}.  Skipping.")
        continue

    T = seq.shape[0]
    for start in range(0, T - args.win + 1, args.stride):
        X.append(seq[start : start + args.win])
        y.append(cls)

X = np.stack(X).astype(np.float32)    
y = np.asarray(y, dtype=np.int64)

out_dir = pathlib.Path(args.out); out_dir.mkdir(exist_ok=True)
np.save(out_dir / "X.npy", X)
np.save(out_dir / "y.npy", y)
print(f"[✓] wrote {out_dir/'X.npy'}  {X.shape},   {out_dir/'y.npy'}  {y.shape}")
