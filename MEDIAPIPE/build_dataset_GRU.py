#!/usr/bin/env python

# window = 60 frames  (≈2 s at 30 FPS).  Change --win if needed.
# stride  = 15 frames (75 % overlap).   Change --stride.
# curl_*.npy = 0
# pushup_*.npy = 1
# situp_*.npy = 2

import argparse, pathlib, re, numpy as np

label_map = {'curl':0, 'pushup':1, 'situp':2}

p = argparse.ArgumentParser()
p.add_argument("--np_dir", default="data", help="folder with *.npy pose seqs")
p.add_argument("--win", type=int, default=60)
p.add_argument("--stride", type=int, default=15)
p.add_argument("--out", default="dataset")
args = p.parse_args()

X, y = [], []
for npy in pathlib.Path(args.np_dir).glob("*.npy"):
    cls = None
    for k in label_map:
        if k in npy.stem.lower():
            cls = label_map[k]; break
    if cls is None: continue
    seq = np.load(npy)                    
    for s in range(0, len(seq)-args.win+1, args.stride):
        X.append(seq[s:s+args.win])
        y.append(cls)

X = np.stack(X).astype(np.float32)       
y = np.asarray(y, dtype=np.int64)
pathlib.Path(args.out).mkdir(exist_ok=True)
np.save(f"{args.out}/X.npy", X)
np.save(f"{args.out}/y.npy", y)
print(f"[✓] wrote {args.out}/X.npy  {X.shape},   {args.out}/y.npy  {y.shape}")
