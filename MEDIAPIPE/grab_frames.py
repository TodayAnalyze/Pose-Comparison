#!/usr/bin/env python

import cv2, argparse, pathlib, numpy as np

def sample_frames(video_path: pathlib.Path,
                  out_root: pathlib.Path,
                  n_samples: int = 50) -> int:
    """Extract n_samples frames and write JPEGs into out_root / <video_stem>."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError(f"No frames found in {video_path}")

    # Avoid duplicates
    idxs = np.linspace(0, total - 1, num=min(n_samples, total), dtype=int)
    dst_dir = out_root / video_path.stem
    dst_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for frame_id in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] {video_path.name}: couldn’t read frame {frame_id}")
            continue
        cv2.imwrite(str(dst_dir / f"f{frame_id:05}.jpg"), frame)
        saved += 1

    cap.release()
    return saved

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True,
                    help="space-separated list of MP4s")
    ap.add_argument("--per_clip", type=int, default=50,
                    help="frames to sample from each clip (default 50)")
    ap.add_argument("--out_dir", default="frames",
                    help="root folder for extracted JPEGs")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(exist_ok=True)

    for vid in args.videos:
        vid_path = pathlib.Path(vid)
        saved = sample_frames(vid_path, out_root, args.per_clip)
        print(f"[Done] {vid_path.name}: saved {saved}/{args.per_clip} frames")
