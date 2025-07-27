#!/usr/bin/env python

import cv2, time, json, csv, argparse, pathlib, sys, math, numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def angle(a, b, c):
    """Return ∠abc in degrees for three (x,y) points."""
    a, b, c = map(np.array, (a, b, c))
    ang = abs(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) -
                           math.atan2(a[1]-b[1], a[0]-b[0])))
    return 360 - ang if ang > 180 else ang

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--video",  required=True, help="Path to MP4/AVI")
    ap.add_argument("--model",  default="yolo11m-pose.pt",
                    help="Ultralytics-YOLO pose checkpoint (*.pt)")
    ap.add_argument("--img_size", type=int, default=960,
                    help="Resize long side before inference")
    ap.add_argument("--out_kp", required=True, help="Output JSON file")
    ap.add_argument("--out_log", required=True, help="Output CSV  file")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu  (auto)")
    ap.add_argument("--conf_thres", type=float, default=0.25)
    return ap.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.fuse()
    if args.device:
        model.to(args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERR] cannot open video: {args.video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] video opened at {fps_in:.1f} FPS")

    time_log, all_kp = [], []
    rep_cnt, stage = 0, None
    frame_id = 0
    NKPT = 17
    L_SH, L_EL, L_WR = 5, 7, 9   

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar  = tqdm(total=total or None, unit="f", desc="processing")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        orig_h, orig_w = frame.shape[:2]

        t0 = time.perf_counter()
        results = model.predict(frame,
                                imgsz=args.img_size,
                                conf=args.conf_thres,
                                verbose=False,
                                device=args.device)
        infer_ms = (time.perf_counter() - t0)*1000
        time_log.append(infer_ms)

        if results and results[0].keypoints is not None:
            # Take the highest-confidence person
            r = results[0]
            person_id = int(r.boxes.conf.argmax())
            kp_xy_t   = r.keypoints.xy[person_id].cpu()     
            kp_conf_t = r.keypoints.conf[person_id].cpu()    
            kp_xy     = kp_xy_t.numpy()                     
            kp_conf   = kp_conf_t.numpy()                    

            kp_list = [{"x":  float(kp_xy[i,0]/orig_w),
                        "y":  float(kp_xy[i,1]/orig_h),
                        "score": float(kp_conf[i])}
                       for i in range(NKPT)]

            # Curl counter
            sh, el, wr = kp_xy[[L_SH, L_EL, L_WR]]
            a = angle(sh, el, wr)
            if a > 150:
                stage = "down"
            if a < 35 and stage == "down":
                stage = "up"; rep_cnt += 1
        else:
            kp_list = []

        all_kp.append({"frame": frame_id, "kp": kp_list})
        frame_id += 1
        pbar.update(1)
    pbar.close()

    pathlib.Path(args.out_kp).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_kp, "w") as jf: json.dump(all_kp, jf)

    with open(args.out_log, "w", newline="") as cf:
        wr = csv.writer(cf); wr.writerow(["frame","infer_ms"])
        for i, ms in enumerate(time_log):
            wr.writerow([i, f"{ms:.3f}"])

    mean_ms = np.mean(time_log)
    p95_ms  = np.percentile(time_log, 95)
    print(f"[DONE] frames: {frame_id} | mean {mean_ms:.1f} ms "
          f"| p95 {p95_ms:.1f} ms | reps counted = {rep_cnt}")

if __name__ == "__main__":
    main()
