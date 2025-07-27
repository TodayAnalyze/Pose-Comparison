#!/usr/bin/env python

import cv2, argparse, numpy as np, math, time
from ultralytics import YOLO

def angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ang = abs(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) -
                           math.atan2(a[1]-b[1], a[0]-b[0])))
    return 360 - ang if ang > 180 else ang

L_SH, L_EL, L_WR = 5, 7, 9
SKELETON_PAIRS   = [(5,7), (7,9)]     

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model",  default="yolo11m-pose.pt",
                    help="Ultralytics pose checkpoint")
    ap.add_argument("--img_size", type=int, default=640,
                    help="Inference resolution (square)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold")
    ap.add_argument("--device", default=None,
                    help="cuda, cuda:0, cpu  (auto if not set)")
    return ap.parse_args()

def main():
    args   = parse_args()
    model  = YOLO(args.model)
    model.fuse()
    if args.device: model.to(args.device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    counter, stage = 0, None
    font_big  = cv2.FONT_HERSHEY_SIMPLEX
    font_small= cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        t0 = time.time()
        res = model.predict(
            frame, imgsz=args.img_size, conf=args.conf,
            device=args.device, verbose=False)
        latency_ms = (time.time() - t0)*1000

        if res and res[0].keypoints is not None:
            r   = res[0]
            pid = int(r.boxes.conf.argmax())          
            xy  = r.keypoints.xy[pid].cpu().numpy()   
            kp  = xy                                  

            # Curl counter
            a = angle(kp[L_SH], kp[L_EL], kp[L_WR])
            if a > 150:
                stage = "down"
            if a < 35 and stage == "down":
                stage = "up"; counter += 1

            # Draw Skeleton
            for i,j in SKELETON_PAIRS:
                cv2.line(frame, tuple(kp[i].astype(int)),
                                tuple(kp[j].astype(int)), (255,255,255), 2)
            for i in [L_SH, L_EL, L_WR]:
                cv2.circle(frame, tuple(kp[i].astype(int)), 4, (0,0,255), -1)

            # Angle shown
            cv2.putText(frame, f"{a:.1f}", tuple(kp[L_EL].astype(int)),
                        font_small, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Counter overlay
        cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(frame, "REPS",  (15,12), font_small, 0.5, (0,0,0), 1)
        cv2.putText(frame, str(counter), (10,60), font_big, 2, (255,255,255), 2)
        cv2.putText(frame, "STAGE", (65,12), font_small, 0.5, (0,0,0), 1)
        cv2.putText(frame, stage if stage else "-", (60,60),
                    font_big, 2, (255,255,255), 2)

        # FPS display
        cv2.putText(frame, f"{1000/latency_ms:.1f} FPS",
                    (W-100, 30), font_small, 0.6, (0,255,0), 2)

        cv2.imshow("YOLO-Pose Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
