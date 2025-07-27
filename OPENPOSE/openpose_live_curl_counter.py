
#!/usr/bin/env python

import argparse, time, math, logging, cv2, numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh, get_graph_path

logging.basicConfig(level=logging.WARNING)

L_SH, L_EL, L_WR = 5, 6, 7  

def angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ang = abs(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) -
                           math.atan2(a[1]-b[1], a[0]-b[0])))
    return 360-ang if ang > 180 else ang

class CurlCounter:
    def __init__(self): self.stage, self.reps = None, 0
    def update(self, ang):
        if ang > 150:      self.stage = "down"
        if ang <  35 and self.stage == "down":
            self.stage = "up"; self.reps += 1
        return self.reps, self.stage

# CLI
p = argparse.ArgumentParser()
p.add_argument('--camera', default=0, help='cam index or file/rtsp path')
p.add_argument('--model',  default='mobilenet_thin',
               help='mobilenet_thin / cmu / mobilenet_v2_large / …')
p.add_argument('--resize', default='432x368',
               help='net input WxH; 0x0 = model default')
p.add_argument('--resize-out-ratio', type=float, default=4.0)
args = p.parse_args()

w, h = model_wh(args.resize)
if w == 0 or h == 0:
    w, h = 432, 368
est = TfPoseEstimator(get_graph_path(args.model),
                      target_size=(w, h))

# Webcam
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened(): raise RuntimeError("Cannot open camera")

counter = CurlCounter()
fps_t0  = time.time()

while True:
    ok, frame = cap.read()
    if not ok: break
    H, W = frame.shape[:2]

    humans = est.inference(frame, resize_to_default=True,
                           upsample_size=args.resize_out_ratio)

    # Pick the most confident person
    if humans:
        h0 = max(humans, key=lambda h: sum(b.score for b in h.body_parts.values()))
        kp  = {bp.part_idx: (bp.x*W, bp.y*H) for bp in h0.body_parts.values()}

        if all(i in kp for i in (L_SH, L_EL, L_WR)):
            a = angle(kp[L_SH], kp[L_EL], kp[L_WR])
            reps, stage = counter.update(a)

            cv2.putText(frame, f"{a:.1f}",
                        tuple(map(int, kp[L_EL])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1, cv2.LINE_AA)

    # Draw skeleton
    frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

    # Rep counter
    cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(frame, 'REPS',  (15,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, str(counter.reps), (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    cv2.putText(frame, 'STAGE', (65,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, counter.stage if counter.stage else '-',
                (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    # FPS
    fps = 1.0 / (time.time() - fps_t0); fps_t0 = time.time()
    cv2.putText(frame, f"{fps:.1f} FPS", (W-100,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('OpenPose (tf-pose) Live', frame)
    if cv2.waitKey(1) & 0xFF == 27: break   

cap.release(); cv2.destroyAllWindows()
