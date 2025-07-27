#!/usr/bin/env python

import cv2, time, math, numpy as np, argparse, logging
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks  import get_graph_path, model_wh
from pose_action       import PoseActionGRU  

logging.getLogger("tensorflow").setLevel(logging.ERROR)

LABELS = {0:"Curl", 1:"Push-up", 2:"Sit-up"}
LIMBS  = 18
COCO_PAIRS = [(0,1),(1,2),(2,3),(3,4),        # head
              (1,5),(5,6),(6,7),              # left arm
              (2,8),(8,9),(9,10),             # right arm
              (1,11),(11,12),(12,13),         # left leg
              (2,14),(14,15),(15,16)]         # right leg
RED = list(range(LIMBS))

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--model",  default="mobilenet_thin")
ap.add_argument("--resize", default="432x368")
ap.add_argument("--gru",    default="openpose_gru.pt")
ap.add_argument("--camera", default=0)
args = ap.parse_args()
 
w,h = model_wh(args.resize)
if w==0 or h==0: w,h = 432,368
est = TfPoseEstimator(get_graph_path(args.model), target_size=(w,h))
gru = PoseActionGRU(args.gru, device="cpu")   

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened(): raise RuntimeError("no cam")
fps_t0, fps, cur = time.time(), 0, "–"

while True:
    ok, frame = cap.read()
    if not ok: break
    H,W = frame.shape[:2]

    humans = est.inference(frame, resize_to_default=True, upsample_size=4.0)
    if humans:
        h0  = max(humans,
                  key=lambda h: sum(b.score for b in h.body_parts.values()))
        xy  = np.zeros((LIMBS,2), dtype=np.float32)
        vis = np.zeros(LIMBS, dtype=bool)
        for bp in h0.body_parts.values():
            xy[bp.part_idx] = (bp.x*W, bp.y*H)
            vis[bp.part_idx]= True

        # GRU
        xy_flat = xy.flatten().tolist()
        aid = gru.update(xy_flat)
        if aid is not None: cur = LABELS[aid]

        # Drawing kp
        for i,j in COCO_PAIRS:
            if vis[i] and vis[j]:
                cv2.line(frame, tuple(xy[i].astype(int)),
                               tuple(xy[j].astype(int)), (255,255,255), 2)
        for i in RED:
            if vis[i]:
                cv2.circle(frame, tuple(xy[i].astype(int)),
                           4, (0,0,255), -1)

    # UI stuff
    fps = 0.9*fps + 0.1*(1/(time.time()-fps_t0)); fps_t0=time.time()
    cv2.rectangle(frame,(0,0),(200,35),(0,0,0),-1)
    cv2.putText(frame,cur,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,f"{fps:4.1f} FPS",(W-120,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("OpenPose + GRU Action", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release(); cv2.destroyAllWindows()
