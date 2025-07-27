#!/usr/bin/env python

import cv2, time, math, numpy as np, argparse, logging
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks  import get_graph_path, model_wh
from pose_action       import PoseActionGRU

logging.getLogger("tensorflow").setLevel(logging.ERROR)

LABELS = {0: "Curl", 1: "Push-up", 2: "Sit-up"}

# COCO-18 order 
L_SH, L_EL, L_WR =  5,  6,  7     
R_SH, R_EL, R_WR =  2,  3,  4      
R_HIP,R_KNEE,R_ANK =  8,  9, 10

COCO_PAIRS = [(0,1),(1,2),(2,3),(3,4),
              (1,5),(5,6),(6,7),
              (2,8),(8,9),(9,10),
              (1,11),(11,12),(12,13),
              (8,11)]                

BAR_X, BAR_Y, BAR_W, BAR_H = 20, 80, 25, 300

CURL_UP , CURL_DN  =  35, 150
PU_TOP  , PU_BTM   = 150, 75
LINE_OK           = 170
SU_DOWN , SU_UP   = 165,  90
KNEE_MIN_BEND     = 100

# Some smoothing
ALPHA   = 0.7                      

def joint_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ang = abs(math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) -
        math.atan2(a[1]-b[1], a[0]-b[0])))
    return 360-ang if ang > 180 else ang

class RangeTracker:
    def __init__(self, top, btm):
        self.top, self.btm = top, btm
        self.stage = None; self.reps = 0; self.bar = 0
    def update(self, ang):
        self.bar = np.clip((self.top-ang)/(self.top-self.btm),0,1)*100
        if ang > self.top: self.stage = "up"
        if ang < self.btm and self.stage == "up":
            self.stage = "down"; self.reps += 1

class SitupTracker:
    LOW  =  90    
    HIGH = 130      
    def __init__(self):
        self.stage = None   
        self.reps  = 0
        self.bar   = 0

    def update(self, hip_ang):
        self.bar = np.clip((self.HIGH - hip_ang) /
                           (self.HIGH - self.LOW), 0, 1) * 100

        if hip_ang > self.HIGH - 5:       
            self.stage = "down"
        if hip_ang < self.LOW and self.stage == "down":
            self.stage = "up"
            self.reps += 1


curl_gui = RangeTracker(CURL_DN, CURL_UP)
push_gui = RangeTracker(PU_TOP , PU_BTM)
sit_gui  = SitupTracker()

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--model",  default="mobilenet_thin",
                help="cmu / mobilenet_thin / mobilenet_v2_large …")
ap.add_argument("--resize", default="432x368")
ap.add_argument("--gru",    default="openpose_gru.pt")
ap.add_argument("--camera", default=0)
args = ap.parse_args()

w, h = model_wh(args.resize)
if w==0 or h==0: w, h = 432, 368
est = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
gru = PoseActionGRU(args.gru, device="cpu")          

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened(): raise RuntimeError("camera not found")

prev_xy = None
fps_t0, fps, cur = time.time(), 0, "–"

while True:
    ok, frame = cap.read()
    if not ok: break
    H, W = frame.shape[:2]

    humans = est.inference(frame, resize_to_default=True, upsample_size=4.0)
    if humans:
        h0 = max(humans, key=lambda h: sum(b.score for b in h.body_parts.values()))
        xy  = np.zeros((18,2), np.float32)
        vis = np.zeros(18,     bool)
        for bp in h0.body_parts.values():
            xy[bp.part_idx]  = (bp.x*W, bp.y*H)
            vis[bp.part_idx] = True

        # Smoothing
        if prev_xy is None: prev_xy = xy.copy()
        else:               prev_xy[vis] = ALPHA*xy[vis] + (1-ALPHA)*prev_xy[vis]
        kp = prev_xy.copy()
        kp[~vis] = 0                                      

        # GRU
        aid = gru.update(kp.flatten().tolist())
        if aid is not None: cur = LABELS[aid]

        #Curl
        if cur=="Curl":
            sh, el, wr = kp[[L_SH,L_EL,L_WR]]
            curl_gui.update(joint_angle(sh,el,wr))
            pct = int(curl_gui.bar); filled = int(BAR_H*pct/100)
            cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),
                          (200,200,200),2)
            cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),
                          (BAR_X+BAR_W-2,BAR_Y+BAR_H-2),(255,80,80),-1)
            cv2.putText(frame,f"{pct}%",(BAR_X-5,BAR_Y+BAR_H+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.rectangle(frame,(W-110,40),(W-40,110),(140,140,255),-1)
            cv2.putText(frame,str(curl_gui.reps),(W-100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)

        # Pushup
        elif cur == "Push-up":
            sh, el, wr = kp[[R_SH, R_EL, R_WR]]     
            hip, ank   = kp[[R_HIP, R_ANK]]         

            ang_el = joint_angle(sh, el, wr)       
            hip_ln = joint_angle(ank, hip, sh)      

            el_px = tuple(el.astype(int))          
            cv2.putText(frame, f"{int(ang_el):3d}°",
                        (el_px[0] - 30, el_px[1] - 10),   
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # Progress bar and Reps
            push_gui.update(ang_el)                
            pct  = int(push_gui.bar)
            filled = int(BAR_H * pct / 100)

            cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                        (200, 200, 200), 2)
            cv2.rectangle(frame, (BAR_X + 2, BAR_Y + BAR_H - filled),
                        (BAR_X + BAR_W - 2, BAR_Y + BAR_H - 2),
                        (255, 80, 80), -1)
            cv2.putText(frame, f"{pct}%", (BAR_X - 5, BAR_Y + BAR_H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.rectangle(frame, (W - 110, 40), (W - 40, 110), (140, 140, 255), -1)
            cv2.putText(frame, str(push_gui.reps), (W - 100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)

            # Body straight?
            if hip_ln >= LINE_OK:
                cv2.rectangle(frame, (10, 50), (290, 90), (0, 255, 0), -1)
                cv2.putText(frame, "Perfect Long Line Body", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                cv2.rectangle(frame, (10, 50), (340, 90), (0, 0, 255), -1)
                cv2.putText(frame, "Attention! Keep body straight", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif cur=="Sit-up":
            sh, hip, kne = kp[[R_SH, R_HIP, R_KNEE]]

            hip_ang  = joint_angle(sh, hip, kne)     
            knee_ang = joint_angle(kp[R_ANK], kp[R_KNEE], kp[R_HIP])

            hip_px = tuple(hip.astype(int))              
            cv2.putText(frame, f"{int(hip_ang):3d}°",
                        (hip_px[0] - 30, hip_px[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            sit_gui.update(hip_ang)
            pct  = int(sit_gui.bar)
            filled = int(BAR_H * pct / 100)
            cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),
                          (200,200,200),2)
            cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),
                          (BAR_X+BAR_W-2,BAR_Y+BAR_H-2),(255,80,80),-1)
            cv2.putText(frame,f"{pct}%",(BAR_X-5,BAR_Y+BAR_H+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.rectangle(frame,(W-110,40),(W-40,110),(255,140,255),-1)
            cv2.putText(frame,str(sit_gui.reps),(W-100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)
            if knee_ang < KNEE_MIN_BEND:
                cv2.rectangle(frame,(10,50),(200,90),(0,255,0),-1)
                cv2.putText(frame,"Good form",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            else:
                cv2.rectangle(frame,(10,50),(300,90),(0,0,255),-1)
                cv2.putText(frame,"Bend knees more!",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        # Draw skeleton
        for i,j in COCO_PAIRS:
            if vis[i] and vis[j]:
                cv2.line(frame,tuple(kp[i].astype(int)),tuple(kp[j].astype(int)),
                         (255,255,255),2)
        for i in range(18):
            if vis[i]:
                cv2.circle(frame,tuple(kp[i].astype(int)),3,(0,0,255),-1)

    # FPS
    fps = 0.9*fps + 0.1*(1/(time.time()-fps_t0)); fps_t0=time.time()
    cv2.rectangle(frame,(0,0),(200,35),(0,0,0),-1)
    cv2.putText(frame,cur,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,f"{fps:4.1f} FPS",(W-120,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("OpenPose Guided Exercises", frame)
    if cv2.waitKey(1) & 0xFF == 27: break   

cap.release(); cv2.destroyAllWindows()
