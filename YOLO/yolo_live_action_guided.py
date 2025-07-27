#!/usr/bin/env python

import cv2, time, math, numpy as np, argparse
from ultralytics import YOLO
from pose_action import PoseActionGRU    

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--model",  default="yolo11m-pose.pt")
ap.add_argument("--device", default="cuda:0") 
args = ap.parse_args()

# COCO 17 indices
L_SH, L_EL, L_WR = 5, 7, 9
R_SH, R_EL, R_WR = 6, 8,10
R_HIP, R_KNEE, R_ANK = 12,14,16

BAR_X, BAR_Y =  20, 80
BAR_W, BAR_H =  25, 300

CURL_UP , CURL_DN  =  35, 150       
PU_TOP  , PU_BTM   = 150,  75      
LINE_OK           = 170             
SU_DOWN , SU_UP   = 165,  90           
KNEE_MIN_BEND     = 100                

SU_DOWN_ANG = 130      
SU_UP_ANG   = 90        
REP_EPS     = 5    

LABELS = {0:"Curl",1:"Push-up",2:"Sit-up"}

def joint_angle(a,b,c):
    a,b,c = map(np.array,(a,b,c))
    ang = abs(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) -
                           math.atan2(a[1]-b[1], a[0]-b[0])))
    return 360-ang if ang>180 else ang

class RangeTracker:
    def __init__(self, top, btm):
        self.top, self.btm = top, btm
        self.stage = None
        self.reps  = 0
        self.bar   = 0
    def update(self, ang):
        self.bar = np.clip((self.top-ang)/(self.top-self.btm),0,1)*100
        if ang > self.top:
            self.stage = "up"
        if ang < self.btm and self.stage=="up":
            self.stage="down"; self.reps+=1

class SitupTracker:
    def __init__(self):
        self.stage = None          
        self.reps  = 0
        self.bar   = 0

    def update(self, hip_ang):
        # Progress bar
        span = SU_DOWN_ANG - SU_UP_ANG          
        self.bar = np.clip((SU_DOWN_ANG - hip_ang) / span, 0, 1) * 100

        # Rep count
        if hip_ang > SU_DOWN_ANG - REP_EPS:    
            self.stage = "down"

        if hip_ang < SU_UP_ANG + REP_EPS and self.stage == "down":
            self.stage = "up"
            self.reps += 1

curl_gui  = RangeTracker(CURL_DN, CURL_UP)
push_gui  = RangeTracker(PU_TOP , PU_BTM)
sit_gui   = SitupTracker()

# Models
yolo = YOLO(args.model)
yolo.to(args.device).fuse() 
gru  = PoseActionGRU("yolo_gru.pt", device=args.device)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("Webcam not found")

fps, t0, cur_label = 0, time.time(), "–"

while True:
    ok, frame = cap.read()
    if not ok: break
    H,W = frame.shape[:2]

    res = yolo.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    if res.boxes.shape[0]:
        p = int(res.boxes.conf.argmax())       
        xy  = res.keypoints.xy[p].cpu().numpy()   
        cf  = res.keypoints.conf[p].cpu().numpy()  
        flat = xy.flatten().tolist()

        # GRU prediction
        lbl = gru.update(flat)
        if lbl is not None:
            cur_label = LABELS[lbl]

        kp = xy   

        # Curl
        if cur_label=="Curl":
            sh,el,wr = kp[[L_SH,L_EL,L_WR]]
            curl_gui.update(joint_angle(sh,el,wr))
            pct,filled = int(curl_gui.bar), int(BAR_H*curl_gui.bar/100)
            cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),(200,200,200),2)
            cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),(BAR_X+BAR_W-2,BAR_Y+BAR_H-2),
                          (255,80,80),-1)
            cv2.putText(frame,f"{pct}%",(BAR_X-5,BAR_Y+BAR_H+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.rectangle(frame,(W-110,40),(W-40,110),(140,140,255),-1)
            cv2.putText(frame,str(curl_gui.reps),(W-100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)

        # Pushup
        elif cur_label=="Push-up":
            sh,el,wr = kp[[R_SH,R_EL,R_WR]]
            hip,ank  = kp[[R_HIP,R_ANK]]
            push_gui.update(joint_angle(sh,el,wr))
            pct,filled = int(push_gui.bar), int(BAR_H*push_gui.bar/100)
            cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),(200,200,200),2)
            cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),(BAR_X+BAR_W-2,BAR_Y+BAR_H-2),
                          (255,80,80),-1)
            cv2.putText(frame,f"{pct}%",(BAR_X-5,BAR_Y+BAR_H+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.rectangle(frame,(W-110,40),(W-40,110),(140,140,255),-1)
            cv2.putText(frame,str(push_gui.reps),(W-100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)
            # Straight body 
            hip_ln = joint_angle(ank,hip,sh)
            if hip_ln >= LINE_OK:
                cv2.rectangle(frame,(10,50),(290,90),(0,255,0),-1)
                cv2.putText(frame,"Perfect Long Line Body",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            else:
                cv2.rectangle(frame,(10,50),(340,90),(0,0,255),-1)
                cv2.putText(frame,"Attention! Keep body straight",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # Situp
        elif cur_label=="Sit-up":
            sh,hip,kne = kp[[R_SH,R_HIP,R_KNEE]]
            hip_ang  = joint_angle(sh,hip,kne)
            knee_ang = joint_angle(kp[R_ANK],kp[R_KNEE],kp[R_HIP])
            sit_gui.update(hip_ang)
            pct,filled = int(sit_gui.bar), int(BAR_H* sit_gui.bar/100)
            cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),(200,200,200),2)
            cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),(BAR_X+BAR_W-2,BAR_Y+BAR_H-2),
                          (255,80,80),-1)
            cv2.putText(frame,f"{pct}%",(BAR_X-5,BAR_Y+BAR_H+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.rectangle(frame,(W-110,40),(W-40,110),(255,140,255),-1)
            cv2.putText(frame,str(sit_gui.reps),(W-100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)
            # knee bend 
            if knee_ang < KNEE_MIN_BEND:
                cv2.rectangle(frame,(10,50),(200,90),(0,255,0),-1)
                cv2.putText(frame,"Good form",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            else:
                cv2.rectangle(frame,(10,50),(300,90),(0,0,255),-1)
                cv2.putText(frame,"Bend knees more!",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        #  Draw skeleton
        THR = 0.2
        good = (cf>THR) & ((xy[:,0]!=0)|(xy[:,1]!=0))
        for i,j in [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                    (11,12),(11,13),(13,15),(12,14),(14,16)]:
            if good[i] and good[j]:
                cv2.line(frame,tuple(xy[i].astype(int)),tuple(xy[j].astype(int)),
                         (255,255,255),2)
        for i in range(17):
            if good[i]:
                cv2.circle(frame,tuple(xy[i].astype(int)),3,(0,0,255),-1)

    fps = 0.9*fps + 0.1*(1/(time.time()-t0)); t0=time.time()
    cv2.rectangle(frame,(0,0),(180,40),(0,0,0),-1)
    cv2.putText(frame,cur_label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,f"{fps:4.1f} FPS",(W-120,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("YOLO-Pose Guided Exercises",frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
