#!/usr/bin/env python

import cv2, numpy as np, mediapipe as mp, math, time
from pose_action import PoseActionGRU

# GRU weights
WEIGHTS = "gru_pose_cls.pt"      

LABELS  = {0: "Curl", 1: "Push-up", 2: "Sit-up"}   

# Quit key
WIN_KEY = ord("q")                                       

# Progress bar
BAR_X, BAR_Y = 20, 80          
BAR_W, BAR_H = 25, 300

# Joint indices 
L_SH, L_EL, L_WR = 11, 13, 15          
R_SH, R_EL, R_WR = 12, 14, 16          
R_HIP, R_KNEE, R_ANK = 24, 26, 28      

# Curl angle thresholds
CURL_UP_A, CURL_DN_A = 35, 150

# Pushup angle thresholds
PU_EL_TOP, PU_EL_BTM = 150, 75        
LINE_OK = 170                          

# Situp angle thresholds
SU_DOWN_A = 130         
SU_UP_A   = 90         
KNEE_MIN_BEND = 100      
REP_EPS   = 5            


def joint_angle(a, b, c) -> float:
    a, b, c = map(np.array, (a, b, c))
    ang = abs(
        math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) -
                     math.atan2(a[1]-b[1], a[0]-b[0])))
    return 360-ang if ang > 180 else ang

class CurlTracker:
    def __init__(self):
        self.stage, self.reps, self.bar = None, 0, 0
    def update(self, ang):
        self.bar = np.clip((CURL_DN_A-ang)/(CURL_DN_A-CURL_UP_A), 0, 1)*100
        if ang > CURL_DN_A:
            self.stage = "down"
        if ang < CURL_UP_A and self.stage == "down":
            self.stage = "up"; self.reps += 1

class PushupTracker:
    def __init__(self):
        self.stage, self.reps, self.bar = None, 0, 0
    def update(self, ang_elbow):
        self.bar = np.clip((PU_EL_TOP-ang_elbow)/(PU_EL_TOP-PU_EL_BTM),0,1)*100
        if ang_elbow > PU_EL_TOP:
            self.stage = "up"
        if ang_elbow < PU_EL_BTM and self.stage == "up":
            self.stage = "down"; self.reps += 1

class SitupTracker:
    def __init__(self):
        self.stage = None      
        self.reps  = 0
        self.bar   = 0         

    def update(self, hip_ang: float):
        span = SU_DOWN_A - SU_UP_A          
        self.bar = np.clip((SU_DOWN_A - hip_ang) / span, 0, 1) * 100

        if hip_ang > SU_DOWN_A - REP_EPS:   
            self.stage = "down"

        if hip_ang < SU_UP_A + REP_EPS and self.stage == "down":
            self.stage = "up"
            self.reps += 1


sit_gui = SitupTracker()
curl_gui  = CurlTracker()
push_gui  = PushupTracker()

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_est = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

action_net = PoseActionGRU(WEIGHTS, device="cpu")       

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found")

fps_t0, fps = time.time(), 0
cur_label   = "–"

while True:
    ok, frame = cap.read()
    if not ok: break
    H, W = frame.shape[:2]

    res = pose_est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # GRU classification
    if res.pose_landmarks:
        kp = res.pose_landmarks.landmark
        flat = [kp[i].x*W for i in range(17)] + [kp[i].y*H for i in range(17)]
    else:
        flat = [0.]*34
    cls = action_net.update(flat)
    if cls is not None:
        cur_label = LABELS.get(cls, "?")

    # Rule based veto to kill push-up - sit-up flicker 
    if res.pose_landmarks:
        kp = res.pose_landmarks.landmark
        sh, el, wr = [(kp[i].x*W, kp[i].y*H) for i in (R_SH, R_EL, R_WR)]
        hip, kne, ank = [(kp[i].x*W, kp[i].y*H) for i in (R_HIP, R_KNEE, R_ANK)]

        ang_el  = joint_angle(sh, el, wr)       
        hip_ln  = joint_angle(ank, hip, sh)     
        knee_ln = joint_angle(ank, kne, hip)     
        hip_su  = joint_angle(sh, hip, kne)      

        # Pushup veto
        if hip_ln > 158 and knee_ln > 145 and ang_el < 110:
            cur_label = "Push-up";  action_net.hist.clear()

        # Situp veto
        elif knee_ln < 130 and hip_su < 120:
            cur_label = "Sit-up";   action_net.hist.clear()
                
        # Draw the skeleton
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,255),
                                                        thickness=2))

    # Curl corrective feedback
    if cur_label == "Curl" and res.pose_landmarks:
        sh, el, wr = [(kp[i].x*W, kp[i].y*H) for i in (L_SH, L_EL, L_WR)]
        curl_gui.update(joint_angle(sh, el, wr))

        pct = int(curl_gui.bar)
        filled = int(BAR_H*pct/100)
        cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),(200,200,200),2)
        cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),(BAR_X+BAR_W-2,BAR_Y+BAR_H-2),
                      (255,80,80),-1)
        cv2.putText(frame,f"{pct}%",(BAR_X-5,BAR_Y+BAR_H+25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        cv2.rectangle(frame,(W-110,40),(W-40,110),(140,140,255),-1)
        cv2.putText(frame,str(curl_gui.reps),(W-100,100),
                    cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)

    # Pushup corrective feedback
    if cur_label == "Push-up" and res.pose_landmarks:
        sh, el, wr = [(kp[i].x*W, kp[i].y*H) for i in (R_SH, R_EL, R_WR)]
        hip, ank   = [(kp[i].x*W, kp[i].y*H) for i in (R_HIP, R_ANK)]

        ang_el = joint_angle(sh, el, wr)
        hip_ln = joint_angle(ank, hip, sh)

        push_gui.update(ang_el)
        pct_h = int(BAR_H*push_gui.bar/100)

        pct     = int(push_gui.bar)
        filled  = int(BAR_H * pct / 100)

        cv2.rectangle(frame,                   
                    (BAR_X, BAR_Y),
                    (BAR_X + BAR_W, BAR_Y + BAR_H),
                    (200, 200, 200), 2)

        cv2.rectangle(frame,                     
                    (BAR_X + 2, BAR_Y + BAR_H - filled),
                    (BAR_X + BAR_W - 2, BAR_Y + BAR_H - 2),
                    (255, 80, 80), -1)

        cv2.putText(frame,                       
                    f"{pct:3d}%", (BAR_X - 5, BAR_Y + BAR_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame,(W-110,40),(W-40,110),(140,140,255),-1)
        
        cv2.putText(frame,str(push_gui.reps),(W-100,100),
                    cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,0,0),4)

        if hip_ln >= LINE_OK:
            cv2.rectangle(frame,(10,50),(290,90),(0,255,0),-1)
            cv2.putText(frame,"Perfect Long Line Body",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        else:
            cv2.rectangle(frame,(10,50),(340,90),(0,0,255),-1)
            cv2.putText(frame,"Attention! Keep body straight",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            
    # Situp corrective feedback      
    if cur_label == "Sit-up" and res.pose_landmarks:
        sh, hip, kne = [(kp[i].x*W, kp[i].y*H) for i in (R_SH, R_HIP, R_KNEE)]
        hip_ang = joint_angle(sh, hip, kne)         
        knee_ang = knee_ln                          

        sit_gui.update(hip_ang)

        pct = int(sit_gui.bar); filled = int(BAR_H*pct/100)
        cv2.rectangle(frame,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),(200,200,200),2)
        cv2.rectangle(frame,(BAR_X+2,BAR_Y+BAR_H-filled),(BAR_X+BAR_W-2,BAR_Y+BAR_H-2),
                    (255,80,80),-1)
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

    fps = 0.9*fps + 0.1*(1/(time.time()-fps_t0));  fps_t0=time.time()
    cv2.rectangle(frame,(0,0),(180,40),(0,0,0),-1)
    cv2.putText(frame,cur_label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,f"{fps:4.1f} FPS",(W-120,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("MediaPipe Guided Exercises",frame)
    if cv2.waitKey(1) & 0xFF == WIN_KEY:
        break

cap.release(); cv2.destroyAllWindows()
