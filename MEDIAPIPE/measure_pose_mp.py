import cv2, time, json, csv, argparse, pathlib
import mediapipe as mp
import numpy as np

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="path to MP4")
ap.add_argument("--out_kp", required=True, help="JSON keypoints file")
ap.add_argument("--out_log", required=True, help="CSV timing log")
args = ap.parse_args()

def calculate_angle(a,b,c):
    a,b,c = map(np.array,(a,b,c))
    ang = np.abs(np.arctan2(c[1]-b[1],c[0]-b[0]) - 
                 np.arctan2(a[1]-b[1],a[0]-b[0]))*180/np.pi
    return 360-ang if ang>180 else ang

def kp_to_dict(landmarks):
    """Return list[{'x':..,'y':..,'score':..}] in image coords."""
    return [{"x": lm.x, "y": lm.y, "score": lm.visibility} 
            for lm in landmarks]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Webcam stuff
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] video opened at {fps:.1f} FPS")

time_log   = []          # per-frame inference ms
all_kp     = []          # per-frame keypoints JSON
rep_cnt    = 0
stage      = None

frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    results = pose.process(rgb)
    infer_ms = (time.perf_counter() - t0)*1000
    time_log.append(infer_ms)

    if results.pose_landmarks:                
        lms = results.pose_landmarks.landmark
        kp_list = kp_to_dict(lms)
        all_kp.append({"frame":frame_id, "kp":kp_list})

        sh = kp_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        el = kp_list[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wr = kp_list[mp_pose.PoseLandmark.LEFT_WRIST.value]
        angle = calculate_angle((sh["x"],sh["y"]),
                                (el["x"],el["y"]),
                                (wr["x"],wr["y"]))
        if angle > 150: stage = "down"
        if angle < 35 and stage=="down":
            stage="up"; rep_cnt += 1
    else:
        all_kp.append({"frame":frame_id, "kp":[]})

    frame_id += 1

pathlib.Path(args.out_kp).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_kp, "w") as jf:
    json.dump(all_kp, jf)      

with open(args.out_log, "w", newline="") as cf:   
    writer = csv.writer(cf)
    writer.writerow(["frame","infer_ms"])
    for i,ms in enumerate(time_log):
        writer.writerow([i, f"{ms:.3f}"])

print(f"[Done] frames: {frame_id} | mean {np.mean(time_log):.1f} ms  "
      f"| reps counted = {rep_cnt}")
