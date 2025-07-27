import json, math, pathlib, numpy as np, pandas as pd

YOLO_IDXS = [5, 7, 9]        
NKPT_GT   = 3                 

def get_dims(coco_json):
    img_meta = json.load(open(coco_json))["images"][0]
    return img_meta["width"], img_meta["height"]

def load_gt_3kp(coco_json):
    data = json.load(open(coco_json))
    id2frame = {img["id"]: int(pathlib.Path(img["file_name"]).stem[1:])
                for img in data["images"]}
    gt = {}
    for ann in data["annotations"]:
        f = id2frame[ann["image_id"]]
        k = ann["keypoints"]                 
        gt[f] = [(k[i], k[i+1], k[i+2]) for i in range(0, 9, 3)]
    return gt

def load_pred(pred_json):
    preds = {}
    for fr in json.load(open(pred_json)):
        if fr["kp"]:
            preds[fr["frame"]] = fr["kp"]    
    return preds

def mpjpe(p, g):
    return np.mean([math.hypot(p[i]["x"]-g[i][0], p[i]["y"]-g[i][1])
                    for i in range(NKPT_GT)])

def pck(p, g, thr):
    hits = [math.hypot(p[i]["x"]-g[i][0], p[i]["y"]-g[i][1]) < thr
            for i in range(NKPT_GT)]
    return sum(hits) / NKPT_GT

def elbow_angle(a,b,c):
    A,B,C = np.array(a), np.array(b), np.array(c)
    ang = abs(math.degrees(math.atan2(C[1]-B[1],C[0]-B[0]) -
                           math.atan2(A[1]-B[1],A[0]-B[0])))
    return 360-ang if ang>180 else ang

def evaluate(gt_path, pr_path):
    W, H  = get_dims(gt_path)
    diag  = math.hypot(W, H)
    gt    = load_gt_3kp(gt_path)
    pr    = load_pred(pr_path)
    frames = gt.keys() & pr.keys()

    mpj, p05, p10, ang = [], [], [], []
    for f in frames:
        g3   = gt[f]               
        p17  = pr[f]                
        if len(p17) < max(YOLO_IDXS)+1:
            continue
        p3n  = [p17[i] for i in YOLO_IDXS]    
        p3   = [dict(x=p["x"]*W, y=p["y"]*H, score=p["score"]) for p in p3n]

        mpj.append(mpjpe(p3, g3))
        p05.append(pck(p3, g3, 0.05*diag))
        p10.append(pck(p3, g3, 0.10*diag))

        ang.append(abs(
            elbow_angle(g3[0][:2], g3[1][:2], g3[2][:2]) -
            elbow_angle((p3[0]["x"],p3[0]["y"]),
                        (p3[1]["x"],p3[1]["y"]),
                        (p3[2]["x"],p3[2]["y"]))
        ))

    return dict(frames=len(frames),
                mpjpe = np.mean(mpj),
                pck05 = np.mean(p05),
                pck10 = np.mean(p10),
                ang_err = np.mean(ang))

paths = {
    "slow":   ("gt/slow_curl_gt.json",   "results/yolo_slow_kp.json"),
    "medium": ("gt/medium_curl_gt.json","results/yolo_medium_kp.json"),
    "fast":   ("gt/fast_curl_gt.json",  "results/yolo_fast_kp.json"),
}

rows = [dict(clip=clip, **evaluate(gt, pr))
        for clip, (gt, pr) in paths.items()]

df = pd.DataFrame(rows)
df.to_csv("accuracy_summary_yolo.csv", index=False, float_format="%.2f")
print(df.to_string(index=False, float_format="%.2f"))
print("\n[✓] wrote accuracy_summary_yolo.csv")
