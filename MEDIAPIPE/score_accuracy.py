#!/usr/bin/env python

import json, math, pathlib, argparse
import numpy as np
import pandas as pd


def get_dims(gt_json):
    with open(gt_json, "r") as jf:
        img_meta = json.load(jf)["images"][0]
    return img_meta["width"], img_meta["height"]

 # shoulder, elbow, wrist
MP_IDXS = [11, 13, 15]      

def load_gt(coco_json):
    data = json.load(open(coco_json))
    id2frame = {img["id"]: int(pathlib.Path(img["file_name"]).stem[1:])
                for img in data["images"]}
    gt = {}
    for ann in data["annotations"]:
        f = id2frame[ann["image_id"]]
        kp = ann["keypoints"]
        gt[f] = [(kp[i], kp[i+1], kp[i+2]) for i in range(0, len(kp), 3)]
    return gt


def load_pred(pred_json, w, h):
    preds = {}
    for f in json.load(open(pred_json)):
        if not f["kp"]:
            continue
        full = f["kp"]
        triplet = [(full[i]["x"] * w,
                    full[i]["y"] * h,
                    full[i]["score"]) for i in MP_IDXS]
        preds[f["frame"]] = triplet
    return preds


def mpjpe(px, gx):
    return np.mean([math.hypot(p[0]-g[0], p[1]-g[1])
                    for p, g in zip(px, gx) if g[2] > 0])


def pck(px, gx, thr):
    correct = sum(math.hypot(p[0]-g[0], p[1]-g[1]) < thr
                  for p, g in zip(px, gx) if g[2] > 0)
    total   = sum(g[2] > 0 for g in gx)
    return correct / total if total else np.nan


def elbow_angle(a, b, c):
    A, B, C = map(np.array, (a, b, c))
    ang = abs(np.arctan2(C[1]-B[1], C[0]-B[0]) -
              np.arctan2(A[1]-B[1], A[0]-B[0])) * 180 / np.pi
    return 360-ang if ang > 180 else ang


def evaluate(gt_path, pred_path):
    w, h   = get_dims(gt_path)
    diag   = math.hypot(w, h)
    gt     = load_gt(gt_path)
    pred   = load_pred(pred_path, w, h)
    common = gt.keys() & pred.keys()

    mpj, p05, p10, ang = [], [], [], []
    for f in common:
        gkp, pkp = gt[f], pred[f]
        mpj.append(mpjpe(pkp, gkp))
        p05.append(pck(pkp, gkp, 0.05*diag))
        p10.append(pck(pkp, gkp, 0.10*diag))
        ang.append(abs(elbow_angle(*[g[:2] for g in gkp[:3]]) -
                       elbow_angle(*[p[:2] for p in pkp[:3]])))
    return dict(frames=len(common),
                mpjpe=np.nanmean(mpj),
                pck05=np.nanmean(p05),
                pck10=np.nanmean(p10),
                ang_err=np.nanmean(ang))


if __name__ == "__main__":
    paths = {
        "slow":   ("gt/slow_curl_gt.json",   "logs/mp_slow.json"),
        "medium": ("gt/medium_curl_gt.json", "logs/mp_medium.json"),
        "fast":   ("gt/fast_curl_gt.json",   "logs/mp_fast.json"),
    }

    rows = []
    for clip, (gt_p, pred_p) in paths.items():
        rows.append(dict(clip=clip, **evaluate(gt_p, pred_p)))

    df = pd.DataFrame(rows)
    df.to_csv("accuracy_summary_mp.csv", index=False, float_format="%.2f")
    print(df.to_string(index=False, float_format="%.2f"))
    print("\n[✓] wrote accuracy_summary_mp.csv")
