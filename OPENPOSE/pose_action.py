# pose_action.py
import torch, numpy as np
from collections import deque

class PoseActionGRU:
    """
    Stream-friendly wrapper around a 1-layer GRU + Linear classifier.

    • Input size is read from the checkpoint → works for 34-D (MediaPipe/YOLO)
      **and** 36-D (OpenPose) without edits.
    • Majority-vote smoothing over the last 10 predictions.
    """

    def __init__(self, weight_path: str, device: str = "cpu", hist_len: int = 10):
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---------- load checkpoint ----------
        ckpt   = torch.load(weight_path, map_location="cpu")        # safe load
        self.win = ckpt["win"]                                      # window length

        # Infer feature dimension from GRU weight shape
        feat_dim = ckpt["gru"]["weight_ih_l0"].shape[1]             # 34 or 36 …

        # ---------- build net ----------
        self.gru  = torch.nn.GRU(feat_dim, 64, batch_first=True).to(self.dev)
        self.head = torch.nn.Linear(64, 3).to(self.dev)
        self.gru.load_state_dict(ckpt["gru"])
        self.head.load_state_dict(ckpt["head"])
        self.gru.eval(); self.head.eval()

        # ---------- buffers ----------
        self.buf  = []                       # last <win> frames
        self.hist = deque(maxlen=hist_len)   # smoothing
        self.cur_label = None

    # ------------------------------------------------------------------
    def update(self, kp_flat):
        """
        kp_flat : iterable length = feature-dim (34 or 36), values can be NaN
        returns  : smoothed class id (0/1/2)  or  None while warming up
        """
        self.buf.append(kp_flat)
        if len(self.buf) < self.win:
            return None
        if len(self.buf) > self.win:
            self.buf.pop(0)

        with torch.no_grad():
            x = torch.tensor([self.buf], dtype=torch.float32, device=self.dev)
            out, _ = self.gru(x)
            logit  = self.head(out[:, -1])
            cls_id = int(logit.argmax(1).item())

        # majority vote
        self.hist.append(cls_id)
        if len(self.hist) < self.hist.maxlen:
            return None
        self.cur_label = max(set(self.hist), key=self.hist.count)
        return self.cur_label
