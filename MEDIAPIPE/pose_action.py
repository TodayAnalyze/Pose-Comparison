import torch, numpy as np
from collections import deque
import torch.nn.functional as F

class PoseActionGRU:
    def __init__(self, weight_path, device="cpu"):
        self.hist = deque(maxlen=10)      # last 10 predictions
        self.cur_label = None             # current smoothed label
        ck  = torch.load(weight_path, map_location=device)
        self.win = ck["win"]
        self.gru = torch.nn.GRU(34, 64, batch_first=True)
        self.head= torch.nn.Linear(64, 3)
        self.gru.load_state_dict(ck["gru"])
        self.head.load_state_dict(ck["head"])
        self.gru.eval(); self.head.eval()
        self.dev = device
        self.buf = []                 # ring-buffer of last win frames

    def update(self, kp_flat):
        self.buf.append(kp_flat)
        if len(self.buf) < self.win:
            return None
        if len(self.buf) > self.win:
            self.buf.pop(0)

        with torch.no_grad():
            x = torch.tensor([self.buf], device=self.dev)
            out, _ = self.gru(x)
            logit  = self.head(out[:, -1])
            cls_id = int(logit.argmax().item())

        # Majority vote
        self.hist.append(cls_id)
        if len(self.hist) < self.hist.maxlen:
            return None                     

        return max(set(self.hist), key=self.hist.count)
