import torch, numpy as np
from collections import deque

class PoseActionGRU:
    def __init__(self, weight_path, device="cpu"):
        self.dev = torch.device(device)         

        ckpt = torch.load(weight_path, map_location="cpu")  
        self.win = ckpt["win"]                              

        self.gru  = torch.nn.GRU(34, 64, batch_first=True).to(self.dev)
        self.head = torch.nn.Linear(64, 3).to(self.dev)

        self.gru.load_state_dict(ckpt["gru"])
        self.head.load_state_dict(ckpt["head"])
        self.gru.eval();  self.head.eval()

        self.buf  = []                      
        self.hist = deque(maxlen=10)           
        self.cur_label = None

    def update(self, kp_flat):
        self.buf.append(kp_flat)
        if len(self.buf) < self.win:
            return None
        if len(self.buf) > self.win:
            self.buf.pop(0)

        with torch.no_grad():
            x = torch.tensor([self.buf], dtype=torch.float32, device=self.dev) 
            out, _ = self.gru(x)
            logit  = self.head(out[:, -1]) 
            cls_id = int(logit.argmax(dim=1).item())

        self.hist.append(cls_id)
        if len(self.hist) < self.hist.maxlen:  
            return None

        self.cur_label = max(set(self.hist), key=self.hist.count)
        return self.cur_label
