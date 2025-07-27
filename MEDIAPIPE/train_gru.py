#!/usr/bin/env python

import argparse, numpy as np, torch, torch.nn as nn, torch.utils.data as td

p = argparse.ArgumentParser()
p.add_argument("--data",   default="dataset")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--lr",     type=float, default=1e-3)
args = p.parse_args()

X = np.load(f"{args.data}/X.npy")  
y = np.load(f"{args.data}/y.npy")  
N, T, F = X.shape

# Train/val split 80/20
perm  = np.random.permutation(N)
split = int(0.8*N)
tr, vl = perm[:split], perm[split:]
tr_ds  = td.TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr]))
vl_ds  = td.TensorDataset(torch.tensor(X[vl]), torch.tensor(y[vl]))
tr_dl  = td.DataLoader(tr_ds, 64, shuffle=True)
vl_dl  = td.DataLoader(vl_ds, 64, shuffle=False)

# 1-layer GRU
gru   = nn.GRU(input_size=F, hidden_size=64, batch_first=True)
head  = nn.Linear(64, 3)
opt   = torch.optim.Adam(list(gru.parameters())+list(head.parameters()), args.lr)
lossf = nn.CrossEntropyLoss()

def accuracy(loader):
    gru.eval(); head.eval(); n, ok = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out,_ = gru(xb); logit = head(out[:,-1])
            ok   += (logit.argmax(1)==yb).sum().item()
            n    += len(yb)
    return ok/n

for ep in range(1, args.epochs+1):
    gru.train(); head.train()
    for xb, yb in tr_dl:
        out,_ = gru(xb); logit = head(out[:,-1])
        loss  = lossf(logit, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep%5==0 or ep==args.epochs:
        acc = accuracy(vl_dl)
        print(f"epoch {ep:2d}/{args.epochs}  val-acc={acc*100:.1f}%")

torch.save({"gru": gru.state_dict(), "head": head.state_dict(),
            "win": T}, "gru_pose_cls.pt")
print("\n[✓] saved gru_pose_cls.pt")
