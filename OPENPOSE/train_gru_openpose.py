#!/usr/bin/env python

import argparse, numpy as np, torch, torch.nn as nn, torch.utils.data as td

p = argparse.ArgumentParser()
p.add_argument("--data",   default="dataset_openpose")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--device", default="cuda:0")
p.add_argument("--save",   default="openpose_gru.pt")
args = p.parse_args()

# Load
X = np.load(f"{args.data}/X.npy")     
y = np.load(f"{args.data}/y.npy")
N, T, F = X.shape                     

dev  = torch.device(args.device)
X_t  = torch.tensor(X, dtype=torch.float32).to(dev)
y_t  = torch.tensor(y, dtype=torch.long).to(dev)

# Split
perm   = torch.randperm(N, device=dev)
split  = int(0.8 * N)
tr_ds  = td.TensorDataset(X_t[perm[:split]],  y_t[perm[:split]])
vl_ds  = td.TensorDataset(X_t[perm[split:]],  y_t[perm[split:]])
tr_dl  = td.DataLoader(tr_ds, 64, shuffle=True)
vl_dl  = td.DataLoader(vl_ds, 64)

# Model
gru  = nn.GRU(F, 64, batch_first=True).to(dev)
head = nn.Linear(64, 3).to(dev)
opt  = torch.optim.Adam([*gru.parameters(), *head.parameters()], 1e-3)
ce   = nn.CrossEntropyLoss()

@torch.no_grad()
def acc(loader):
    gru.eval(); head.eval(); hit = tot = 0
    for xb, yb in loader:
        out, _ = gru(xb)
        logit  = head(out[:, -1])
        hit   += (logit.argmax(1) == yb).sum().item()
        tot   += len(yb)
    return hit / tot

# Train
for ep in range(1, args.epochs + 1):
    gru.train(); head.train()
    for xb, yb in tr_dl:
        out, _ = gru(xb)
        logit  = head(out[:, -1])
        loss   = ce(logit, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 5 == 0 or ep == args.epochs:
        print(f"epoch {ep:2d}/{args.epochs}  val-acc = {acc(vl_dl)*100:.1f}%")

torch.save({"gru":  gru.state_dict(),
            "head": head.state_dict(),
            "win":  T}, args.save)
print(f"\n[✓] saved {args.save}")
