#!/usr/bin/env python

import argparse, os, numpy as np, torch, torch.nn as nn, torch.utils.data as td

# CLI
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--data",   default="dataset", help="folder with X.npy / y.npy")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--lr",     type=float, default=1e-3)
ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 …")
ap.add_argument("--save",   default="gru_pose_cls.pt", help="output weight file")
args = ap.parse_args()

dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {dev}")

# Data
X = np.load(os.path.join(args.data, "X.npy")).astype(np.float32)   
y = np.load(os.path.join(args.data, "y.npy")).astype(np.int64)     
N, T, F = X.shape
print(f"[INFO] dataset: {N} windows, window = {T}×{F}")

perm  = np.random.permutation(N)
split = int(0.8 * N)
tr_idx, vl_idx = perm[:split], perm[split:]

tr_ds = td.TensorDataset(torch.from_numpy(X[tr_idx]),
                         torch.from_numpy(y[tr_idx]))
vl_ds = td.TensorDataset(torch.from_numpy(X[vl_idx]),
                         torch.from_numpy(y[vl_idx]))
tr_dl = td.DataLoader(tr_ds, 64, shuffle=True)
vl_dl = td.DataLoader(vl_ds, 64, shuffle=False)

# Model
gru  = nn.GRU(input_size=F, hidden_size=64, batch_first=True).to(dev)
head = nn.Linear(64, 3).to(dev)

opt   = torch.optim.Adam((*gru.parameters(), *head.parameters()), lr=args.lr)
lossf = nn.CrossEntropyLoss()

@torch.no_grad()
def accuracy(loader):
    gru.eval(); head.eval(); ok = 0; tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out,_  = gru(xb); logit = head(out[:,-1])
        ok    += (logit.argmax(1) == yb).sum().item()
        tot   += yb.size(0)
    return ok / tot

# Train
for ep in range(1, args.epochs + 1):
    gru.train(); head.train()
    for xb, yb in tr_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        logit, _ = gru(xb)
        loss = lossf(head(logit[:, -1]), yb)
        opt.zero_grad(); loss.backward(); opt.step()

    if ep % 5 == 0 or ep == args.epochs:
        val_acc = accuracy(vl_dl) * 100
        print(f"epoch {ep:2d}/{args.epochs}  val-acc = {val_acc:.1f}%")

torch.save({"gru": gru.state_dict(),
            "head": head.state_dict(),
            "win": T}, args.save)
print(f"\n[✓] saved {args.save}")
