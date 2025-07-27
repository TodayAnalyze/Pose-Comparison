
import argparse, pathlib, re
import pandas as pd
import numpy as np

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--log_dir", required=True,
                help="folder that holds *.csv timing logs")
ap.add_argument("--out_csv", default="speed_summary.csv",
                help="output file (will be overwritten)")
args = ap.parse_args()

# Collecting files
log_dir = pathlib.Path(args.log_dir)
csv_files = sorted(log_dir.glob("*.csv"))       
pattern   = re.compile(r"(?P<backend>[a-z0-9]+)[._](?P<clip>[^.]+)\.csv", re.I)

rows = []
for csv_path in csv_files:
    m = pattern.match(csv_path.name)
    if not m:
        print(f"[WARN] skip {csv_path.name} (unexpected name)")
        continue

    backend = m["backend"]                
    clip    = m["clip"]                    

    t = pd.read_csv(csv_path)["infer_ms"]
    mean_ms = t.mean()
    p95_ms  = t.quantile(0.95)
    fps     = 1000.0 / mean_ms

    rows.append(dict(backend=backend,
                     clip=clip,
                     frames=len(t),
                     mean_ms=round(mean_ms, 2),
                     p95_ms=round(p95_ms, 2),
                     fps=round(fps, 1)))

summary = pd.DataFrame(rows)\
            .sort_values(["backend", "clip"])

summary.to_csv(args.out_csv, index=False)
print(f"[Done] wrote {args.out_csv}")
print(summary)
