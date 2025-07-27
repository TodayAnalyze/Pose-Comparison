import pandas as pd
t = pd.read_csv("logs/mp.fast.csv")["infer_ms"]
print(t.describe())         # mean, p50, p95 …
print(f"FPS ≈ {1000 / t.mean():.1f}")
