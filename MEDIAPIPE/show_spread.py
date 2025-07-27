import pandas as pd

for clip in ["slow","medium","fast"]:
    t = pd.read_csv(f"logs/mp.{clip}.csv")["infer_ms"]
    print(f"{clip:<6} mean={t.mean():5.1f}ms  p95={t.quantile(0.95):5.1f}ms  fps≈{1000/t.mean():4.1f}")
