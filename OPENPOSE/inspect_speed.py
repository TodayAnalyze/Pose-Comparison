import sys, pandas as pd

if len(sys.argv) < 2:
    print("Usage: python inspect_speed.py <csv_path>")
    sys.exit(0)

t = pd.read_csv(sys.argv[1])["infer_ms"]
print(t.describe())                       # count / mean / std / quartiles
print(f"FPS ≈ {1000 / t.mean():.1f}")