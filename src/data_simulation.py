"""Generate synthetic demand time-series for multiple SKUs."""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def generate_demand(n_skus=5, periods=180, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods)
    rows = []
    for sku in range(n_skus):
        base = np.random.randint(20, 100)
        trend = np.linspace(0, np.random.randint(-5, 20), periods)
        season = 10 * np.sin(np.linspace(0, 6 * np.pi, periods) + np.random.rand())
        noise = np.random.normal(0, 5, size=periods)
        demand = np.clip(base + trend + season + noise, a_min=0, a_max=None).round().astype(int)
        df = pd.DataFrame({"date": dates, "sku": f"SKU_{sku+1}", "demand": demand})
        rows.append(df)
    out = pd.concat(rows)
    out.to_csv(OUT / "demo_demand.csv", index=False)
    return out


if __name__ == "__main__":
    df = generate_demand()
    print("Saved:", df.shape)
