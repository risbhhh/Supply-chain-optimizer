import pandas as pd
import numpy as np


def generate_demand_data(n_days=180, seed=42):
    """
    Generate synthetic daily demand data with seasonality + randomness.
    """
    np.random.seed(seed)
    days = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    base_demand = 50 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_days))  # seasonality
    noise = np.random.normal(0, 5, n_days)  # random fluctuation
    demand = np.maximum(10, base_demand + noise).astype(int)

    df = pd.DataFrame({"date": days, "demand": demand})
    return df


if __name__ == "__main__":
    df = generate_demand_data()
    df.to_csv("data/demo_demand.csv", index=False)
    print("✅ Demand data saved to data/demo_demand.csv")
