import pandas as pd
import matplotlib.pyplot as plt


def load_demand_data(path="data/demo_demand.csv"):
    """
    Load demand data from CSV.
    """
    return pd.read_csv(path, parse_dates=["date"])


def plot_demand(df, title="Demand over time"):
    """
    Quick plot of demand data.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["demand"], marker="o", markersize=2, linestyle="-")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
