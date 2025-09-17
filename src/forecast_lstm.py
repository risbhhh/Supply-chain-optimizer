"""Simple LSTM forecasting using PyTorch. Trains on each SKU separately (for clarity)."""
import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import SeriesScaler

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)


class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def create_sequences(data, seq_len=14):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


def train_for_sku(series, epochs=120, lr=1e-3, seq_len=14, device='cpu'):
    scaler = SeriesScaler()
    s = scaler.fit_transform(series)
    X, y = create_sequences(s, seq_len=seq_len)
    X = X.reshape(-1, seq_len, 1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = LSTMForecast()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1).to(device)

    for ep in range(epochs):
        model.train()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 40 == 0:
            model.eval()
            with torch.no_grad():
                valp = model(X_val_t)
                vloss = loss_fn(valp, y_val_t).item()
            print(f"Epoch {ep} train_loss={loss.item():.4f} val_loss={vloss:.4f}")

    # last seq -> forecast horizon
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(X[-1:].astype(np.float32)).to(device)
        fut = model(last_seq).cpu().numpy().flatten()
    fut_inv = scaler.inverse_transform(fut)
    return model, fut_inv


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/demo_demand.csv")
    sku = df['sku'].unique()[0]
    series = df[df['sku']==sku].sort_values('date')['demand'].values
    model, forecast = train_for_sku(series, epochs=100)
    print("Forecast (1-step):", forecast)
