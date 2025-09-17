import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out


def prepare_data(series, lookback=14):
    """
    Convert a demand series into sliding windows for LSTM.
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def train_lstm(df, lookback=14, epochs=20, lr=0.01):
    """
    Train an LSTM on demand data and return model + scaler.
    """
    scaler = MinMaxScaler()
    demand_scaled = scaler.fit_transform(df["demand"].values.reshape(-1, 1))

    X, y = prepare_data(demand_scaled, lookback)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (batch, seq, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model, scaler


def forecast(model, scaler, df, steps=14, lookback=14):
    """
    Forecast future demand using trained LSTM.
    """
    history = scaler.transform(df["demand"].values.reshape(-1, 1)).flatten().tolist()
    preds = []

    for _ in range(steps):
        x = torch.tensor(history[-lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)
        history.append(pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.DataFrame({"date": future_dates, "forecast": preds})


if __name__ == "__main__":
    df = pd.read_csv("data/demo_demand.csv", parse_dates=["date"])
    model, scaler = train_lstm(df)
    forecast_df = forecast(model, scaler, df)
    print(forecast_df.head())
