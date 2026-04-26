from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/home/kwang/RAM-FinGAN")
PANEL_FILE = ROOT / "data_clean" / "ram_panel" / "ALL_STOCKS_panel.csv"
OUT_DIR = ROOT / "outputs" / "results" / "baselines"
MODEL_DIR = ROOT / "outputs" / "models" / "baselines"
LOG_DIR = ROOT / "outputs" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
HIDDEN = 32
TICKER_EMB = 8
MARKET_HIDDEN = 32
PATIENCE = 6

X_COLS = [f"x_{i}" for i in range(1, 11)]
ID_COLS = {"date", "ticker", "sector_etf", "split", "y"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PanelDataset(Dataset):
    def __init__(self, df, market_cols, ticker_to_id, use_market, x_mean, x_std, m_mean=None, m_std=None):
        self.df = df.reset_index(drop=True)
        self.use_market = use_market
        self.market_cols = market_cols
        self.ticker_to_id = ticker_to_id

        x = self.df[X_COLS].values.astype(np.float32)
        self.x = (x - x_mean) / x_std
        self.x = self.x[:, :, None]

        self.y = self.df["y"].values.astype(np.float32)[:, None]
        self.ticker_id = self.df["ticker"].map(ticker_to_id).values.astype(np.int64)

        if use_market:
            m = self.df[market_cols].values.astype(np.float32)
            self.m = (m - m_mean) / m_std
        else:
            self.m = np.zeros((len(self.df), 1), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "m": torch.tensor(self.m[idx], dtype=torch.float32),
            "ticker": torch.tensor(self.ticker_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

class LSTMModel(nn.Module):
    def __init__(self, n_tickers, use_market=False, n_market=0):
        super().__init__()
        self.use_market = use_market

        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, batch_first=True)
        self.ticker_emb = nn.Embedding(n_tickers, TICKER_EMB)

        if use_market:
            self.market_net = nn.Sequential(
                nn.Linear(n_market, MARKET_HIDDEN),
                nn.ReLU(),
                nn.Linear(MARKET_HIDDEN, MARKET_HIDDEN),
                nn.ReLU(),
            )
            in_dim = HIDDEN + TICKER_EMB + MARKET_HIDDEN
        else:
            in_dim = HIDDEN + TICKER_EMB

        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x, m, ticker):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        e = self.ticker_emb(ticker)

        if self.use_market:
            mh = self.market_net(m)
            z = torch.cat([h, e, mh], dim=1)
        else:
            z = torch.cat([h, e], dim=1)

        return self.head(z)

def make_loaders(df, market_cols, ticker_to_id, use_market):
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    x_mean = train[X_COLS].values.astype(np.float32).mean(axis=0, keepdims=True)
    x_std = train[X_COLS].values.astype(np.float32).std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    if use_market:
        m_mean = train[market_cols].values.astype(np.float32).mean(axis=0, keepdims=True)
        m_std = train[market_cols].values.astype(np.float32).std(axis=0, keepdims=True)
        m_std[m_std == 0] = 1.0
    else:
        m_mean = None
        m_std = None

    ds_train = PanelDataset(train, market_cols, ticker_to_id, use_market, x_mean, x_std, m_mean, m_std)
    ds_val = PanelDataset(val, market_cols, ticker_to_id, use_market, x_mean, x_std, m_mean, m_std)
    ds_test = PanelDataset(test, market_cols, ticker_to_id, use_market, x_mean, x_std, m_mean, m_std)

    return (
        DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False),
        train, val, test,
    )

def train_one_model(name, df, market_cols, ticker_to_id, use_market):
    print(f"\n==================== {name} ====================")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = make_loaders(
        df, market_cols, ticker_to_id, use_market
    )

    model = LSTMModel(
        n_tickers=len(ticker_to_id),
        use_market=use_market,
        n_market=len(market_cols)
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = MODEL_DIR / f"{name}.pt"
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            m = batch["m"].to(DEVICE)
            ticker = batch["ticker"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            pred = model(x, m, ticker)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())

        val_loss = evaluate_loss(model, val_loader, loss_fn)

        print(f"epoch={epoch:03d} train_mse={np.mean(train_losses):.8f} val_mse={val_loss:.8f}")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    val_pred = predict(model, val_loader)
    test_pred = predict(model, test_loader)

    val_metrics, val_detail = compute_metrics(val_df, val_pred)
    test_metrics, test_detail = compute_metrics(test_df, test_pred)

    val_detail.to_csv(OUT_DIR / f"{name}_val_predictions.csv", index=False)
    test_detail.to_csv(OUT_DIR / f"{name}_test_predictions.csv", index=False)

    result = {
        "name": name,
        "use_market": use_market,
        "best_val_mse": best_val,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(OUT_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nVAL metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nTEST metrics:")
    print(json.dumps(test_metrics, indent=2))

    return result

def evaluate_loss(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            m = batch["m"].to(DEVICE)
            ticker = batch["ticker"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            pred = model(x, m, ticker)
            losses.append(loss_fn(pred, y).item())
    return float(np.mean(losses))

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            m = batch["m"].to(DEVICE)
            ticker = batch["ticker"].to(DEVICE)
            pred = model(x, m, ticker)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds).reshape(-1)

def compute_metrics(df_part, pred):
    d = df_part.copy().reset_index(drop=True)
    d["pred"] = pred
    d["pnl_bp"] = 10000.0 * np.sign(d["pred"].values) * d["y"].values

    rmse = float(np.sqrt(np.mean((d["pred"].values - d["y"].values) ** 2)))
    mae = float(np.mean(np.abs(d["pred"].values - d["y"].values)))
    direction_acc = float(np.mean(np.sign(d["pred"].values) == np.sign(d["y"].values)))
    mean_pnl_bp = float(d["pnl_bp"].mean())

    # 每只股票每天有两个 time units，先 ticker-date 聚合，再算 Sharpe
    daily = d.groupby(["ticker", "date"], as_index=False)["pnl_bp"].sum()
    daily_mean = daily["pnl_bp"].mean()
    daily_std = daily["pnl_bp"].std(ddof=0)
    sharpe = float(np.sqrt(252) * daily_mean / daily_std) if daily_std > 0 else 0.0

    per_ticker = []
    for ticker, g in d.groupby("ticker"):
        gd = g.groupby("date", as_index=False)["pnl_bp"].sum()
        m = gd["pnl_bp"].mean()
        s = gd["pnl_bp"].std(ddof=0)
        sr = float(np.sqrt(252) * m / s) if s > 0 else 0.0
        per_ticker.append({
            "ticker": ticker,
            "rmse": float(np.sqrt(np.mean((g["pred"].values - g["y"].values) ** 2))),
            "mae": float(np.mean(np.abs(g["pred"].values - g["y"].values))),
            "direction_acc": float(np.mean(np.sign(g["pred"].values) == np.sign(g["y"].values))),
            "mean_pnl_bp": float(g["pnl_bp"].mean()),
            "daily_sharpe": sr,
        })

    per_ticker_df = pd.DataFrame(per_ticker)
    per_ticker_df.to_csv(OUT_DIR / f"per_ticker_temp.csv", index=False)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "direction_acc": direction_acc,
        "mean_pnl_bp": mean_pnl_bp,
        "daily_sharpe": sharpe,
        "n_rows": int(len(d)),
        "n_ticker_days": int(len(daily)),
    }
    return metrics, d

def main():
    print("DEVICE:", DEVICE)
    df = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    tickers = sorted(df["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}

    market_cols = [
        c for c in df.columns
        if c not in ID_COLS and c not in X_COLS
    ]

    print("panel shape:", df.shape)
    print("tickers:", tickers)
    print("market feature dim:", len(market_cols))
    print("first 10 market cols:", market_cols[:10])
    print("split counts:")
    print(df["split"].value_counts())

    baseline = train_one_model(
        name="baseline_lstm_xonly",
        df=df,
        market_cols=market_cols,
        ticker_to_id=ticker_to_id,
        use_market=False,
    )

    ram = train_one_model(
        name="ram_lstm_x_market",
        df=df,
        market_cols=market_cols,
        ticker_to_id=ticker_to_id,
        use_market=True,
    )

    summary = pd.DataFrame([
        {
            "model": baseline["name"],
            **{f"val_{k}": v for k, v in baseline["val"].items()},
            **{f"test_{k}": v for k, v in baseline["test"].items()},
        },
        {
            "model": ram["name"],
            **{f"val_{k}": v for k, v in ram["val"].items()},
            **{f"test_{k}": v for k, v in ram["test"].items()},
        }
    ])

    summary.to_csv(OUT_DIR / "lstm_vs_ram_lstm_summary.csv", index=False)
    print("\n==================== SUMMARY ====================")
    print(summary)

if __name__ == "__main__":
    main()
