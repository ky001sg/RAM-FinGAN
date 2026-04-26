from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/home/kwang/RAM-FinGAN")
PANEL_FILE = ROOT / "data_clean" / "ram_panel" / "ALL_STOCKS_panel.csv"
REGIME_FILE = ROOT / "outputs" / "results" / "regime_baselines" / "regime_factors_by_date.csv"

OUT_DIR = ROOT / "outputs" / "results" / "regime_econ_lstm_v4"
MODEL_DIR = ROOT / "outputs" / "models" / "regime_econ_lstm_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 8
LR = 1e-3

HIDDEN = 32
REGIME_HIDDEN = 16
TICKER_EMB = 8

W_MSE = 1.0
W_DIR = 0.10
W_PNL = 0.02
W_SR = 0.01
TANH_COEFF = 100.0
EPS = 1e-8

X_COLS = [f"x_{i}" for i in range(1, 11)]
FACTOR_COLS = [f"regime_factor_{i}" for i in range(1, 6)] + ["regime_0", "regime_1", "regime_2"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class PanelDataset(Dataset):
    def __init__(self, df, ticker_to_id, x_mean, x_std, r_mean, r_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)

        x = self.df[X_COLS].values.astype(np.float32)
        self.x = ((x - x_mean) / x_std)[:, :, None]

        r = self.df[FACTOR_COLS].values.astype(np.float32)
        self.r = (r - r_mean) / r_std

        y = self.df["y"].values.astype(np.float32)[:, None]
        self.y_raw = y
        self.y_scaled = (y - y_mean) / y_std

        self.ticker_id = self.df["ticker"].map(ticker_to_id).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "r": torch.tensor(self.r[idx], dtype=torch.float32),
            "ticker": torch.tensor(self.ticker_id[idx], dtype=torch.long),
            "y_raw": torch.tensor(self.y_raw[idx], dtype=torch.float32),
            "y_scaled": torch.tensor(self.y_scaled[idx], dtype=torch.float32),
        }


class RegimeEconLSTM(nn.Module):
    def __init__(self, n_tickers, regime_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, batch_first=True)
        self.regime_net = nn.Sequential(
            nn.Linear(regime_dim, REGIME_HIDDEN),
            nn.ReLU(),
            nn.Linear(REGIME_HIDDEN, REGIME_HIDDEN),
            nn.ReLU(),
        )
        self.ticker_emb = nn.Embedding(n_tickers, TICKER_EMB)

        self.head = nn.Sequential(
            nn.Linear(HIDDEN + REGIME_HIDDEN + TICKER_EMB, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, r, ticker):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rh = self.regime_net(r)
        e = self.ticker_emb(ticker)
        return self.head(torch.cat([h, rh, e], dim=1))


def load_data():
    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    regimes = pd.read_csv(REGIME_FILE, parse_dates=["date"])[["date"] + FACTOR_COLS]
    df = panel.merge(regimes, on="date", how="left")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    tickers = sorted(df["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    x_mean = train[X_COLS].values.astype(np.float32).mean(axis=0, keepdims=True)
    x_std = train[X_COLS].values.astype(np.float32).std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    r_mean = train[FACTOR_COLS].values.astype(np.float32).mean(axis=0, keepdims=True)
    r_std = train[FACTOR_COLS].values.astype(np.float32).std(axis=0, keepdims=True)
    r_std[r_std == 0] = 1.0

    y_mean = train["y"].values.astype(np.float32).mean()
    y_std = train["y"].values.astype(np.float32).std()
    if y_std == 0:
        y_std = 1.0

    loaders, dfs = {}, {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        ds = PanelDataset(part, ticker_to_id, x_mean, x_std, r_mean, r_std, y_mean, y_std)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(name == "train"))
        dfs[name] = part.reset_index(drop=True)

    stats = {"y_mean": float(y_mean), "y_std": float(y_std)}
    return df, loaders, dfs, ticker_to_id, stats


def scaled_to_raw(y_scaled, stats):
    return y_scaled * stats["y_std"] + stats["y_mean"]


def econ_loss(pred_scaled, pred_raw, y_scaled, y_raw):
    mse = torch.mean((pred_scaled - y_scaled) ** 2)

    target_dir = (y_raw > 0).float()
    dir_loss = nn.functional.binary_cross_entropy_with_logits(20.0 * pred_raw, target_dir)

    position = torch.tanh(TANH_COEFF * pred_raw)
    pnl_each = position * y_raw
    pnl_bp = 10000.0 * pnl_each.mean()

    sr = pnl_each.mean() / (pnl_each.std(unbiased=False) + EPS)

    loss = W_MSE * mse + W_DIR * dir_loss - W_PNL * pnl_bp - W_SR * sr

    return loss, {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "dir_loss": float(dir_loss.detach().cpu()),
        "pnl_bp": float(pnl_bp.detach().cpu()),
        "sr_batch": float(sr.detach().cpu()),
    }


def evaluate(model, loader, df_part, stats):
    model.eval()
    preds = []

    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            r = b["r"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            pred_scaled = model(x, r, ticker)
            pred_raw = scaled_to_raw(pred_scaled, stats)
            preds.append(pred_raw.cpu().numpy())

    pred = np.vstack(preds).reshape(-1)

    d = df_part.copy().reset_index(drop=True)
    d["pred"] = pred
    d["position"] = np.tanh(TANH_COEFF * d["pred"].values)
    d["pnl_bp"] = 10000.0 * d["position"].values * d["y"].values

    rmse = float(np.sqrt(np.mean((d["pred"].values - d["y"].values) ** 2)))
    mae = float(np.mean(np.abs(d["pred"].values - d["y"].values)))
    direction_acc = float(np.mean(np.sign(d["pred"].values) == np.sign(d["y"].values)))
    mean_pnl_bp = float(d["pnl_bp"].mean())

    daily = d.groupby(["ticker", "date"], as_index=False)["pnl_bp"].sum()
    daily_mean = daily["pnl_bp"].mean()
    daily_std = daily["pnl_bp"].std(ddof=0)
    sharpe = float(np.sqrt(252) * daily_mean / daily_std) if daily_std > 0 else 0.0

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "direction_acc": direction_acc,
        "mean_pnl_bp": mean_pnl_bp,
        "daily_sharpe": sharpe,
        "avg_abs_position": float(np.abs(d["position"]).mean()),
        "n_rows": int(len(d)),
        "n_ticker_days": int(len(daily)),
    }
    return metrics, d


def train():
    df, loaders, dfs, ticker_to_id, stats = load_data()

    print("DEVICE:", DEVICE)
    print("panel shape:", df.shape)
    print("tickers:", sorted(ticker_to_id.keys()))
    print("y stats:", stats)
    print("loss weights:", {
        "W_MSE": W_MSE,
        "W_DIR": W_DIR,
        "W_PNL": W_PNL,
        "W_SR": W_SR,
    })
    print("split counts:")
    print(df["split"].value_counts())

    model = RegimeEconLSTM(len(ticker_to_id), len(FACTOR_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1e9
    bad = 0
    best_path = MODEL_DIR / "regime_econ_lstm_v4.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        parts_all = []

        for b in loaders["train"]:
            x = b["x"].to(DEVICE)
            r = b["r"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            y_scaled = b["y_scaled"].to(DEVICE)
            y_raw = b["y_raw"].to(DEVICE)

            pred_scaled = model(x, r, ticker)
            pred_raw = scaled_to_raw(pred_scaled, stats)

            loss, parts = econ_loss(pred_scaled, pred_raw, y_scaled, y_raw)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            parts_all.append(parts)

        val_metrics, _ = evaluate(model, loaders["val"], dfs["val"], stats)
        avg = {k: float(np.mean([p[k] for p in parts_all])) for k in parts_all[0].keys()}

        row = {
            "epoch": epoch,
            **avg,
            "val_sharpe": val_metrics["daily_sharpe"],
            "val_direction_acc": val_metrics["direction_acc"],
            "val_mean_pnl_bp": val_metrics["mean_pnl_bp"],
            "val_rmse": val_metrics["rmse"],
            "val_avg_abs_position": val_metrics["avg_abs_position"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} loss={row['loss']:.6f} mse={row['mse']:.6f} "
            f"pnl_bp={row['pnl_bp']:.4f} sr_batch={row['sr_batch']:.4f} "
            f"val_sr={row['val_sharpe']:.4f} val_dir={row['val_direction_acc']:.4f} "
            f"val_pnl={row['val_mean_pnl_bp']:.4f} val_pos={row['val_avg_abs_position']:.4f}"
        )

        if row["val_sharpe"] > best_val:
            best_val = row["val_sharpe"]
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    val_metrics, val_pred = evaluate(model, loaders["val"], dfs["val"], stats)
    test_metrics, test_pred = evaluate(model, loaders["test"], dfs["test"], stats)

    val_pred.to_csv(OUT_DIR / "regime_econ_lstm_v4_val_predictions.csv", index=False)
    test_pred.to_csv(OUT_DIR / "regime_econ_lstm_v4_test_predictions.csv", index=False)

    result = {
        "model": "regime_econ_lstm_v4",
        "description": "regime-factor LSTM trained with Fin-GAN-style economic loss: MSE + direction - PnL - Sharpe",
        "factor_cols": FACTOR_COLS,
        "loss_weights": {
            "W_MSE": W_MSE,
            "W_DIR": W_DIR,
            "W_PNL": W_PNL,
            "W_SR": W_SR,
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(OUT_DIR / "regime_econ_lstm_v4_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    summary = pd.DataFrame([{
        "model": "regime_econ_lstm_v4",
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }])
    summary.to_csv(OUT_DIR / "regime_econ_lstm_v4_summary.csv", index=False)

    print("\n===== VAL =====")
    print(json.dumps(val_metrics, indent=2))
    print("\n===== TEST =====")
    print(json.dumps(test_metrics, indent=2))
    print("\n===== SUMMARY =====")
    print(summary)


if __name__ == "__main__":
    train()
