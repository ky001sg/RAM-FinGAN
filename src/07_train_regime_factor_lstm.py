from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/kwang/RAM-FinGAN")
PANEL_FILE = ROOT / "data_clean" / "ram_panel" / "ALL_STOCKS_panel.csv"

OUT_DIR = ROOT / "outputs" / "results" / "regime_baselines"
MODEL_DIR = ROOT / "outputs" / "models" / "regime_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
HIDDEN = 32
TICKER_EMB = 8
REGIME_HIDDEN = 16
PATIENCE = 6

N_FACTORS = 5
N_REGIMES = 3

X_COLS = [f"x_{i}" for i in range(1, 11)]
ID_COLS = {"date", "ticker", "sector_etf", "split", "y"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class PanelDataset(Dataset):
    def __init__(self, df, feature_cols, ticker_to_id, x_mean, x_std, f_mean=None, f_std=None):
        self.df = df.reset_index(drop=True)
        x = self.df[X_COLS].values.astype(np.float32)
        self.x = ((x - x_mean) / x_std)[:, :, None]

        f = self.df[feature_cols].values.astype(np.float32)
        self.f = (f - f_mean) / f_std

        self.y = self.df["y"].values.astype(np.float32)[:, None]
        self.ticker_id = self.df["ticker"].map(ticker_to_id).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "f": torch.tensor(self.f[idx], dtype=torch.float32),
            "ticker": torch.tensor(self.ticker_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class RegimeFactorLSTM(nn.Module):
    def __init__(self, n_tickers, n_features):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, batch_first=True)
        self.ticker_emb = nn.Embedding(n_tickers, TICKER_EMB)

        self.regime_net = nn.Sequential(
            nn.Linear(n_features, REGIME_HIDDEN),
            nn.ReLU(),
            nn.Linear(REGIME_HIDDEN, REGIME_HIDDEN),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(HIDDEN + TICKER_EMB + REGIME_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x, f, ticker):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        e = self.ticker_emb(ticker)
        r = self.regime_net(f)
        z = torch.cat([h, e, r], dim=1)
        return self.head(z)


def compute_pca_and_clusters(df, market_cols):
    train_dates = df[df["split"] == "train"][["date"]].drop_duplicates().sort_values("date")
    all_dates = df[["date"] + market_cols].drop_duplicates("date").sort_values("date").reset_index(drop=True)

    train_market = all_dates[all_dates["date"].isin(train_dates["date"])].copy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_market[market_cols].values)
    all_scaled = scaler.transform(all_dates[market_cols].values)

    pca = PCA(n_components=N_FACTORS, random_state=SEED)
    train_factors = pca.fit_transform(train_scaled)
    all_factors = pca.transform(all_scaled)

    kmeans = KMeans(n_clusters=N_REGIMES, random_state=SEED, n_init=20)
    train_cluster = kmeans.fit_predict(train_factors)
    all_cluster = kmeans.predict(all_factors)

    factor_cols = [f"regime_factor_{i+1}" for i in range(N_FACTORS)]
    factor_df = all_dates[["date"]].copy()

    for i, c in enumerate(factor_cols):
        factor_df[c] = all_factors[:, i]

    factor_df["regime_cluster"] = all_cluster

    cluster_dummies = pd.get_dummies(factor_df["regime_cluster"], prefix="regime")
    factor_df = pd.concat([factor_df, cluster_dummies], axis=1)

    explained = pca.explained_variance_ratio_

    info = {
        "n_factors": N_FACTORS,
        "n_regimes": N_REGIMES,
        "pca_explained_variance_ratio": explained.tolist(),
        "pca_cumulative_explained_variance": float(explained.sum()),
        "cluster_counts_all_dates": pd.Series(all_cluster).value_counts().sort_index().to_dict(),
        "cluster_counts_train_dates": pd.Series(train_cluster).value_counts().sort_index().to_dict(),
    }

    factor_df.to_csv(OUT_DIR / "regime_factors_by_date.csv", index=False)

    with open(OUT_DIR / "regime_factor_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n===== Regime factor info =====")
    print(json.dumps(info, indent=2))

    return factor_df


def make_loaders(df, feature_cols, ticker_to_id):
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    x_mean = train[X_COLS].values.astype(np.float32).mean(axis=0, keepdims=True)
    x_std = train[X_COLS].values.astype(np.float32).std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    f_mean = train[feature_cols].values.astype(np.float32).mean(axis=0, keepdims=True)
    f_std = train[feature_cols].values.astype(np.float32).std(axis=0, keepdims=True)
    f_std[f_std == 0] = 1.0

    return (
        DataLoader(PanelDataset(train, feature_cols, ticker_to_id, x_mean, x_std, f_mean, f_std),
                   batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(PanelDataset(val, feature_cols, ticker_to_id, x_mean, x_std, f_mean, f_std),
                   batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(PanelDataset(test, feature_cols, ticker_to_id, x_mean, x_std, f_mean, f_std),
                   batch_size=BATCH_SIZE, shuffle=False),
        train, val, test,
    )


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            f = b["f"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            y = b["y"].to(DEVICE)
            pred = model(x, f, ticker)
            losses.append(loss_fn(pred, y).item())
    return float(np.mean(losses))


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            f = b["f"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            pred = model(x, f, ticker)
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

    return {
        "rmse": rmse,
        "mae": mae,
        "direction_acc": direction_acc,
        "mean_pnl_bp": mean_pnl_bp,
        "daily_sharpe": sharpe,
        "n_rows": int(len(d)),
        "n_ticker_days": int(len(daily)),
    }, d, pd.DataFrame(per_ticker)


def train_model(df, feature_cols, ticker_to_id):
    train_loader, val_loader, test_loader, train_df, val_df, test_df = make_loaders(
        df, feature_cols, ticker_to_id
    )

    model = RegimeFactorLSTM(
        n_tickers=len(ticker_to_id),
        n_features=len(feature_cols),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = MODEL_DIR / "regime_factor_lstm.pt"
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for b in train_loader:
            x = b["x"].to(DEVICE)
            f = b["f"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            y = b["y"].to(DEVICE)

            pred = model(x, f, ticker)
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

    val_metrics, val_detail, val_per_ticker = compute_metrics(val_df, val_pred)
    test_metrics, test_detail, test_per_ticker = compute_metrics(test_df, test_pred)

    val_detail.to_csv(OUT_DIR / "regime_factor_lstm_val_predictions.csv", index=False)
    test_detail.to_csv(OUT_DIR / "regime_factor_lstm_test_predictions.csv", index=False)
    val_per_ticker.to_csv(OUT_DIR / "regime_factor_lstm_val_per_ticker.csv", index=False)
    test_per_ticker.to_csv(OUT_DIR / "regime_factor_lstm_test_per_ticker.csv", index=False)

    result = {
        "model": "regime_factor_lstm",
        "feature_cols": feature_cols,
        "best_val_mse": best_val,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(OUT_DIR / "regime_factor_lstm_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    summary = pd.DataFrame([{
        "model": "regime_factor_lstm",
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }])
    summary.to_csv(OUT_DIR / "regime_factor_lstm_summary.csv", index=False)

    print("\n===== VAL =====")
    print(json.dumps(val_metrics, indent=2))
    print("\n===== TEST =====")
    print(json.dumps(test_metrics, indent=2))
    print("\n===== SUMMARY =====")
    print(summary)

    return result


def main():
    print("DEVICE:", DEVICE)

    df = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    raw_market_cols = [c for c in df.columns if c not in ID_COLS and c not in X_COLS]

    factor_df = compute_pca_and_clusters(df, raw_market_cols)

    df2 = df.merge(factor_df, on="date", how="left")
    df2 = df2.dropna().reset_index(drop=True)

    factor_cols = [f"regime_factor_{i+1}" for i in range(N_FACTORS)]
    regime_dummy_cols = [c for c in df2.columns if c.startswith("regime_") and c not in factor_cols]
    feature_cols = factor_cols + regime_dummy_cols

    tickers = sorted(df2["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}

    print("panel shape:", df2.shape)
    print("tickers:", tickers)
    print("regime feature cols:", feature_cols)
    print("split counts:")
    print(df2["split"].value_counts())

    train_model(df2, feature_cols, ticker_to_id)


if __name__ == "__main__":
    main()
