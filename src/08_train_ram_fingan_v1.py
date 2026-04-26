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
REGIME_FILE = ROOT / "outputs" / "results" / "regime_baselines" / "regime_factors_by_date.csv"

OUT_DIR = ROOT / "outputs" / "results" / "ram_fingan_v1"
MODEL_DIR = ROOT / "outputs" / "models" / "ram_fingan_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 50
LR_G = 1e-4
LR_D = 1e-4
PATIENCE = 8

X_DIM = 10
Z_DIM = 8
HIDDEN = 32
REGIME_HIDDEN = 16
TICKER_EMB = 8

PNL_ALPHA = 1.0
MSE_BETA = 1.0
TANH_COEFF = 100.0

X_COLS = [f"x_{i}" for i in range(1, 11)]
FACTOR_COLS = [f"regime_factor_{i}" for i in range(1, 6)] + ["regime_0", "regime_1", "regime_2"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RAMGANDataset(Dataset):
    def __init__(self, df, ticker_to_id, x_mean, x_std, r_mean, r_std):
        self.df = df.reset_index(drop=True)

        x = self.df[X_COLS].values.astype(np.float32)
        self.x = ((x - x_mean) / x_std)[:, :, None]

        r = self.df[FACTOR_COLS].values.astype(np.float32)
        self.r = (r - r_mean) / r_std

        self.y = self.df["y"].values.astype(np.float32)[:, None]
        self.ticker_id = self.df["ticker"].map(ticker_to_id).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "r": torch.tensor(self.r[idx], dtype=torch.float32),
            "ticker": torch.tensor(self.ticker_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class RAMGenerator(nn.Module):
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

        self.net = nn.Sequential(
            nn.Linear(HIDDEN + REGIME_HIDDEN + TICKER_EMB + Z_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, r, ticker, z):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rh = self.regime_net(r)
        e = self.ticker_emb(ticker)
        inp = torch.cat([h, rh, e, z], dim=1)
        return self.net(inp)


class RAMDiscriminator(nn.Module):
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

        self.net = nn.Sequential(
            nn.Linear(HIDDEN + REGIME_HIDDEN + TICKER_EMB + 1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, r, ticker, y):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rh = self.regime_net(r)
        e = self.ticker_emb(ticker)
        inp = torch.cat([h, rh, e, y], dim=1)
        return self.net(inp)


def make_data():
    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    regimes = pd.read_csv(REGIME_FILE, parse_dates=["date"])

    keep = ["date"] + FACTOR_COLS
    regimes = regimes[keep].copy()

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

    loaders = {}
    dfs = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        ds = RAMGANDataset(part, ticker_to_id, x_mean, x_std, r_mean, r_std)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(name == "train"))
        dfs[name] = part.reset_index(drop=True)

    return df, loaders, dfs, ticker_to_id


def generator_loss(fake_score, fake_y, real_y):
    bce = nn.functional.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
    pnl = torch.mean(torch.tanh(TANH_COEFF * fake_y) * real_y)
    mse = torch.mean((fake_y - real_y) ** 2)
    return bce - PNL_ALPHA * pnl + MSE_BETA * mse, {
        "g_bce": float(bce.detach().cpu()),
        "g_pnl": float(pnl.detach().cpu()),
        "g_mse": float(mse.detach().cpu()),
    }


def train():
    df, loaders, dfs, ticker_to_id = make_data()

    print("DEVICE:", DEVICE)
    print("panel shape:", df.shape)
    print("tickers:", sorted(ticker_to_id.keys()))
    print("factor cols:", FACTOR_COLS)
    print("split counts:")
    print(df["split"].value_counts())

    G = RAMGenerator(len(ticker_to_id), len(FACTOR_COLS)).to(DEVICE)
    D = RAMDiscriminator(len(ticker_to_id), len(FACTOR_COLS)).to(DEVICE)

    opt_g = torch.optim.RMSprop(G.parameters(), lr=LR_G)
    opt_d = torch.optim.RMSprop(D.parameters(), lr=LR_D)

    best_val_sharpe = -1e9
    best_path_g = MODEL_DIR / "ram_fingan_v1_G.pt"
    best_path_d = MODEL_DIR / "ram_fingan_v1_D.pt"
    bad = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        G.train()
        D.train()

        d_losses, g_losses, pnls, mses = [], [], [], []

        for batch in loaders["train"]:
            x = batch["x"].to(DEVICE)
            r = batch["r"].to(DEVICE)
            ticker = batch["ticker"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            bs = x.size(0)

            z = torch.randn(bs, Z_DIM, device=DEVICE)
            fake_y = G(x, r, ticker, z).detach()

            real_score = D(x, r, ticker, y)
            fake_score = D(x, r, ticker, fake_y)

            d_loss_real = nn.functional.binary_cross_entropy(real_score, torch.ones_like(real_score))
            d_loss_fake = nn.functional.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            opt_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_d.step()

            z = torch.randn(bs, Z_DIM, device=DEVICE)
            fake_y = G(x, r, ticker, z)
            fake_score = D(x, r, ticker, fake_y)

            g_loss, parts = generator_loss(fake_score, fake_y, y)

            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()

            d_losses.append(float(d_loss.detach().cpu()))
            g_losses.append(float(g_loss.detach().cpu()))
            pnls.append(parts["g_pnl"])
            mses.append(parts["g_mse"])

        val_metrics, _ = evaluate(G, loaders["val"], dfs["val"], n_samples=20)
        row = {
            "epoch": epoch,
            "d_loss": float(np.mean(d_losses)),
            "g_loss": float(np.mean(g_losses)),
            "g_pnl": float(np.mean(pnls)),
            "g_mse": float(np.mean(mses)),
            "val_sharpe": val_metrics["daily_sharpe"],
            "val_direction_acc": val_metrics["direction_acc"],
            "val_mean_pnl_bp": val_metrics["mean_pnl_bp"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"d_loss={row['d_loss']:.6f} "
            f"g_loss={row['g_loss']:.6f} "
            f"g_pnl={row['g_pnl']:.8f} "
            f"val_sr={row['val_sharpe']:.4f} "
            f"val_dir={row['val_direction_acc']:.4f} "
            f"val_pnl={row['val_mean_pnl_bp']:.4f}"
        )

        if row["val_sharpe"] > best_val_sharpe:
            best_val_sharpe = row["val_sharpe"]
            bad = 0
            torch.save(G.state_dict(), best_path_g)
            torch.save(D.state_dict(), best_path_d)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    G.load_state_dict(torch.load(best_path_g, map_location=DEVICE))

    val_metrics, val_pred = evaluate(G, loaders["val"], dfs["val"], n_samples=100)
    test_metrics, test_pred = evaluate(G, loaders["test"], dfs["test"], n_samples=100)

    val_pred.to_csv(OUT_DIR / "ram_fingan_v1_val_predictions.csv", index=False)
    test_pred.to_csv(OUT_DIR / "ram_fingan_v1_test_predictions.csv", index=False)

    result = {
        "model": "ram_fingan_v1",
        "loss": "BCE - PnL + MSE",
        "z_dim": Z_DIM,
        "factor_cols": FACTOR_COLS,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(OUT_DIR / "ram_fingan_v1_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    summary = pd.DataFrame([{
        "model": "ram_fingan_v1",
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }])
    summary.to_csv(OUT_DIR / "ram_fingan_v1_summary.csv", index=False)

    print("\n===== VAL =====")
    print(json.dumps(val_metrics, indent=2))
    print("\n===== TEST =====")
    print(json.dumps(test_metrics, indent=2))
    print("\n===== SUMMARY =====")
    print(summary)


def evaluate(G, loader, df_part, n_samples=50):
    G.eval()
    preds_mean = []
    probs_up = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            r = batch["r"].to(DEVICE)
            ticker = batch["ticker"].to(DEVICE)
            bs = x.size(0)

            samples = []
            for _ in range(n_samples):
                z = torch.randn(bs, Z_DIM, device=DEVICE)
                fake_y = G(x, r, ticker, z)
                samples.append(fake_y.cpu().numpy())

            arr = np.stack(samples, axis=0).reshape(n_samples, bs)
            preds_mean.append(arr.mean(axis=0))
            probs_up.append((arr >= 0).mean(axis=0))

    pred = np.concatenate(preds_mean)
    pu = np.concatenate(probs_up)

    d = df_part.copy().reset_index(drop=True)
    d["pred_mean"] = pred
    d["p_up"] = pu
    d["position"] = 2.0 * d["p_up"] - 1.0
    d["pnl_bp"] = 10000.0 * d["position"].values * d["y"].values

    rmse = float(np.sqrt(np.mean((d["pred_mean"].values - d["y"].values) ** 2)))
    mae = float(np.mean(np.abs(d["pred_mean"].values - d["y"].values)))
    direction_acc = float(np.mean(np.sign(d["pred_mean"].values) == np.sign(d["y"].values)))
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
        "n_rows": int(len(d)),
        "n_ticker_days": int(len(daily)),
    }
    return metrics, d


if __name__ == "__main__":
    train()
