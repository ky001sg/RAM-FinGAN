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

OUT_DIR = ROOT / "outputs" / "results" / "ram_fingan_v2_pretrain"
MODEL_DIR = ROOT / "outputs" / "models" / "ram_fingan_v2_pretrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
PRETRAIN_EPOCHS = 15
ADV_EPOCHS = 40
PATIENCE = 8

LR_G_PRE = 1e-3
LR_G_ADV = 2e-4
LR_D = 2e-4

X_DIM = 10
Z_DIM = 8
HIDDEN = 32
REGIME_HIDDEN = 16
TICKER_EMB = 8

ADV_WEIGHT = 0.20
MSE_WEIGHT = 1.00
DIR_WEIGHT = 0.10
PNL_WEIGHT = 0.01
TANH_COEFF = 100.0

X_COLS = [f"x_{i}" for i in range(1, 11)]
FACTOR_COLS = [f"regime_factor_{i}" for i in range(1, 6)] + ["regime_0", "regime_1", "regime_2"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RAMGANDataset(Dataset):
    def __init__(self, df, ticker_to_id, x_mean, x_std, r_mean, r_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)

        x = self.df[X_COLS].values.astype(np.float32)
        self.x = ((x - x_mean) / x_std)[:, :, None]

        r = self.df[FACTOR_COLS].values.astype(np.float32)
        self.r = (r - r_mean) / r_std

        y_raw = self.df["y"].values.astype(np.float32)[:, None]
        self.y_raw = y_raw
        self.y_scaled = (y_raw - y_mean) / y_std

        self.ticker_id = self.df["ticker"].map(ticker_to_id).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "r": torch.tensor(self.r[idx], dtype=torch.float32),
            "ticker": torch.tensor(self.ticker_id[idx], dtype=torch.long),
            "y_scaled": torch.tensor(self.y_scaled[idx], dtype=torch.float32),
            "y_raw": torch.tensor(self.y_raw[idx], dtype=torch.float32),
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
        return self.net(torch.cat([h, rh, e, z], dim=1))


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

    def forward(self, x, r, ticker, y_scaled):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rh = self.regime_net(r)
        e = self.ticker_emb(ticker)
        return self.net(torch.cat([h, rh, e, y_scaled], dim=1))


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
        ds = RAMGANDataset(part, ticker_to_id, x_mean, x_std, r_mean, r_std, y_mean, y_std)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(name == "train"))
        dfs[name] = part.reset_index(drop=True)

    stats = {"y_mean": float(y_mean), "y_std": float(y_std)}
    return df, loaders, dfs, ticker_to_id, stats


def scaled_to_raw(y_scaled, stats):
    return y_scaled * stats["y_std"] + stats["y_mean"]


def supervised_loss(fake_scaled, y_scaled, fake_raw, y_raw):
    mse = torch.mean((fake_scaled - y_scaled) ** 2)
    target_dir = (y_raw > 0).float()
    dir_loss = nn.functional.binary_cross_entropy_with_logits(20.0 * fake_raw, target_dir)
    pnl_bp = 10000.0 * torch.mean(torch.tanh(TANH_COEFF * fake_raw) * y_raw)
    loss = MSE_WEIGHT * mse + DIR_WEIGHT * dir_loss - PNL_WEIGHT * pnl_bp
    return loss, mse, dir_loss, pnl_bp


def evaluate(G, loader, df_part, stats, n_samples=100):
    G.eval()
    pred_mean, p_up = [], []

    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            r = b["r"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            bs = x.size(0)

            samples = []
            for _ in range(n_samples):
                z = torch.randn(bs, Z_DIM, device=DEVICE)
                fake_scaled = G(x, r, ticker, z)
                fake_raw = scaled_to_raw(fake_scaled, stats)
                samples.append(fake_raw.cpu().numpy())

            arr = np.stack(samples, axis=0).reshape(n_samples, bs)
            pred_mean.append(arr.mean(axis=0))
            p_up.append((arr >= 0).mean(axis=0))

    pred = np.concatenate(pred_mean)
    pu = np.concatenate(p_up)

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


def train():
    df, loaders, dfs, ticker_to_id, stats = load_data()

    print("DEVICE:", DEVICE)
    print("panel shape:", df.shape)
    print("tickers:", sorted(ticker_to_id.keys()))
    print("y stats:", stats)
    print("split counts:")
    print(df["split"].value_counts())

    G = RAMGenerator(len(ticker_to_id), len(FACTOR_COLS)).to(DEVICE)
    D = RAMDiscriminator(len(ticker_to_id), len(FACTOR_COLS)).to(DEVICE)

    opt_g_pre = torch.optim.Adam(G.parameters(), lr=LR_G_PRE)

    print("\n===== PRETRAIN GENERATOR =====")
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        G.train()
        losses = []
        for b in loaders["train"]:
            x = b["x"].to(DEVICE)
            r = b["r"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            y_scaled = b["y_scaled"].to(DEVICE)
            y_raw = b["y_raw"].to(DEVICE)

            z = torch.randn(x.size(0), Z_DIM, device=DEVICE)
            fake_scaled = G(x, r, ticker, z)
            fake_raw = scaled_to_raw(fake_scaled, stats)

            loss, mse, dir_loss, pnl_bp = supervised_loss(fake_scaled, y_scaled, fake_raw, y_raw)

            opt_g_pre.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g_pre.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(G, loaders["val"], dfs["val"], stats, n_samples=30)
        print(f"pre_epoch={epoch:03d} loss={np.mean(losses):.6f} val_sr={val_metrics['daily_sharpe']:.4f} val_dir={val_metrics['direction_acc']:.4f}")

    opt_g = torch.optim.RMSprop(G.parameters(), lr=LR_G_ADV)
    opt_d = torch.optim.RMSprop(D.parameters(), lr=LR_D)

    best_val = -1e9
    bad = 0
    best_g = MODEL_DIR / "ram_fingan_v2_G.pt"
    best_d = MODEL_DIR / "ram_fingan_v2_D.pt"
    history = []

    print("\n===== ADVERSARIAL FINETUNE =====")
    for epoch in range(1, ADV_EPOCHS + 1):
        G.train()
        D.train()
        d_losses, g_losses = [], []

        for b in loaders["train"]:
            x = b["x"].to(DEVICE)
            r = b["r"].to(DEVICE)
            ticker = b["ticker"].to(DEVICE)
            y_scaled = b["y_scaled"].to(DEVICE)
            y_raw = b["y_raw"].to(DEVICE)
            bs = x.size(0)

            z = torch.randn(bs, Z_DIM, device=DEVICE)
            fake_scaled = G(x, r, ticker, z).detach()

            real_score = D(x, r, ticker, y_scaled)
            fake_score = D(x, r, ticker, fake_scaled)

            d_loss = 0.5 * (
                nn.functional.binary_cross_entropy(real_score, torch.ones_like(real_score)) +
                nn.functional.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
            )

            opt_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_d.step()

            z = torch.randn(bs, Z_DIM, device=DEVICE)
            fake_scaled = G(x, r, ticker, z)
            fake_raw = scaled_to_raw(fake_scaled, stats)
            fake_score = D(x, r, ticker, fake_scaled)

            adv = nn.functional.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
            sup, mse, dir_loss, pnl_bp = supervised_loss(fake_scaled, y_scaled, fake_raw, y_raw)
            g_loss = ADV_WEIGHT * adv + sup

            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()

            d_losses.append(float(d_loss.detach().cpu()))
            g_losses.append(float(g_loss.detach().cpu()))

        val_metrics, _ = evaluate(G, loaders["val"], dfs["val"], stats, n_samples=30)
        row = {
            "epoch": epoch,
            "d_loss": float(np.mean(d_losses)),
            "g_loss": float(np.mean(g_losses)),
            "val_sharpe": val_metrics["daily_sharpe"],
            "val_direction_acc": val_metrics["direction_acc"],
            "val_mean_pnl_bp": val_metrics["mean_pnl_bp"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(row)

        print(
            f"adv_epoch={epoch:03d} d_loss={row['d_loss']:.6f} g_loss={row['g_loss']:.6f} "
            f"val_sr={row['val_sharpe']:.4f} val_dir={row['val_direction_acc']:.4f} val_pnl={row['val_mean_pnl_bp']:.4f}"
        )

        if row["val_sharpe"] > best_val:
            best_val = row["val_sharpe"]
            bad = 0
            torch.save(G.state_dict(), best_g)
            torch.save(D.state_dict(), best_d)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at adv epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    G.load_state_dict(torch.load(best_g, map_location=DEVICE))

    val_metrics, val_pred = evaluate(G, loaders["val"], dfs["val"], stats, n_samples=100)
    test_metrics, test_pred = evaluate(G, loaders["test"], dfs["test"], stats, n_samples=100)

    val_pred.to_csv(OUT_DIR / "ram_fingan_v2_val_predictions.csv", index=False)
    test_pred.to_csv(OUT_DIR / "ram_fingan_v2_test_predictions.csv", index=False)

    result = {
        "model": "ram_fingan_v2_pretrain",
        "description": "RAM-FinGAN with supervised generator pretraining and adversarial finetuning",
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(OUT_DIR / "ram_fingan_v2_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    summary = pd.DataFrame([{
        "model": "ram_fingan_v2_pretrain",
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }])
    summary.to_csv(OUT_DIR / "ram_fingan_v2_summary.csv", index=False)

    print("\n===== VAL =====")
    print(json.dumps(val_metrics, indent=2))
    print("\n===== TEST =====")
    print(json.dumps(test_metrics, indent=2))
    print("\n===== SUMMARY =====")
    print(summary)


if __name__ == "__main__":
    train()
