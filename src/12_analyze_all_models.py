from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/kwang/RAM-FinGAN")
OUT = ROOT / "outputs" / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

REGIME_FILE = ROOT / "outputs" / "results" / "regime_baselines" / "regime_factors_by_date.csv"

MODELS = {
    "baseline_lstm_xonly": ROOT / "outputs/results/baselines/baseline_lstm_xonly_test_predictions.csv",
    "naive_ram_lstm": ROOT / "outputs/results/baselines/ram_lstm_x_market_test_predictions.csv",
    "regime_factor_lstm": ROOT / "outputs/results/regime_baselines/regime_factor_lstm_test_predictions.csv",
    "ram_fingan_v1": ROOT / "outputs/results/ram_fingan_v1/ram_fingan_v1_test_predictions.csv",
    "ram_fingan_v2": ROOT / "outputs/results/ram_fingan_v2_pretrain/ram_fingan_v2_test_predictions.csv",
    "ram_fingan_v3": ROOT / "outputs/results/ram_fingan_v3_econ/ram_fingan_v3_test_predictions.csv",
    "regime_econ_lstm_v4": ROOT / "outputs/results/regime_econ_lstm_v4/regime_econ_lstm_v4_test_predictions.csv",
}

COVID_START = pd.Timestamp("2020-02-20")
COVID_END = pd.Timestamp("2020-06-30")

PERIODS = {
    "pre_covid_test": (pd.Timestamp("1900-01-01"), COVID_START - pd.Timedelta(days=1)),
    "covid_shock": (COVID_START, COVID_END),
    "post_covid": (COVID_END + pd.Timedelta(days=1), pd.Timestamp("2100-01-01")),
}


def load_regimes():
    reg = pd.read_csv(REGIME_FILE, parse_dates=["date"])
    reg.columns = [c.strip() for c in reg.columns]

    if "regime_cluster" not in reg.columns:
        dummy_cols = [c for c in reg.columns if c in ["regime_0", "regime_1", "regime_2"]]
        if dummy_cols:
            reg["regime_cluster"] = reg[dummy_cols].values.argmax(axis=1)
        else:
            print("[WARN] No regime_cluster or regime dummies found. Set all regimes to -1.")
            reg["regime_cluster"] = -1

    reg = reg[["date", "regime_cluster"]].drop_duplicates("date")
    reg["regime_cluster"] = reg["regime_cluster"].fillna(-1).astype(int)
    return reg


def load_prediction(path):
    d = pd.read_csv(path, parse_dates=["date"])
    d.columns = [c.strip() for c in d.columns]

    if "pred" not in d.columns:
        if "pred_mean" in d.columns:
            d["pred"] = d["pred_mean"]
        elif "position" in d.columns:
            d["pred"] = d["position"]
        else:
            raise ValueError(f"{path} lacks pred / pred_mean / position")

    if "position" not in d.columns:
        d["position"] = np.sign(d["pred"])

    if "pnl_bp" not in d.columns:
        d["pnl_bp"] = 10000.0 * d["position"] * d["y"]

    return d


def metrics(d):
    d = d.copy()
    rmse = np.sqrt(np.mean((d["pred"].values - d["y"].values) ** 2))
    mae = np.mean(np.abs(d["pred"].values - d["y"].values))
    direction = np.mean(np.sign(d["pred"].values) == np.sign(d["y"].values))
    mean_pnl = d["pnl_bp"].mean()

    daily = d.groupby(["ticker", "date"], as_index=False)["pnl_bp"].sum()
    mu = daily["pnl_bp"].mean()
    sig = daily["pnl_bp"].std(ddof=0)
    sharpe = np.sqrt(252) * mu / sig if sig > 0 else 0.0

    return {
        "n_rows": int(len(d)),
        "n_ticker_days": int(len(daily)),
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_acc": float(direction),
        "mean_pnl_bp": float(mean_pnl),
        "daily_sharpe": float(sharpe),
    }


def main():
    regimes = load_regimes()
    print("[OK] regime file loaded")
    print(regimes.head())
    print("regime counts:")
    print(regimes["regime_cluster"].value_counts().sort_index())

    all_overall = []
    all_ticker = []
    all_regime = []
    all_period = []

    for model, path in MODELS.items():
        if not path.exists():
            print("[WARN] missing prediction:", model, path)
            continue

        d = load_prediction(path)

        # 如果 prediction 文件里已经有 regime_cluster，先删掉，统一用 regime file 重新 merge
        for c in ["regime_cluster", "regime_cluster_x", "regime_cluster_y"]:
            if c in d.columns:
                d = d.drop(columns=[c])

        d = d.merge(regimes, on="date", how="left")
        d["regime_cluster"] = d["regime_cluster"].fillna(-1).astype(int)

        row = metrics(d)
        row["model"] = model
        all_overall.append(row)

        for ticker, g in d.groupby("ticker"):
            row = metrics(g)
            row["model"] = model
            row["ticker"] = ticker
            all_ticker.append(row)

        for reg, g in d.groupby("regime_cluster"):
            row = metrics(g)
            row["model"] = model
            row["regime_cluster"] = int(reg)
            all_regime.append(row)

        for period, (a, b) in PERIODS.items():
            g = d[(d["date"] >= a) & (d["date"] <= b)].copy()
            if len(g) == 0:
                continue
            row = metrics(g)
            row["model"] = model
            row["period"] = period
            row["start"] = str(g["date"].min().date())
            row["end"] = str(g["date"].max().date())
            all_period.append(row)

    overall = pd.DataFrame(all_overall).sort_values("daily_sharpe", ascending=False)
    by_ticker = pd.DataFrame(all_ticker)
    by_regime = pd.DataFrame(all_regime)
    by_period = pd.DataFrame(all_period)

    overall.to_csv(OUT / "01_overall_test_metrics.csv", index=False)
    by_ticker.to_csv(OUT / "02_by_ticker_test_metrics.csv", index=False)
    by_regime.to_csv(OUT / "03_by_regime_test_metrics.csv", index=False)
    by_period.to_csv(OUT / "04_by_period_test_metrics.csv", index=False)

    best_by_ticker = (
        by_ticker.sort_values(["ticker", "daily_sharpe"], ascending=[True, False])
        .groupby("ticker")
        .head(1)
        .reset_index(drop=True)
    )
    best_by_ticker.to_csv(OUT / "05_best_model_by_ticker.csv", index=False)

    regime_pivot = by_regime.pivot_table(index="regime_cluster", columns="model", values="daily_sharpe", aggfunc="first")
    regime_pivot.to_csv(OUT / "06_regime_sharpe_pivot.csv")

    period_pivot = by_period.pivot_table(index="period", columns="model", values="daily_sharpe", aggfunc="first")
    period_pivot.to_csv(OUT / "07_period_sharpe_pivot.csv")

    report = []
    report.append("# RAM-FinGAN Model Analysis Report\n")
    report.append("## Overall test ranking\n")
    report.append(overall[["model", "direction_acc", "mean_pnl_bp", "daily_sharpe", "rmse", "mae"]].to_markdown(index=False))
    report.append("\n## Sharpe by regime\n")
    report.append(regime_pivot.to_markdown())
    report.append("\n## Sharpe by period\n")
    report.append(period_pivot.to_markdown())
    report.append("\n## Best model by ticker\n")
    report.append(best_by_ticker[["ticker", "model", "daily_sharpe", "direction_acc", "mean_pnl_bp"]].to_markdown(index=False))

    report.append("""
## Main interpretation

Current evidence should be read as follows:

1. Directly concatenating raw market features is not robust.
2. Compressing market information into regime factors gives the strongest out-of-sample result so far.
3. GAN-style adversarial training is unstable in this setting.
4. Economic-loss training can overfit validation-period trading objectives.
5. The most defensible current paper direction is regime-factor representation for robust stock--ETF excess-return forecasting, with GAN variants reported as ablation/negative evidence rather than the main claim.
""")

    report_text = "\n".join(report)
    (OUT / "analysis_report.md").write_text(report_text)

    print("\n==================== OVERALL ====================")
    print(overall[["model", "direction_acc", "mean_pnl_bp", "daily_sharpe", "rmse", "mae"]])

    print("\n==================== BY REGIME SHARPE ====================")
    print(regime_pivot)

    print("\n==================== BY PERIOD SHARPE ====================")
    print(period_pivot)

    print("\nGenerated:")
    for p in sorted(OUT.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
