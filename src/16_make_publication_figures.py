from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/kwang/RAM-FinGAN")
FIG = ROOT / "outputs" / "plots" / "publication_figures"
FIG.mkdir(parents=True, exist_ok=True)

ANALYSIS = ROOT / "outputs" / "results" / "analysis"
ROBUST = ROOT / "outputs" / "results" / "robustness"
SMOOTH = ROOT / "outputs" / "results" / "position_smoothing"

PRED_FILES = {
    "LSTM": ROOT / "outputs/results/baselines/baseline_lstm_xonly_test_predictions.csv",
    "Naive RAM-LSTM": ROOT / "outputs/results/baselines/ram_lstm_x_market_test_predictions.csv",
    "Regime-Factor LSTM": ROOT / "outputs/results/regime_baselines/regime_factor_lstm_test_predictions.csv",
    "RAM-FinGAN v2": ROOT / "outputs/results/ram_fingan_v2_pretrain/ram_fingan_v2_test_predictions.csv",
    "RAM-FinGAN v3": ROOT / "outputs/results/ram_fingan_v3_econ/ram_fingan_v3_test_predictions.csv",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close()

def load_pred(path):
    d = pd.read_csv(path, parse_dates=["date"])
    if "pred" not in d.columns:
        if "pred_mean" in d.columns:
            d["pred"] = d["pred_mean"]
        elif "position" in d.columns:
            d["pred"] = d["position"]
    if "position" not in d.columns:
        d["position"] = np.sign(d["pred"])
    if "pnl_bp" not in d.columns:
        d["pnl_bp"] = 10000.0 * d["position"] * d["y"]
    return d.sort_values(["date", "ticker"])

def daily_pnl(d):
    return d.groupby("date", as_index=False)["pnl_bp"].sum().sort_values("date")

def figure_overall_bar():
    df = pd.read_csv(ANALYSIS / "01_overall_test_metrics.csv")
    df = df.sort_values("daily_sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.barh(df["model"], df["daily_sharpe"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Test Sharpe ratio")
    ax.set_ylabel("")
    savefig("fig1_overall_model_ranking")

def figure_cumulative_pnl():
    fig, ax = plt.subplots(figsize=(3.6, 2.5))

    for name, path in PRED_FILES.items():
        if not path.exists():
            continue
        d = load_pred(path)
        daily = daily_pnl(d)
        daily["cum_pnl"] = daily["pnl_bp"].cumsum()
        ax.plot(daily["date"], daily["cum_pnl"], linewidth=1.0, label=name)

    ax.axhline(0, linewidth=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (bp)")
    ax.legend(frameon=False, ncol=1)
    savefig("fig2_cumulative_pnl_test")

def figure_period_sharpe():
    df = pd.read_csv(ANALYSIS / "04_by_period_test_metrics.csv")
    keep = ["baseline_lstm_xonly", "naive_ram_lstm", "regime_factor_lstm", "ram_fingan_v2", "ram_fingan_v3"]
    df = df[df["model"].isin(keep)]

    pivot = df.pivot(index="period", columns="model", values="daily_sharpe")
    order = ["pre_covid_test", "covid_shock", "post_covid"]
    pivot = pivot.loc[[x for x in order if x in pivot.index]]

    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    x = np.arange(len(pivot.index))
    width = 0.15

    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - len(pivot.columns)/2)*width + width/2, pivot[col].values, width, label=col)

    ax.axhline(0, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=7)
    savefig("fig3_period_sharpe")

def figure_regime_sharpe():
    df = pd.read_csv(ANALYSIS / "03_by_regime_test_metrics.csv")
    keep = ["baseline_lstm_xonly", "naive_ram_lstm", "regime_factor_lstm", "ram_fingan_v2", "ram_fingan_v3"]
    df = df[df["model"].isin(keep)]

    pivot = df.pivot(index="regime_cluster", columns="model", values="daily_sharpe").sort_index()

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(len(pivot.index))
    width = 0.15

    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - len(pivot.columns)/2)*width + width/2, pivot[col].values, width, label=col)

    ax.axhline(0, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Regime {int(i)}" for i in pivot.index])
    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=7)
    savefig("fig4_regime_sharpe")

def figure_transaction_cost():
    df = pd.read_csv(ROBUST / "01_transaction_cost_robustness.csv")
    keep = ["baseline_lstm_xonly", "naive_ram_lstm", "regime_factor_lstm", "ram_fingan_v2", "ram_fingan_v3"]
    df = df[df["model"].isin(keep)]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for model, g in df.groupby("model"):
        g = g.sort_values("cost_bp")
        ax.plot(g["cost_bp"], g["daily_sharpe_net"], marker="o", linewidth=1.0, markersize=3, label=model)

    ax.axhline(0, linewidth=0.7)
    ax.set_xlabel("Transaction cost (bp)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.legend(frameon=False, fontsize=7)
    savefig("fig5_transaction_cost_robustness")

def figure_smoothing_effect():
    df = pd.read_csv(SMOOTH / "02_best_smoothing_by_cost.csv")
    raw = pd.read_csv(ROOT / "outputs/results/final_summary/03_raw_vs_smoothed_regime_factor.csv")

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(raw["cost_bp"], raw["raw_net_sharpe"], marker="o", linewidth=1.0, markersize=3, label="Raw")
    ax.plot(raw["cost_bp"], raw["smooth_net_sharpe"], marker="s", linewidth=1.0, markersize=3, label="Smoothed")
    ax.axhline(0, linewidth=0.7)
    ax.set_xlabel("Transaction cost (bp)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.legend(frameon=False)
    savefig("fig6_smoothing_sharpe")

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(raw["cost_bp"], raw["raw_mean_turnover"], marker="o", linewidth=1.0, markersize=3, label="Raw")
    ax.plot(raw["cost_bp"], raw["smooth_mean_turnover"], marker="s", linewidth=1.0, markersize=3, label="Smoothed")
    ax.set_xlabel("Transaction cost (bp)")
    ax.set_ylabel("Mean turnover")
    ax.legend(frameon=False)
    savefig("fig7_smoothing_turnover")

def figure_ticker_heatmap_like():
    df = pd.read_csv(ANALYSIS / "02_by_ticker_test_metrics.csv")
    keep = ["baseline_lstm_xonly", "naive_ram_lstm", "regime_factor_lstm", "ram_fingan_v2", "ram_fingan_v3"]
    df = df[df["model"].isin(keep)]

    pivot = df.pivot(index="ticker", columns="model", values="daily_sharpe")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sharpe ratio")
    savefig("fig8_ticker_model_sharpe_matrix")

def figure_covid_zoom():
    fig, ax = plt.subplots(figsize=(3.6, 2.5))

    start = pd.Timestamp("2020-02-20")
    end = pd.Timestamp("2020-06-30")

    for name, path in PRED_FILES.items():
        if not path.exists():
            continue
        d = load_pred(path)
        d = d[(d["date"] >= start) & (d["date"] <= end)].copy()
        daily = daily_pnl(d)
        daily["cum_pnl"] = daily["pnl_bp"].cumsum()
        ax.plot(daily["date"], daily["cum_pnl"], linewidth=1.0, label=name)

    ax.axhline(0, linewidth=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (bp)")
    ax.legend(frameon=False, fontsize=7)
    savefig("fig9_covid_cumulative_pnl")

def main():
    figure_overall_bar()
    figure_cumulative_pnl()
    figure_period_sharpe()
    figure_regime_sharpe()
    figure_transaction_cost()
    figure_smoothing_effect()
    figure_ticker_heatmap_like()
    figure_covid_zoom()

    print("Saved figures to:", FIG)
    for p in sorted(FIG.glob("*")):
        print(p)

if __name__ == "__main__":
    main()
