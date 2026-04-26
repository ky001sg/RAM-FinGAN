from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/kwang/RAM-FinGAN")
PRED_FILE = ROOT / "outputs" / "results" / "regime_baselines" / "regime_factor_lstm_test_predictions.csv"
OUT = ROOT / "outputs" / "results" / "position_smoothing"
OUT.mkdir(parents=True, exist_ok=True)

COST_BPS = [0, 1, 2, 5, 10]

# threshold: if abs(pred) < threshold, position target becomes 0
THRESHOLDS = [
    0.0,
    0.0001,
    0.0002,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
]

# alpha: position_t = (1-alpha) * position_{t-1} + alpha * target_t
# alpha=1 means no smoothing
ALPHAS = [1.0, 0.75, 0.50, 0.30, 0.20, 0.10]

# k controls tanh position sharpness
# large k approaches sign-like position
K_VALUES = [50, 100, 200, 500]


def load_pred():
    d = pd.read_csv(PRED_FILE, parse_dates=["date"])
    if "pred" not in d.columns:
        if "pred_mean" in d.columns:
            d["pred"] = d["pred_mean"]
        else:
            raise ValueError("Prediction file has no pred or pred_mean column.")

    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    return d


def make_positions(d, threshold, alpha, k):
    out = d.copy()

    raw_signal = out["pred"].values.copy()
    target = np.tanh(k * raw_signal)

    # no-trade band
    target[np.abs(raw_signal) < threshold] = 0.0
    out["target_position"] = target

    # smoothing by ticker
    smoothed = np.zeros(len(out), dtype=float)

    for ticker, idx in out.groupby("ticker").groups.items():
        idx = np.array(list(idx))
        pos_prev = 0.0
        for j in idx:
            pos = (1.0 - alpha) * pos_prev + alpha * out.at[j, "target_position"]
            smoothed[j] = pos
            pos_prev = pos

    out["position"] = smoothed
    out["gross_pnl_bp"] = 10000.0 * out["position"] * out["y"]

    out["prev_position"] = out.groupby("ticker")["position"].shift(1).fillna(0.0)
    out["turnover"] = (out["position"] - out["prev_position"]).abs()

    return out


def evaluate(d, cost_bp):
    g = d.copy()
    g["cost_paid_bp"] = cost_bp * g["turnover"]
    g["net_pnl_bp"] = g["gross_pnl_bp"] - g["cost_paid_bp"]

    daily = g.groupby(["ticker", "date"], as_index=False).agg(
        gross_pnl_bp=("gross_pnl_bp", "sum"),
        net_pnl_bp=("net_pnl_bp", "sum"),
        turnover=("turnover", "sum"),
        avg_abs_position=("position", lambda x: np.mean(np.abs(x))),
    )

    mu = daily["net_pnl_bp"].mean()
    sig = daily["net_pnl_bp"].std(ddof=0)
    sharpe = np.sqrt(252) * mu / sig if sig > 0 else 0.0

    gross_mu = daily["gross_pnl_bp"].mean()
    gross_sig = daily["gross_pnl_bp"].std(ddof=0)
    gross_sharpe = np.sqrt(252) * gross_mu / gross_sig if gross_sig > 0 else 0.0

    return {
        "cost_bp": cost_bp,
        "mean_gross_pnl_bp": float(gross_mu),
        "mean_net_pnl_bp": float(mu),
        "gross_sharpe": float(gross_sharpe),
        "net_sharpe": float(sharpe),
        "mean_turnover": float(daily["turnover"].mean()),
        "avg_abs_position": float(daily["avg_abs_position"].mean()),
        "n_ticker_days": int(len(daily)),
    }, daily


def main():
    base = load_pred()
    print("[OK] loaded:", PRED_FILE)
    print("shape:", base.shape)
    print("date:", base["date"].min(), "to", base["date"].max())
    print("tickers:", sorted(base["ticker"].unique()))

    rows = []
    best_daily_by_cost = {}
    best_config_by_cost = {}

    for threshold in THRESHOLDS:
        for alpha in ALPHAS:
            for k in K_VALUES:
                pos_df = make_positions(base, threshold=threshold, alpha=alpha, k=k)

                for cost in COST_BPS:
                    m, daily = evaluate(pos_df, cost)
                    row = {
                        "threshold": threshold,
                        "alpha": alpha,
                        "k": k,
                        **m,
                    }
                    rows.append(row)

                    key = cost
                    if key not in best_config_by_cost or row["net_sharpe"] > best_config_by_cost[key]["net_sharpe"]:
                        best_config_by_cost[key] = row
                        best_daily_by_cost[key] = daily.copy()
                        best_daily_by_cost[key]["threshold"] = threshold
                        best_daily_by_cost[key]["alpha"] = alpha
                        best_daily_by_cost[key]["k"] = k
                        best_daily_by_cost[key]["cost_bp"] = cost

    grid = pd.DataFrame(rows)
    grid = grid.sort_values(["cost_bp", "net_sharpe"], ascending=[True, False])
    grid.to_csv(OUT / "01_smoothing_grid_results.csv", index=False)

    best = pd.DataFrame(list(best_config_by_cost.values())).sort_values("cost_bp")
    best.to_csv(OUT / "02_best_smoothing_by_cost.csv", index=False)

    for cost, daily in best_daily_by_cost.items():
        daily.to_csv(OUT / f"daily_pnl_best_cost_{cost}bp.csv", index=False)

    # Compare with original sign-like regime_factor_lstm from robustness result
    report = []
    report.append("# Position Smoothing / No-Trade Band Results\n")
    report.append("## Best configuration by transaction cost\n")
    report.append(best[[
        "cost_bp", "threshold", "alpha", "k",
        "mean_gross_pnl_bp", "mean_net_pnl_bp",
        "gross_sharpe", "net_sharpe",
        "mean_turnover", "avg_abs_position"
    ]].to_markdown(index=False))

    report.append("""
## Interpretation

- `threshold` creates a no-trade band: if `abs(pred) < threshold`, the target position is zero.
- `alpha` controls smoothing: smaller alpha means slower position adjustment and lower turnover.
- `k` controls how aggressively predictions are mapped into positions through `tanh(k * pred)`.
- The goal is not to maximize zero-cost Sharpe only, but to preserve Sharpe under realistic transaction costs.
""")

    report.append("\n## Top 10 configurations for each cost\n")
    for cost in COST_BPS:
        top = grid[grid["cost_bp"] == cost].head(10)
        report.append(f"\n### Cost = {cost} bp\n")
        report.append(top[[
            "threshold", "alpha", "k",
            "mean_net_pnl_bp", "net_sharpe",
            "mean_turnover", "avg_abs_position"
        ]].to_markdown(index=False))

    (OUT / "position_smoothing_report.md").write_text("\n".join(report))

    print("\n===== BEST BY COST =====")
    print(best[[
        "cost_bp", "threshold", "alpha", "k",
        "mean_net_pnl_bp", "net_sharpe",
        "mean_turnover", "avg_abs_position"
    ]])

    print("\nGenerated:")
    for p in sorted(OUT.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
