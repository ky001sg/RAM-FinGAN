from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/kwang/RAM-FinGAN")
OUT = ROOT / "outputs" / "results" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "baseline_lstm_xonly": ROOT / "outputs/results/baselines/baseline_lstm_xonly_test_predictions.csv",
    "naive_ram_lstm": ROOT / "outputs/results/baselines/ram_lstm_x_market_test_predictions.csv",
    "regime_factor_lstm": ROOT / "outputs/results/regime_baselines/regime_factor_lstm_test_predictions.csv",
    "ram_fingan_v1": ROOT / "outputs/results/ram_fingan_v1/ram_fingan_v1_test_predictions.csv",
    "ram_fingan_v2": ROOT / "outputs/results/ram_fingan_v2_pretrain/ram_fingan_v2_test_predictions.csv",
    "ram_fingan_v3": ROOT / "outputs/results/ram_fingan_v3_econ/ram_fingan_v3_test_predictions.csv",
    "regime_econ_lstm_v4": ROOT / "outputs/results/regime_econ_lstm_v4/regime_econ_lstm_v4_test_predictions.csv",
}

COST_BPS = [0, 1, 2, 5, 10]
N_BOOT = 2000
SEED = 42
rng = np.random.default_rng(SEED)


def load_pred(path):
    d = pd.read_csv(path, parse_dates=["date"])

    if "pred" not in d.columns:
        if "pred_mean" in d.columns:
            d["pred"] = d["pred_mean"]
        elif "position" in d.columns:
            d["pred"] = d["position"]
        else:
            raise ValueError(f"Cannot infer prediction column from {path}")

    if "position" not in d.columns:
        # LSTM-style models: deterministic sign position
        d["position"] = np.sign(d["pred"])

    d["position"] = d["position"].fillna(0.0).clip(-1.0, 1.0)
    d["gross_pnl_bp"] = 10000.0 * d["position"] * d["y"]

    # turnover by ticker, ordered by date and within-day rows
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    d["prev_position"] = d.groupby("ticker")["position"].shift(1).fillna(0.0)
    d["turnover"] = (d["position"] - d["prev_position"]).abs()

    return d


def daily_metrics(d, cost_bp):
    g = d.copy()
    g["cost_bp_paid"] = cost_bp * g["turnover"]
    g["net_pnl_bp"] = g["gross_pnl_bp"] - g["cost_bp_paid"]

    daily = g.groupby(["ticker", "date"], as_index=False).agg(
        gross_pnl_bp=("gross_pnl_bp", "sum"),
        net_pnl_bp=("net_pnl_bp", "sum"),
        turnover=("turnover", "sum"),
    )

    mu = daily["net_pnl_bp"].mean()
    sig = daily["net_pnl_bp"].std(ddof=0)
    sr = np.sqrt(252) * mu / sig if sig > 0 else 0.0

    return {
        "cost_bp": cost_bp,
        "mean_net_pnl_bp": float(mu),
        "daily_sharpe_net": float(sr),
        "mean_turnover": float(daily["turnover"].mean()),
        "n_ticker_days": int(len(daily)),
    }, daily


def bootstrap_diff(daily_a, daily_b):
    # align by ticker-date
    a = daily_a[["ticker", "date", "net_pnl_bp"]].rename(columns={"net_pnl_bp": "a"})
    b = daily_b[["ticker", "date", "net_pnl_bp"]].rename(columns={"net_pnl_bp": "b"})
    m = a.merge(b, on=["ticker", "date"], how="inner")
    diff = (m["a"] - m["b"]).values

    n = len(diff)
    boot_means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot_means[i] = diff[idx].mean()

    obs = diff.mean()
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    p_value = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())

    return {
        "mean_diff_net_pnl_bp": float(obs),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value_two_sided": float(p_value),
        "n_aligned": int(n),
    }


def main():
    preds = {}
    for model, path in MODELS.items():
        if not path.exists():
            print("[WARN] missing:", model, path)
            continue
        preds[model] = load_pred(path)
        print("[OK] loaded", model, preds[model].shape)

    cost_rows = []
    daily_by_model_cost = {}

    for model, d in preds.items():
        for cost in COST_BPS:
            row, daily = daily_metrics(d, cost)
            row["model"] = model
            cost_rows.append(row)
            daily_by_model_cost[(model, cost)] = daily

    cost_table = pd.DataFrame(cost_rows)
    cost_table = cost_table.sort_values(["cost_bp", "daily_sharpe_net"], ascending=[True, False])
    cost_table.to_csv(OUT / "01_transaction_cost_robustness.csv", index=False)

    # bootstrap comparisons against current best model
    main_model = "regime_factor_lstm"
    comparisons = [
        "baseline_lstm_xonly",
        "naive_ram_lstm",
        "ram_fingan_v2",
        "ram_fingan_v3",
        "regime_econ_lstm_v4",
    ]

    boot_rows = []
    for cost in COST_BPS:
        if (main_model, cost) not in daily_by_model_cost:
            continue
        for other in comparisons:
            if (other, cost) not in daily_by_model_cost:
                continue
            res = bootstrap_diff(
                daily_by_model_cost[(main_model, cost)],
                daily_by_model_cost[(other, cost)]
            )
            res["cost_bp"] = cost
            res["model_a"] = main_model
            res["model_b"] = other
            boot_rows.append(res)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "02_bootstrap_regime_factor_vs_others.csv", index=False)

    # best model by transaction cost
    best_by_cost = cost_table.sort_values(["cost_bp", "daily_sharpe_net"], ascending=[True, False]).groupby("cost_bp").head(1)
    best_by_cost.to_csv(OUT / "03_best_model_by_cost.csv", index=False)

    report = []
    report.append("# Robustness: Transaction Costs and Bootstrap Tests\n")
    report.append("## Best model by transaction cost\n")
    report.append(best_by_cost[["cost_bp", "model", "mean_net_pnl_bp", "daily_sharpe_net", "mean_turnover"]].to_markdown(index=False))
    report.append("\n## Full transaction cost table\n")
    report.append(cost_table[["cost_bp", "model", "mean_net_pnl_bp", "daily_sharpe_net", "mean_turnover"]].to_markdown(index=False))
    report.append("\n## Bootstrap: regime_factor_lstm minus competitors\n")
    report.append(boot[["cost_bp", "model_a", "model_b", "mean_diff_net_pnl_bp", "ci95_low", "ci95_high", "p_value_two_sided"]].to_markdown(index=False))

    report.append("""
## Interpretation guide

- `daily_sharpe_net` is the annualized Sharpe after subtracting transaction costs.
- Cost is applied as `cost_bp * abs(position_t - position_{t-1})`.
- Positive bootstrap mean difference means `regime_factor_lstm` has higher net daily PnL than the comparison model.
- If the 95% CI is above zero, the improvement is robust under bootstrap resampling.
""")

    (OUT / "robustness_report.md").write_text("\n".join(report))

    print("\n===== BEST BY COST =====")
    print(best_by_cost[["cost_bp", "model", "mean_net_pnl_bp", "daily_sharpe_net", "mean_turnover"]])

    print("\n===== BOOTSTRAP =====")
    print(boot[["cost_bp", "model_b", "mean_diff_net_pnl_bp", "ci95_low", "ci95_high", "p_value_two_sided"]])

    print("\nGenerated:")
    for p in sorted(OUT.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
