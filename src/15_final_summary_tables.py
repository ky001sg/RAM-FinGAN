from pathlib import Path
import pandas as pd

ROOT = Path("/home/kwang/RAM-FinGAN")
OUT = ROOT / "outputs" / "results" / "final_summary"
OUT.mkdir(parents=True, exist_ok=True)

analysis_dir = ROOT / "outputs" / "results" / "analysis"
robust_dir = ROOT / "outputs" / "results" / "robustness"
smooth_dir = ROOT / "outputs" / "results" / "position_smoothing"

overall = pd.read_csv(analysis_dir / "01_overall_test_metrics.csv")
tc = pd.read_csv(robust_dir / "01_transaction_cost_robustness.csv")
best_cost = pd.read_csv(robust_dir / "03_best_model_by_cost.csv")
smooth_best = pd.read_csv(smooth_dir / "02_best_smoothing_by_cost.csv")
by_period = pd.read_csv(analysis_dir / "04_by_period_test_metrics.csv")
by_regime = pd.read_csv(analysis_dir / "03_by_regime_test_metrics.csv")
by_ticker_best = pd.read_csv(analysis_dir / "05_best_model_by_ticker.csv")

# 1. Overall model ranking
overall_rank = overall[[
    "model", "direction_acc", "mean_pnl_bp", "daily_sharpe", "rmse", "mae"
]].sort_values("daily_sharpe", ascending=False)
overall_rank.to_csv(OUT / "01_overall_model_ranking.csv", index=False)

# 2. Transaction cost table, focused on key models
key_models = [
    "regime_factor_lstm",
    "baseline_lstm_xonly",
    "naive_ram_lstm",
    "ram_fingan_v2",
    "ram_fingan_v3",
    "regime_econ_lstm_v4",
]
tc_key = tc[tc["model"].isin(key_models)].copy()
tc_key = tc_key[[
    "cost_bp", "model", "mean_net_pnl_bp", "daily_sharpe_net", "mean_turnover"
]].sort_values(["cost_bp", "daily_sharpe_net"], ascending=[True, False])
tc_key.to_csv(OUT / "02_transaction_cost_key_models.csv", index=False)

# 3. Original regime factor vs smoothed regime factor
regime_raw = tc[tc["model"] == "regime_factor_lstm"][[
    "cost_bp", "mean_net_pnl_bp", "daily_sharpe_net", "mean_turnover"
]].copy()
regime_raw = regime_raw.rename(columns={
    "mean_net_pnl_bp": "raw_mean_net_pnl_bp",
    "daily_sharpe_net": "raw_net_sharpe",
    "mean_turnover": "raw_mean_turnover",
})

smooth = smooth_best[[
    "cost_bp", "threshold", "alpha", "k",
    "mean_net_pnl_bp", "net_sharpe", "mean_turnover", "avg_abs_position"
]].copy()
smooth = smooth.rename(columns={
    "mean_net_pnl_bp": "smooth_mean_net_pnl_bp",
    "net_sharpe": "smooth_net_sharpe",
    "mean_turnover": "smooth_mean_turnover",
})

deploy = regime_raw.merge(smooth, on="cost_bp", how="outer")
deploy["sharpe_improvement_smooth_minus_raw"] = deploy["smooth_net_sharpe"] - deploy["raw_net_sharpe"]
deploy["turnover_reduction_pct"] = 100 * (1 - deploy["smooth_mean_turnover"] / deploy["raw_mean_turnover"])
deploy.to_csv(OUT / "03_raw_vs_smoothed_regime_factor.csv", index=False)

# 4. Period table
period_pivot = by_period.pivot_table(
    index="period",
    columns="model",
    values="daily_sharpe",
    aggfunc="first"
)
period_pivot.to_csv(OUT / "04_period_sharpe_pivot.csv")

# 5. Regime table
regime_pivot = by_regime.pivot_table(
    index="regime_cluster",
    columns="model",
    values="daily_sharpe",
    aggfunc="first"
)
regime_pivot.to_csv(OUT / "05_regime_sharpe_pivot.csv")

# 6. Best model by ticker
by_ticker_best[[
    "ticker", "model", "daily_sharpe", "direction_acc", "mean_pnl_bp"
]].to_csv(OUT / "06_best_model_by_ticker.csv", index=False)

# 7. Final paper-style compact table
compact_rows = []

best_overall = overall_rank.iloc[0]
compact_rows.append({
    "finding": "Best predictive model without transaction cost",
    "evidence": f"{best_overall['model']} has test Sharpe {best_overall['daily_sharpe']:.3f}, direction accuracy {best_overall['direction_acc']:.3%}, mean PnL {best_overall['mean_pnl_bp']:.3f} bp.",
})

covid = by_period[(by_period["period"] == "covid_shock") & (by_period["model"] == "regime_factor_lstm")]
if len(covid):
    r = covid.iloc[0]
    compact_rows.append({
        "finding": "Regime-factor model is strongest during COVID shock",
        "evidence": f"During COVID shock, regime_factor_lstm Sharpe = {r['daily_sharpe']:.3f}.",
    })

raw_2bp = deploy[deploy["cost_bp"] == 2]
if len(raw_2bp):
    r = raw_2bp.iloc[0]
    compact_rows.append({
        "finding": "Transaction costs expose high-turnover weakness",
        "evidence": f"At 2bp cost, raw regime_factor_lstm Sharpe drops to {r['raw_net_sharpe']:.3f} with turnover {r['raw_mean_turnover']:.3f}.",
    })

smooth_5bp = deploy[deploy["cost_bp"] == 5]
if len(smooth_5bp):
    r = smooth_5bp.iloc[0]
    compact_rows.append({
        "finding": "Position smoothing improves deployability under costs",
        "evidence": f"At 5bp cost, smoothing improves Sharpe from {r['raw_net_sharpe']:.3f} to {r['smooth_net_sharpe']:.3f}, reducing turnover by {r['turnover_reduction_pct']:.1f}%.",
    })

compact = pd.DataFrame(compact_rows)
compact.to_csv(OUT / "07_key_findings_for_paper.csv", index=False)

# Markdown report
report = []
report.append("# Final RAM-FinGAN / Regime-Factor Experiment Summary\n")

report.append("## 1. Overall model ranking\n")
report.append(overall_rank.to_markdown(index=False))

report.append("\n## 2. Transaction-cost robustness: key models\n")
report.append(tc_key.to_markdown(index=False))

report.append("\n## 3. Raw vs smoothed regime-factor deployment\n")
report.append(deploy.to_markdown(index=False))

report.append("\n## 4. Sharpe by period\n")
report.append(period_pivot.to_markdown())

report.append("\n## 5. Sharpe by regime cluster\n")
report.append(regime_pivot.to_markdown())

report.append("\n## 6. Best model by ticker\n")
report.append(by_ticker_best[[
    "ticker", "model", "daily_sharpe", "direction_acc", "mean_pnl_bp"
]].to_markdown(index=False))

report.append("\n## 7. Key findings for paper\n")
report.append(compact.to_markdown(index=False))

report.append("""
## Final interpretation

The strongest evidence does not support a simple claim that GAN variants dominate.
Instead, the evidence supports a more nuanced and more defensible claim:

1. Raw market-state features are not robust when directly concatenated with LSTM inputs.
2. Compressing market information into regime factors gives the strongest out-of-sample predictive performance.
3. Regime-factor LSTM is particularly strong in regime-sensitive periods such as COVID shock.
4. GAN-style adversarial training and direct economic-loss training are unstable in this small-signal financial setting.
5. The main remaining practical issue is turnover. Position smoothing / no-trade deployment makes the regime-factor signal more robust under transaction costs.

Suggested paper direction:

**Regime-factor representation and turnover-aware deployment for robust stock--ETF excess-return forecasting under market regime shifts.**

The GAN variants should be kept as ablation studies, not as the main final model.
""")

(OUT / "final_summary_report.md").write_text("\n".join(report))

print("\nGenerated final summary files:")
for p in sorted(OUT.glob("*")):
    print(p)

print("\nMain report:")
print(OUT / "final_summary_report.md")
