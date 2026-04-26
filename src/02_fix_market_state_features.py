from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/kwang/RAM-FinGAN")
IN_DIR = ROOT / "data_clean" / "ram_features"
OUT_DIR = IN_DIR

market = pd.read_csv(IN_DIR / "market_etf_features.csv", parse_dates=["date"])
external = pd.read_csv(IN_DIR / "external_macro_features.csv", parse_dates=["date"])

# 只用 market ETF 的交易日作为最终日期骨架，避免 FRED 非交易日进入模型
base_dates = market[["date"]].drop_duplicates().sort_values("date").copy()

# 去掉 external 中重复的 proxy 列，避免 _x/_y
drop_cols = [c for c in external.columns if c.startswith("credit_proxy_LQD_minus_HYG")]
external = external.drop(columns=drop_cols, errors="ignore")

# 合并
final = base_dates.merge(market, on="date", how="left")
final = final.merge(external, on="date", how="left")

# 保存 BAML 原始是否可用的标记：注意这里是在 fill 之前做
if "BAMLH0A0HYM2" in final.columns:
    final["credit_oas_raw_available"] = final["BAMLH0A0HYM2"].notna().astype(int)

# FRED 宏观变量按交易日向前填充
macro_cols = ["VIXCLS", "DGS10", "DGS2", "DGS3MO", "BAMLH0A0HYM2"]
for c in macro_cols:
    if c in final.columns:
        final[c] = final[c].ffill()

# 重新构造宏观衍生变量
if {"DGS10", "DGS2"}.issubset(final.columns):
    final["yield_slope_10y_2y"] = final["DGS10"] - final["DGS2"]
if {"DGS10", "DGS3MO"}.issubset(final.columns):
    final["yield_slope_10y_3mo"] = final["DGS10"] - final["DGS3MO"]
if "VIXCLS" in final.columns:
    final["vix_change"] = final["VIXCLS"].diff()
if "BAMLH0A0HYM2" in final.columns:
    final["credit_oas_change"] = final["BAMLH0A0HYM2"].diff()

# 用 HYG/LQD 重新构造信用代理
if {"LQD_ret", "HYG_ret"}.issubset(final.columns):
    final["credit_proxy_LQD_minus_HYG"] = final["LQD_ret"] - final["HYG_ret"]
    roll_mean = final["credit_proxy_LQD_minus_HYG"].rolling(252, min_periods=60).mean()
    roll_std = final["credit_proxy_LQD_minus_HYG"].rolling(252, min_periods=60).std()
    final["credit_proxy_z"] = (final["credit_proxy_LQD_minus_HYG"] - roll_mean) / roll_std

# 对于 2000 年早期不存在的 ETF 特征，不强行删除日期；保留缺失，让后续模型使用 mask/填补
# 这里先生成两个版本：
# 1. 原始带缺失版本
# 2. 模型可用的填充版本

raw_out = OUT_DIR / "market_state_features_trading_days_raw.csv"
final.to_csv(raw_out, index=False)

model = final.copy()

# 对市场特征：先 forward-fill，再对最前面仍缺失的部分填 0
feature_cols = [c for c in model.columns if c != "date"]
model[feature_cols] = model[feature_cols].ffill()
model[feature_cols] = model[feature_cols].fillna(0.0)

model_out = OUT_DIR / "market_state_features_trading_days_model.csv"
model.to_csv(model_out, index=False)

print("[OK] raw trading-day features:", raw_out)
print("shape:", final.shape)
print("date:", final["date"].min(), "to", final["date"].max())

print("\n[OK] model-ready trading-day features:", model_out)
print("shape:", model.shape)
print("missing total:", int(model.isna().sum().sum()))

print("\nTop missing columns in raw version:")
print(final.isna().sum().sort_values(ascending=False).head(20))
