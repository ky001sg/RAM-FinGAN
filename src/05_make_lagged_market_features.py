from pathlib import Path
import pandas as pd

ROOT = Path("/home/kwang/RAM-FinGAN")
IN_FILE = ROOT / "data_clean" / "ram_features" / "market_state_features_trading_days_model.csv"
OUT_FILE = ROOT / "data_clean" / "ram_features" / "market_state_features_trading_days_model_lag1.csv"

df = pd.read_csv(IN_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "date"]

lagged = df.copy()
lagged[feature_cols] = lagged[feature_cols].shift(1)

# 第一行没有前一交易日信息，填 0
lagged[feature_cols] = lagged[feature_cols].fillna(0.0)

lagged.to_csv(OUT_FILE, index=False)

print("[OK] saved:", OUT_FILE)
print("shape:", lagged.shape)
print("date:", lagged["date"].min(), "to", lagged["date"].max())
print("missing:", int(lagged.isna().sum().sum()))
