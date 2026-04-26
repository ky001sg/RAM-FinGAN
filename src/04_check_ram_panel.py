from pathlib import Path
import pandas as pd

ROOT = Path("/home/kwang/RAM-FinGAN")
PANEL_DIR = ROOT / "data_clean" / "ram_panel"

full = pd.read_csv(PANEL_DIR / "ALL_STOCKS_panel.csv", parse_dates=["date"])

print("shape:", full.shape)
print("date:", full["date"].min(), "to", full["date"].max())
print("\nTickers:", sorted(full["ticker"].unique()))
print("\nSplit counts:")
print(full["split"].value_counts())

print("\nRows by ticker:")
print(full.groupby("ticker").size())

print("\nMissing total:", int(full.isna().sum().sum()))
print("\nTop missing columns:")
print(full.isna().sum().sort_values(ascending=False).head(20))

print("\nCore columns:")
core = ["date", "ticker", "sector_etf", "y"] + [f"x_{i}" for i in range(1, 11)] + ["split"]
print(core)
print("\nSample:")
print(full[core].head())
