from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/kwang/RAM-FinGAN")
TICKERS_DIR = ROOT / "data_clean" / "tickers"
MAP_FILE = ROOT / "data_clean" / "stocks-etfs-list.csv"
STATE_FILE = ROOT / "data_clean" / "ram_features" / "market_state_features_trading_days_model_lag1.csv"
OUT_DIR = ROOT / "data_clean" / "ram_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2021-12-31"

L = 10
PRED = 1
H = 1

TR = 0.8
VL = 0.1

STOCKS = [
    "AMZN", "HD", "NKE",
    "CL", "EL", "KO", "PEP",
    "APA", "OXY",
    "WFC", "GS", "BLK",
    "PFE", "HUM",
    "FDX", "GD",
    "IBM", "TER",
    "ECL", "IP",
    "DTE", "WEC"
]

def load_ticker(ticker):
    path = TICKERS_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["AdjOpen", "AdjClose"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "AdjOpen", "AdjClose"])
    df = df[(df["AdjOpen"] > 0) & (df["AdjClose"] > 0)].copy()
    return df

def open_close_log_returns(df):
    log_open = np.log(df["AdjOpen"].to_numpy())
    log_close = np.log(df["AdjClose"].to_numpy())

    log_series = np.empty(2 * len(df))
    log_series[0::2] = log_open
    log_series[1::2] = log_close

    ret = np.diff(log_series)
    ret = np.clip(ret, -0.15, 0.15)

    # 每个 return 对应的“结束日期”
    # Open_t -> Close_t 和 Close_t -> Open_{t+1} 都映射到交易日日期序列附近
    dates = []
    d = df["date"].to_numpy()
    for i in range(len(df)):
        if i == 0:
            dates.append(d[i])
        else:
            dates.append(d[i])
        dates.append(d[i])
    dates = np.array(dates[1:])

    return ret, pd.to_datetime(dates)

def get_etf_map():
    m = pd.read_csv(MAP_FILE)
    if "ticker_x" not in m.columns or "ticker_y" not in m.columns:
        raise ValueError("stocks-etfs-list.csv 需要包含 ticker_x 和 ticker_y 两列")
    m["ticker_x"] = m["ticker_x"].astype(str).str.upper().str.strip()
    m["ticker_y"] = m["ticker_y"].astype(str).str.upper().str.strip()
    m["ticker_y"] = m["ticker_y"].replace({"XKL": "XLK"})
    return dict(zip(m["ticker_x"], m["ticker_y"]))

def build_stock_excess_series(stock, etf_map):
    etf = etf_map[stock]

    s_df = load_ticker(stock)
    e_df = load_ticker(etf)

    # 严格按共同日期对齐
    common_dates = sorted(set(s_df["date"]).intersection(set(e_df["date"])))
    s_df = s_df[s_df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
    e_df = e_df[e_df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)

    s_ret, ret_dates = open_close_log_returns(s_df)
    e_ret, _ = open_close_log_returns(e_df)

    n = min(len(s_ret), len(e_ret), len(ret_dates))
    x = s_ret[:n] - e_ret[:n]
    dates = ret_dates[:n]

    return pd.DataFrame({
        "date": dates,
        "ticker": stock,
        "sector_etf": etf,
        "excess_return": x
    })

def make_windows(excess_df, state_df):
    x = excess_df["excess_return"].to_numpy()
    dates = excess_df["date"].to_numpy()

    rows = []
    n_total = len(x)
    for start in range(0, n_total - L - PRED + 1, H):
        cond = x[start:start+L]
        y = x[start+L]
        target_date = pd.to_datetime(dates[start+L])

        row = {
            "date": target_date,
            "ticker": excess_df["ticker"].iloc[0],
            "sector_etf": excess_df["sector_etf"].iloc[0],
            "y": y,
        }
        for j in range(L):
            row[f"x_{j+1}"] = cond[j]
        rows.append(row)

    panel = pd.DataFrame(rows)

    # 按日期合并 market state features
    panel = panel.merge(state_df, on="date", how="left")

    # split 按原始 Fin-GAN 的 80/10/10 逻辑
    N = len(panel)
    n_tr = int(TR * N)
    n_vl = int(VL * N)

    panel["split"] = "test"
    panel.loc[:n_tr-1, "split"] = "train"
    panel.loc[n_tr:n_tr+n_vl-1, "split"] = "val"

    return panel

def main():
    state = pd.read_csv(STATE_FILE, parse_dates=["date"])
    state = state.sort_values("date").reset_index(drop=True)

    etf_map = get_etf_map()

    all_panels = []

    for stock in STOCKS:
        print(f"\n===== {stock} =====")
        excess = build_stock_excess_series(stock, etf_map)
        panel = make_windows(excess, state)

        missing = int(panel.isna().sum().sum())
        print("panel shape:", panel.shape)
        print("date:", panel["date"].min(), "to", panel["date"].max())
        print("split counts:")
        print(panel["split"].value_counts())
        print("missing total:", missing)

        if missing > 0:
            # model-ready: 前向填充后再填 0
            panel = panel.sort_values("date")
            feature_cols = [c for c in panel.columns if c not in ["date", "ticker", "sector_etf", "split"]]
            panel[feature_cols] = panel[feature_cols].ffill().fillna(0.0)

        out = OUT_DIR / f"{stock}_panel.csv"
        panel.to_csv(out, index=False)
        print("saved:", out)

        all_panels.append(panel)

    full = pd.concat(all_panels, ignore_index=True)
    full_out = OUT_DIR / "ALL_STOCKS_panel.csv"
    full.to_csv(full_out, index=False)

    print("\nDONE.")
    print("saved full panel:", full_out)
    print("full shape:", full.shape)
    print("full split counts:")
    print(full["split"].value_counts())

if __name__ == "__main__":
    main()
