from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/kwang/RAM-FinGAN")
RAW_CRSP = ROOT / "data_raw" / "crsp"
RAW_EXT = ROOT / "data_raw" / "external"
CLEAN = ROOT / "data_clean"
TICKERS_DIR = CLEAN / "tickers"
OUT = ROOT / "data_clean" / "ram_features"
OUT.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2021-12-31"

MARKET_ETFS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "LQD"]

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "ticker":
            rename[c] = "TICKER"
        elif cl == "tsymbol":
            rename[c] = "TSYMBOL"
        elif cl == "prc":
            rename[c] = "PRC"
        elif cl == "openprc":
            rename[c] = "OPENPRC"
        elif cl == "cfacpr":
            rename[c] = "CFACPR"
        elif cl == "vol":
            rename[c] = "VOL"
        elif cl == "shrout":
            rename[c] = "SHROUT"
        elif cl == "ret":
            rename[c] = "RET"
        elif cl == "retx":
            rename[c] = "RETX"
        elif cl == "date":
            rename[c] = "date"
    df = df.rename(columns=rename)
    if "TICKER" not in df.columns and "TSYMBOL" in df.columns:
        df["TICKER"] = df["TSYMBOL"]
    return df

def load_market_etfs():
    path = RAW_CRSP / "Market-ETFs-data.csv"
    df = pd.read_csv(path)
    df = normalize_cols(df)

    required = ["date", "TICKER", "PRC", "OPENPRC", "CFACPR", "VOL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Market-ETFs-data.csv 缺少必要列: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["TICKER"] = df["TICKER"].astype(str).str.upper().str.strip()

    # 关键处理：QQQQ 是 QQQ 的历史 ticker，统一合并成 QQQ
    df["TICKER"] = df["TICKER"].replace({"QQQQ": "QQQ"})

    for c in ["PRC", "OPENPRC", "CFACPR", "VOL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "SHROUT" in df.columns:
        df["SHROUT"] = pd.to_numeric(df["SHROUT"], errors="coerce")
    else:
        df["SHROUT"] = np.nan

    df["AdjClose"] = df["PRC"].abs() / df["CFACPR"]
    df["AdjOpen"] = df["OPENPRC"].abs() / df["CFACPR"]

    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df = df[df["TICKER"].isin(MARKET_ETFS)].copy()
    df = df.dropna(subset=["date", "TICKER", "AdjClose", "AdjOpen"])
    df = df[(df["AdjClose"] > 0) & (df["AdjOpen"] > 0)].copy()

    # 如果 QQQ / QQQQ 合并后出现重复日期，保留最后一条
    df = df.sort_values(["TICKER", "date"])
    df = df.drop_duplicates(subset=["TICKER", "date"], keep="last")

    out_file = OUT / "market_etfs_clean.csv"
    df.to_csv(out_file, index=False)

    print("\n[OK] market_etfs_clean.csv")
    print(df.groupby("TICKER")["date"].agg(["min", "max", "count"]))

    return df

def make_market_features(market_df):
    frames = []
    for ticker, g in market_df.groupby("TICKER"):
        g = g.sort_values("date").copy()
        g[f"{ticker}_log_close"] = np.log(g["AdjClose"])
        g[f"{ticker}_ret"] = g[f"{ticker}_log_close"].diff()
        g[f"{ticker}_rv20"] = g[f"{ticker}_ret"].rolling(20).std()
        g[f"{ticker}_mom20"] = g[f"{ticker}_ret"].rolling(20).sum()
        g[f"{ticker}_drawdown60"] = g["AdjClose"] / g["AdjClose"].rolling(60).max() - 1

        if "SHROUT" in g.columns:
            g[f"{ticker}_turnover"] = g["VOL"] / g["SHROUT"].replace(0, np.nan)
        else:
            g[f"{ticker}_turnover"] = np.nan

        keep = ["date", f"{ticker}_ret", f"{ticker}_rv20", f"{ticker}_mom20", f"{ticker}_drawdown60", f"{ticker}_turnover"]
        frames.append(g[keep])

    feat = frames[0]
    for f in frames[1:]:
        feat = feat.merge(f, on="date", how="outer")

    feat = feat.sort_values("date")
    feat = feat[(feat["date"] >= START) & (feat["date"] <= END)].copy()

    # 相对风险偏好特征
    if {"QQQ_ret", "SPY_ret"}.issubset(feat.columns):
        feat["QQQ_minus_SPY"] = feat["QQQ_ret"] - feat["SPY_ret"]
    if {"IWM_ret", "SPY_ret"}.issubset(feat.columns):
        feat["IWM_minus_SPY"] = feat["IWM_ret"] - feat["SPY_ret"]
    if {"LQD_ret", "HYG_ret"}.issubset(feat.columns):
        # 数值越大，表示 HYG 相对 LQD 越弱，信用压力越高
        feat["credit_proxy_LQD_minus_HYG"] = feat["LQD_ret"] - feat["HYG_ret"]

    out_file = OUT / "market_etf_features.csv"
    feat.to_csv(out_file, index=False)

    print("\n[OK] market_etf_features.csv")
    print("shape:", feat.shape)
    print("date:", feat["date"].min(), "to", feat["date"].max())

    return feat

def load_fred_series(filename, value_name):
    path = RAW_EXT / filename
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # FRED 常见格式：observation_date,value 或 DATE,SERIES
    date_col = None
    for c in df.columns:
        if c.lower() in ["date", "observation_date"]:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    val_col = [c for c in df.columns if c != date_col][0]

    out = df[[date_col, val_col]].copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"])
    out[value_name] = pd.to_numeric(out[value_name].replace(".", np.nan), errors="coerce")
    out = out[(out["date"] >= START) & (out["date"] <= END)].copy()
    return out

def make_external_features(market_feat):
    vix = load_fred_series("VIXCLS.csv", "VIXCLS")
    dgs10 = load_fred_series("DGS10.csv", "DGS10")
    dgs2 = load_fred_series("DGS2.csv", "DGS2")
    dgs3mo = load_fred_series("DGS3MO.csv", "DGS3MO")
    baml = load_fred_series("BAMLH0A0HYM2.csv", "BAMLH0A0HYM2")

    ext = vix.merge(dgs10, on="date", how="outer")
    ext = ext.merge(dgs2, on="date", how="outer")
    ext = ext.merge(dgs3mo, on="date", how="outer")
    ext = ext.merge(baml, on="date", how="outer")

    ext = ext.sort_values("date")
    ext = ext[(ext["date"] >= START) & (ext["date"] <= END)].copy()

    # FRED 非交易日和假日会缺失，先按日期 forward-fill
    ext[["VIXCLS", "DGS10", "DGS2", "DGS3MO", "BAMLH0A0HYM2"]] = (
        ext[["VIXCLS", "DGS10", "DGS2", "DGS3MO", "BAMLH0A0HYM2"]].ffill()
    )

    ext["yield_slope_10y_2y"] = ext["DGS10"] - ext["DGS2"]
    ext["yield_slope_10y_3mo"] = ext["DGS10"] - ext["DGS3MO"]
    ext["vix_change"] = ext["VIXCLS"].diff()
    ext["credit_oas_change"] = ext["BAMLH0A0HYM2"].diff()
    ext["credit_oas_missing"] = ext["BAMLH0A0HYM2"].isna().astype(int)

    # 合并 HYG-LQD 代理，补 BAMLH0A0HYM2 后段缺失
    proxy = market_feat[["date", "credit_proxy_LQD_minus_HYG"]].copy() if "credit_proxy_LQD_minus_HYG" in market_feat.columns else pd.DataFrame({"date": []})
    ext = ext.merge(proxy, on="date", how="left")

    # 用滚动 z-score 标准化信用代理
    if "credit_proxy_LQD_minus_HYG" in ext.columns:
        roll_mean = ext["credit_proxy_LQD_minus_HYG"].rolling(252, min_periods=60).mean()
        roll_std = ext["credit_proxy_LQD_minus_HYG"].rolling(252, min_periods=60).std()
        ext["credit_proxy_z"] = (ext["credit_proxy_LQD_minus_HYG"] - roll_mean) / roll_std
    else:
        ext["credit_proxy_z"] = np.nan

    # OAS 有值就用 OAS；没有值则保留 proxy，后面建模时可二者同时输入
    ext["credit_oas_available"] = ext["BAMLH0A0HYM2"].notna().astype(int)

    out_file = OUT / "external_macro_features.csv"
    ext.to_csv(out_file, index=False)

    print("\n[OK] external_macro_features.csv")
    print("shape:", ext.shape)
    print("date:", ext["date"].min(), "to", ext["date"].max())
    print("BAMLH0A0HYM2 available until:", ext.loc[ext["BAMLH0A0HYM2"].notna(), "date"].max())

    return ext

def make_final_market_state(market_feat, external_feat):
    final = market_feat.merge(external_feat, on="date", how="outer")
    final = final.sort_values("date")
    final = final[(final["date"] >= START) & (final["date"] <= END)].copy()

    # 对宏观和市场特征做简单 forward fill，避免假日/不同市场日期不一致
    final = final.ffill()

    out_file = OUT / "market_state_features.csv"
    final.to_csv(out_file, index=False)

    print("\n[OK] market_state_features.csv")
    print("shape:", final.shape)
    print("date:", final["date"].min(), "to", final["date"].max())
    print("\nMissing values by column, top 20:")
    print(final.isna().sum().sort_values(ascending=False).head(20))

    return final

def check_existing_clean_tickers():
    files = sorted(TICKERS_DIR.glob("*.csv"))
    print("\n[OK] existing clean ticker files:", len(files))
    expected = {
        "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY","WFC","GS","BLK",
        "PFE","HUM","FDX","GD","IBM","TER","ECL","IP","DTE","WEC",
        "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU"
    }
    found = {p.stem for p in files}
    print("missing:", sorted(expected - found))
    print("extra:", sorted(found - expected))

def main():
    check_existing_clean_tickers()
    market_df = load_market_etfs()
    market_feat = make_market_features(market_df)
    external_feat = make_external_features(market_feat)
    make_final_market_state(market_feat, external_feat)

    print("\nDONE.")
    print("Generated files:")
    print(OUT / "market_etfs_clean.csv")
    print(OUT / "market_etf_features.csv")
    print(OUT / "external_macro_features.csv")
    print(OUT / "market_state_features.csv")

if __name__ == "__main__":
    main()
