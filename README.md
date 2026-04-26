# RAM-FinGAN / Regime-Factor Forecasting Project

This project extends the original Fin-GAN stock--ETF excess-return forecasting setting by adding market-regime information. The main empirical finding is that compressed market-regime representations improve out-of-sample robustness more reliably than direct market-feature concatenation or unstable adversarial GAN training.

## 1. Research Goal

The original Fin-GAN framework models stock--ETF excess returns using historical return windows:

\[
p(y_{t+1} \mid C_t)
\]

where \(C_t\) is the past return window and \(y_{t+1}\) is the next excess return.

This project extends the conditioning information to include market-regime state:

\[
p(y_{t+1} \mid C_t, R_t)
\]

where \(R_t\) is a compressed regime representation extracted from market ETFs and macro/financial stress variables.

## 2. Data

### 2.1 CRSP data

The project uses daily CRSP data for:

- 22 stocks:
  - AMZN, HD, NKE
  - CL, EL, KO, PEP
  - APA, OXY
  - WFC, GS, BLK
  - PFE, HUM
  - FDX, GD
  - IBM, TER
  - ECL, IP
  - DTE, WEC

- 9 sector ETFs:
  - XLY, XLP, XLE, XLF, XLV, XLI, XLK, XLB, XLU

- Market-state ETFs:
  - SPY, QQQ, IWM, DIA, TLT, HYG, LQD

### 2.2 External market-state data

The project also uses FRED-style external variables:

- VIXCLS
- DGS10
- DGS2
- DGS3MO
- BAMLH0A0HYM2

The credit-spread variable is supplemented by the HYG-LQD relative return proxy when needed.

## 3. Directory Structure

```text
/home/kwang/RAM-FinGAN/
├── configs/
│   └── default.yaml
├── data_raw/
│   ├── crsp/
│   │   ├── Stocks-data.csv
│   │   ├── ETFs-data.csv
│   │   └── Market-ETFs-data.csv
│   └── external/
│       ├── VIXCLS.csv
│       ├── DGS10.csv
│       ├── DGS2.csv
│       ├── DGS3MO.csv
│       └── BAMLH0A0HYM2.csv
├── data_clean/
│   ├── tickers/
│   ├── ram_features/
│   └── ram_panel/
├── outputs/
│   ├── logs/
│   ├── models/
│   ├── plots/
│   └── results/
└── src/