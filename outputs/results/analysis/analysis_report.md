# RAM-FinGAN Model Analysis Report

## Overall test ranking

| model               |   direction_acc |   mean_pnl_bp |   daily_sharpe |      rmse |        mae |
|:--------------------|----------------:|--------------:|---------------:|----------:|-----------:|
| regime_factor_lstm  |        0.516219 |      2.8808   |       0.550735 | 0.0119862 | 0.00696949 |
| ram_fingan_v2       |        0.511785 |      2.26487  |       0.454076 | 0.012081  | 0.0070667  |
| ram_fingan_v3       |        0.512647 |      1.79847  |       0.373847 | 0.0121353 | 0.00715108 |
| baseline_lstm_xonly |        0.511867 |      1.56566  |       0.302691 | 0.0119943 | 0.00696997 |
| regime_econ_lstm_v4 |        0.512031 |      0.45149  |       0.29094  | 0.0121415 | 0.007192   |
| ram_fingan_v1       |        0.51392  |      0.853635 |       0.238147 | 0.0170617 | 0.0111999  |
| naive_ram_lstm      |        0.510142 |      1.20089  |       0.230985 | 0.0120515 | 0.00701428 |

## Sharpe by regime

|   regime_cluster |   baseline_lstm_xonly |   naive_ram_lstm |   ram_fingan_v1 |   ram_fingan_v2 |   ram_fingan_v3 |   regime_econ_lstm_v4 |   regime_factor_lstm |
|-----------------:|----------------------:|-----------------:|----------------:|----------------:|----------------:|----------------------:|---------------------:|
|                1 |              0.474162 |        -0.13085  |        0.302336 |        1.02019  |        0.538135 |              0.799298 |             1.43337  |
|                2 |              0.285065 |         0.321898 |        0.247505 |        0.362128 |        0.365049 |              0.185855 |             0.397382 |

## Sharpe by period

| period         |   baseline_lstm_xonly |   naive_ram_lstm |   ram_fingan_v1 |   ram_fingan_v2 |   ram_fingan_v3 |   regime_econ_lstm_v4 |   regime_factor_lstm |
|:---------------|----------------------:|-----------------:|----------------:|----------------:|----------------:|----------------------:|---------------------:|
| covid_shock    |              0.171921 |        -0.12572  |      0.00497591 |        0.580163 |        0.242386 |              0.259914 |             0.796581 |
| post_covid     |              0.32884  |         0.320877 |      0.383213   |        0.367343 |        0.43549  |              0.374535 |             0.399311 |
| pre_covid_test |              0.640907 |         0.760487 |      0.426385   |        0.809464 |        0.621483 |              0.715892 |             0.999883 |

## Best model by ticker

| ticker   | model               |   daily_sharpe |   direction_acc |   mean_pnl_bp |
|:---------|:--------------------|---------------:|----------------:|--------------:|
| AMZN     | ram_fingan_v2       |     -0.445613  |        0.485095 |     -1.90665  |
| APA      | ram_fingan_v2       |      1.08283   |        0.528455 |     11.5699   |
| BLK      | naive_ram_lstm      |      0.0553678 |        0.505872 |      0.257839 |
| CL       | ram_fingan_v3       |      2.40111   |        0.550136 |      6.47031  |
| DTE      | baseline_lstm_xonly |      1.34618   |        0.540199 |      5.79011  |
| ECL      | ram_fingan_v1       |      1.33082   |        0.527552 |      3.74354  |
| EL       | regime_factor_lstm  |      0.940044  |        0.517615 |      5.08808  |
| FDX      | regime_factor_lstm  |      1.10306   |        0.496838 |      6.21317  |
| GD       | ram_fingan_v1       |      1.57797   |        0.526649 |      2.5415   |
| GS       | regime_econ_lstm_v4 |      1.4628    |        0.531165 |      0.809938 |
| HD       | ram_fingan_v1       |      1.03134   |        0.493225 |      2.23919  |
| HUM      | regime_econ_lstm_v4 |      1.3653    |        0.523035 |      1.58017  |
| IBM      | ram_fingan_v1       |      0.130974  |        0.504968 |      0.465607 |
| IP       | ram_fingan_v1       |      1.29179   |        0.522132 |      3.72352  |
| KO       | regime_factor_lstm  |      2.50568   |        0.557362 |      6.79827  |
| NKE      | naive_ram_lstm      |      0.603059  |        0.504968 |      2.75602  |
| OXY      | baseline_lstm_xonly |      0.468821  |        0.516712 |      4.24792  |
| PEP      | ram_fingan_v3       |      3.29779   |        0.558266 |      6.88866  |
| PFE      | regime_econ_lstm_v4 |      1.35805   |        0.493225 |      1.99601  |
| TER      | regime_factor_lstm  |      0.0424356 |        0.527552 |      0.271419 |
| WEC      | baseline_lstm_xonly |      1.99699   |        0.538392 |      6.78045  |
| WFC      | regime_factor_lstm  |      0.591884  |        0.519422 |      2.69327  |

## Main interpretation

Current evidence should be read as follows:

1. Directly concatenating raw market features is not robust.
2. Compressing market information into regime factors gives the strongest out-of-sample result so far.
3. GAN-style adversarial training is unstable in this setting.
4. Economic-loss training can overfit validation-period trading objectives.
5. The most defensible current paper direction is regime-factor representation for robust stock--ETF excess-return forecasting, with GAN variants reported as ablation/negative evidence rather than the main claim.
