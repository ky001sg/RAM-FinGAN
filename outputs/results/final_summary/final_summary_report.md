# Final RAM-FinGAN / Regime-Factor Experiment Summary

## 1. Overall model ranking

| model               |   direction_acc |   mean_pnl_bp |   daily_sharpe |      rmse |        mae |
|:--------------------|----------------:|--------------:|---------------:|----------:|-----------:|
| regime_factor_lstm  |        0.516219 |      2.8808   |       0.550735 | 0.0119862 | 0.00696949 |
| ram_fingan_v2       |        0.511785 |      2.26487  |       0.454076 | 0.012081  | 0.0070667  |
| ram_fingan_v3       |        0.512647 |      1.79847  |       0.373847 | 0.0121353 | 0.00715108 |
| baseline_lstm_xonly |        0.511867 |      1.56566  |       0.302691 | 0.0119943 | 0.00696997 |
| regime_econ_lstm_v4 |        0.512031 |      0.45149  |       0.29094  | 0.0121415 | 0.007192   |
| ram_fingan_v1       |        0.51392  |      0.853635 |       0.238147 | 0.0170617 | 0.0111999  |
| naive_ram_lstm      |        0.510142 |      1.20089  |       0.230985 | 0.0120515 | 0.00701428 |

## 2. Transaction-cost robustness: key models

|   cost_bp | model               |   mean_net_pnl_bp |   daily_sharpe_net |   mean_turnover |
|----------:|:--------------------|------------------:|-------------------:|----------------:|
|         0 | regime_factor_lstm  |         5.75641   |         0.550735   |        1.72071  |
|         0 | ram_fingan_v2       |         4.52565   |         0.454076   |        2.05489  |
|         0 | ram_fingan_v3       |         3.59369   |         0.373847   |        1.7826   |
|         0 | baseline_lstm_xonly |         3.12849   |         0.302691   |        1.89219  |
|         0 | regime_econ_lstm_v4 |         0.902165  |         0.29094    |        0.435658 |
|         0 | naive_ram_lstm      |         2.39961   |         0.230985   |        1.73958  |
|         1 | regime_factor_lstm  |         4.0357    |         0.386631   |        1.72071  |
|         1 | ram_fingan_v2       |         2.47076   |         0.248204   |        2.05489  |
|         1 | ram_fingan_v3       |         1.81109   |         0.188694   |        1.7826   |
|         1 | regime_econ_lstm_v4 |         0.466507  |         0.1506     |        0.435658 |
|         1 | baseline_lstm_xonly |         1.2363    |         0.119771   |        1.89219  |
|         1 | naive_ram_lstm      |         0.660028  |         0.0636186  |        1.73958  |
|         2 | regime_factor_lstm  |         2.31499   |         0.222063   |        1.72071  |
|         2 | ram_fingan_v2       |         0.415866  |         0.0418249  |        2.05489  |
|         2 | regime_econ_lstm_v4 |         0.0308485 |         0.00996858 |        0.435658 |
|         2 | ram_fingan_v3       |         0.0284875 |         0.00297245 |        1.7826   |
|         2 | baseline_lstm_xonly |        -0.655891  |        -0.0636188  |        1.89219  |
|         2 | naive_ram_lstm      |        -1.07955   |        -0.104186   |        1.73958  |
|         5 | regime_factor_lstm  |        -2.84714   |        -0.274008   |        1.72071  |
|         5 | regime_econ_lstm_v4 |        -1.27613   |        -0.413517   |        0.435658 |
|         5 | ram_fingan_v3       |        -5.31932   |        -0.557296   |        1.7826   |
|         5 | ram_fingan_v2       |        -5.74881   |        -0.57995    |        2.05489  |
|         5 | naive_ram_lstm      |        -6.29829   |        -0.609829   |        1.73958  |
|         5 | baseline_lstm_xonly |        -6.33246   |        -0.616127   |        1.89219  |
|        10 | regime_factor_lstm  |       -11.4507    |        -1.10611    |        1.72071  |
|        10 | regime_econ_lstm_v4 |        -3.45442   |        -1.12369    |        0.435658 |
|        10 | naive_ram_lstm      |       -14.9962    |        -1.45752    |        1.73958  |
|        10 | ram_fingan_v3       |       -14.2323    |        -1.49948    |        1.7826   |
|        10 | baseline_lstm_xonly |       -15.7934    |        -1.54179    |        1.89219  |
|        10 | ram_fingan_v2       |       -16.0233    |        -1.6225     |        2.05489  |

## 3. Raw vs smoothed regime-factor deployment

|   cost_bp |   raw_mean_net_pnl_bp |   raw_net_sharpe |   raw_mean_turnover |   threshold |   alpha |   k |   smooth_mean_net_pnl_bp |   smooth_net_sharpe |   smooth_mean_turnover |   avg_abs_position |   sharpe_improvement_smooth_minus_raw |   turnover_reduction_pct |
|----------:|----------------------:|-----------------:|--------------------:|------------:|--------:|----:|-------------------------:|--------------------:|-----------------------:|-------------------:|--------------------------------------:|-------------------------:|
|         0 |               5.75641 |         0.550735 |             1.72071 |       0     |    1    | 500 |                2.0217    |            0.389916 |             0.58008    |         0.233459   |                           -0.160819   |                  66.2883 |
|         1 |               4.0357  |         0.386631 |             1.72071 |       0     |    0.75 | 500 |                1.2302    |            0.279292 |             0.389591   |         0.193378   |                           -0.107339   |                  77.3587 |
|         2 |               2.31499 |         0.222063 |             1.72071 |       0     |    0.2  | 500 |                0.622011  |            0.231351 |             0.0887087  |         0.13391    |                            0.00928806 |                  94.8446 |
|         5 |              -2.84714 |        -0.274008 |             1.72071 |       0     |    0.1  | 500 |                0.325333  |            0.149408 |             0.0436843  |         0.124236   |                            0.423416   |                  97.4613 |
|        10 |             -11.4507  |        -1.10611  |             1.72071 |       0.003 |    0.2  |  50 |                0.0710038 |            0.108297 |             0.00330458 |         0.00496984 |                            1.21441    |                  99.808  |

## 4. Sharpe by period

| period         |   baseline_lstm_xonly |   naive_ram_lstm |   ram_fingan_v1 |   ram_fingan_v2 |   ram_fingan_v3 |   regime_econ_lstm_v4 |   regime_factor_lstm |
|:---------------|----------------------:|-----------------:|----------------:|----------------:|----------------:|----------------------:|---------------------:|
| covid_shock    |              0.171921 |        -0.12572  |      0.00497591 |        0.580163 |        0.242386 |              0.259914 |             0.796581 |
| post_covid     |              0.32884  |         0.320877 |      0.383213   |        0.367343 |        0.43549  |              0.374535 |             0.399311 |
| pre_covid_test |              0.640907 |         0.760487 |      0.426385   |        0.809464 |        0.621483 |              0.715892 |             0.999883 |

## 5. Sharpe by regime cluster

|   regime_cluster |   baseline_lstm_xonly |   naive_ram_lstm |   ram_fingan_v1 |   ram_fingan_v2 |   ram_fingan_v3 |   regime_econ_lstm_v4 |   regime_factor_lstm |
|-----------------:|----------------------:|-----------------:|----------------:|----------------:|----------------:|----------------------:|---------------------:|
|                1 |              0.474162 |        -0.13085  |        0.302336 |        1.02019  |        0.538135 |              0.799298 |             1.43337  |
|                2 |              0.285065 |         0.321898 |        0.247505 |        0.362128 |        0.365049 |              0.185855 |             0.397382 |

## 6. Best model by ticker

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

## 7. Key findings for paper

| finding                                               | evidence                                                                                 |
|:------------------------------------------------------|:-----------------------------------------------------------------------------------------|
| Best predictive model without transaction cost        | regime_factor_lstm has test Sharpe 0.551, direction accuracy 51.622%, mean PnL 2.881 bp. |
| Regime-factor model is strongest during COVID shock   | During COVID shock, regime_factor_lstm Sharpe = 0.797.                                   |
| Transaction costs expose high-turnover weakness       | At 2bp cost, raw regime_factor_lstm Sharpe drops to 0.222 with turnover 1.721.           |
| Position smoothing improves deployability under costs | At 5bp cost, smoothing improves Sharpe from -0.274 to 0.149, reducing turnover by 97.5%. |

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
