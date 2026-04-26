# Robustness: Transaction Costs and Bootstrap Tests

## Best model by transaction cost

|   cost_bp | model              |   mean_net_pnl_bp |   daily_sharpe_net |   mean_turnover |
|----------:|:-------------------|------------------:|-------------------:|----------------:|
|         0 | regime_factor_lstm |           5.75641 |           0.550735 |        1.72071  |
|         1 | regime_factor_lstm |           4.0357  |           0.386631 |        1.72071  |
|         2 | regime_factor_lstm |           2.31499 |           0.222063 |        1.72071  |
|         5 | regime_factor_lstm |          -2.84714 |          -0.274008 |        1.72071  |
|        10 | ram_fingan_v1      |          -7.55755 |          -1.06336  |        0.926328 |

## Full transaction cost table

|   cost_bp | model               |   mean_net_pnl_bp |   daily_sharpe_net |   mean_turnover |
|----------:|:--------------------|------------------:|-------------------:|----------------:|
|         0 | regime_factor_lstm  |         5.75641   |         0.550735   |        1.72071  |
|         0 | ram_fingan_v2       |         4.52565   |         0.454076   |        2.05489  |
|         0 | ram_fingan_v3       |         3.59369   |         0.373847   |        1.7826   |
|         0 | baseline_lstm_xonly |         3.12849   |         0.302691   |        1.89219  |
|         0 | regime_econ_lstm_v4 |         0.902165  |         0.29094    |        0.435658 |
|         0 | ram_fingan_v1       |         1.70573   |         0.238147   |        0.926328 |
|         0 | naive_ram_lstm      |         2.39961   |         0.230985   |        1.73958  |
|         1 | regime_factor_lstm  |         4.0357    |         0.386631   |        1.72071  |
|         1 | ram_fingan_v2       |         2.47076   |         0.248204   |        2.05489  |
|         1 | ram_fingan_v3       |         1.81109   |         0.188694   |        1.7826   |
|         1 | regime_econ_lstm_v4 |         0.466507  |         0.1506     |        0.435658 |
|         1 | baseline_lstm_xonly |         1.2363    |         0.119771   |        1.89219  |
|         1 | ram_fingan_v1       |         0.779401  |         0.108914   |        0.926328 |
|         1 | naive_ram_lstm      |         0.660028  |         0.0636186  |        1.73958  |
|         2 | regime_factor_lstm  |         2.31499   |         0.222063   |        1.72071  |
|         2 | ram_fingan_v2       |         0.415866  |         0.0418249  |        2.05489  |
|         2 | regime_econ_lstm_v4 |         0.0308485 |         0.00996858 |        0.435658 |
|         2 | ram_fingan_v3       |         0.0284875 |         0.00297245 |        1.7826   |
|         2 | ram_fingan_v1       |        -0.146927  |        -0.0205495  |        0.926328 |
|         2 | baseline_lstm_xonly |        -0.655891  |        -0.0636188  |        1.89219  |
|         2 | naive_ram_lstm      |        -1.07955   |        -0.104186   |        1.73958  |
|         5 | regime_factor_lstm  |        -2.84714   |        -0.274008   |        1.72071  |
|         5 | ram_fingan_v1       |        -2.92591   |        -0.410227   |        0.926328 |
|         5 | regime_econ_lstm_v4 |        -1.27613   |        -0.413517   |        0.435658 |
|         5 | ram_fingan_v3       |        -5.31932   |        -0.557296   |        1.7826   |
|         5 | ram_fingan_v2       |        -5.74881   |        -0.57995    |        2.05489  |
|         5 | naive_ram_lstm      |        -6.29829   |        -0.609829   |        1.73958  |
|         5 | baseline_lstm_xonly |        -6.33246   |        -0.616127   |        1.89219  |
|        10 | ram_fingan_v1       |        -7.55755   |        -1.06336    |        0.926328 |
|        10 | regime_factor_lstm  |       -11.4507    |        -1.10611    |        1.72071  |
|        10 | regime_econ_lstm_v4 |        -3.45442   |        -1.12369    |        0.435658 |
|        10 | naive_ram_lstm      |       -14.9962    |        -1.45752    |        1.73958  |
|        10 | ram_fingan_v3       |       -14.2323    |        -1.49948    |        1.7826   |
|        10 | baseline_lstm_xonly |       -15.7934    |        -1.54179    |        1.89219  |
|        10 | ram_fingan_v2       |       -16.0233    |        -1.6225     |        2.05489  |

## Bootstrap: regime_factor_lstm minus competitors

|   cost_bp | model_a            | model_b             |   mean_diff_net_pnl_bp |    ci95_low |   ci95_high |   p_value_two_sided |
|----------:|:-------------------|:--------------------|-----------------------:|------------:|------------:|--------------------:|
|         0 | regime_factor_lstm | baseline_lstm_xonly |                2.62792 |   0.40649   |    4.80775  |               0.019 |
|         0 | regime_factor_lstm | naive_ram_lstm      |                3.3568  |   0.854927  |    6.03656  |               0.011 |
|         0 | regime_factor_lstm | ram_fingan_v2       |                1.23075 |  -1.6679    |    4.02892  |               0.398 |
|         0 | regime_factor_lstm | ram_fingan_v3       |                2.16272 |   0.0227434 |    4.34224  |               0.047 |
|         0 | regime_factor_lstm | regime_econ_lstm_v4 |                4.85424 |   2.45612   |    7.27507  |               0     |
|         1 | regime_factor_lstm | baseline_lstm_xonly |                2.7994  |   0.567238  |    4.90641  |               0.017 |
|         1 | regime_factor_lstm | naive_ram_lstm      |                3.37567 |   0.899464  |    5.85935  |               0.005 |
|         1 | regime_factor_lstm | ram_fingan_v2       |                1.56494 |  -1.26429   |    4.24004  |               0.295 |
|         1 | regime_factor_lstm | ram_fingan_v3       |                2.22461 |   0.0524281 |    4.39705  |               0.045 |
|         1 | regime_factor_lstm | regime_econ_lstm_v4 |                3.56919 |   1.24397   |    5.92218  |               0.002 |
|         2 | regime_factor_lstm | baseline_lstm_xonly |                2.97088 |   0.807829  |    5.23795  |               0.008 |
|         2 | regime_factor_lstm | naive_ram_lstm      |                3.39454 |   0.867234  |    5.93982  |               0.009 |
|         2 | regime_factor_lstm | ram_fingan_v2       |                1.89912 |  -1.00243   |    4.68029  |               0.189 |
|         2 | regime_factor_lstm | ram_fingan_v3       |                2.2865  |   0.233373  |    4.51477  |               0.033 |
|         2 | regime_factor_lstm | regime_econ_lstm_v4 |                2.28414 |  -0.343394  |    4.68272  |               0.08  |
|         5 | regime_factor_lstm | baseline_lstm_xonly |                3.48532 |   1.28724   |    5.68256  |               0.001 |
|         5 | regime_factor_lstm | naive_ram_lstm      |                3.45115 |   0.929712  |    5.98595  |               0.006 |
|         5 | regime_factor_lstm | ram_fingan_v2       |                2.90167 |  -0.0972333 |    5.7616   |               0.059 |
|         5 | regime_factor_lstm | ram_fingan_v3       |                2.47218 |   0.24646   |    4.66629  |               0.028 |
|         5 | regime_factor_lstm | regime_econ_lstm_v4 |               -1.57101 |  -3.95867   |    0.700365 |               0.182 |
|        10 | regime_factor_lstm | baseline_lstm_xonly |                4.34272 |   2.07907   |    6.60945  |               0     |
|        10 | regime_factor_lstm | naive_ram_lstm      |                3.54551 |   0.988092  |    6.37101  |               0.007 |
|        10 | regime_factor_lstm | ram_fingan_v2       |                4.5726  |   1.79699   |    7.45512  |               0     |
|        10 | regime_factor_lstm | ram_fingan_v3       |                2.78164 |   0.566269  |    5.03296  |               0.012 |
|        10 | regime_factor_lstm | regime_econ_lstm_v4 |               -7.99627 | -10.4184    |   -5.52783  |               0     |

## Interpretation guide

- `daily_sharpe_net` is the annualized Sharpe after subtracting transaction costs.
- Cost is applied as `cost_bp * abs(position_t - position_{t-1})`.
- Positive bootstrap mean difference means `regime_factor_lstm` has higher net daily PnL than the comparison model.
- If the 95% CI is above zero, the improvement is robust under bootstrap resampling.
