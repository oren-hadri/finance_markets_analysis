Experiments So Far

daily data of 16 crypto coins. Each sample is one coin on one day, with 9 features computed from the past only (recent returns, 7-day return, 7- and 30-day volatility, RSI, ATR, volume). 
The targets are the trend (will the return be positive) 
and the volatility of the next 7 days. 

The data is split by time (first 70% for training, with a 7-day gap), and I compare Logistic Regression / Ridge, Random Forest and XGBoost against simple baselines. I also run the same code on synthetic data: pure noise, and noise with a simulated trend (sine wave).

Results.

Trend: the AUC (ability distinguish between positive and negative classe) is 0.52--0.56, close to random guessing (0.50). On pure noise it is 0.49--0.50, and on noise with a trend it is 0.56--0.62. So the code works, and the real market contains almost no learnable trend.
Volatility: XGBoost improves the 7-day baseline by 20%, but on pure noise it improves it by 24%. Predicting 3 days ahead instead of 7 days did not change the picture.
Features: RSI, ATR and volume do not help.
Regimes: k-means finds regimes that last about 22 days, and they agree 84% with a simple volatility threshold.

Explanation. Price direction is almost unpredictable. The volatility improvement is probably not real signal, because the 7-day baseline is noisy and the models simply average over a longer window. Also, the AUC of 0.55 may not be significant, because the targets overlap and the coins move together.

Next steps.

Add stronger volatility baselines (30-day volatility, coin average).
Run the regimes on pure noise, and test the significance of the AUC.
Report several prediction horizons (1, 3, 7, 14 days) without choosing the best one on the test set.
Add an MLP, an SVM, and the Transformer on sequences of 30 days.



"/Users/orenhadri/Documents/לימודים תואר שני/project_finance_analysis/finance_markets_analysis/.venv/bin/python" /Users/orenhadri/Documents/לימודים תואר שני/project_finance_analysis/finance_markets_analysis/src/machine_learning/data_preprocess.py
16 coins: ['crypto__BTC', 'crypto__ETH', 'crypto__SOL', 'crypto__BNB', 'crypto__XRP', 'crypto__ADA', 'crypto__DOGE', 'crypto__DOT', 'crypto__AVAX', 'crypto__POL28321-USD', 'crypto__LTC', 'crypto__LINK', 'crypto__ATOM', 'crypto__UNI7083-USD', 'crypto__USDT', 'crypto__USDC']
Dataset: (22437, 13) 2022-10-30 00:00:00 to 2026-09-25 00:00:00
coin
crypto__ADA             1427
crypto__ATOM            1427
crypto__AVAX            1427
crypto__BNB             1427
crypto__BTC             1427
crypto__DOGE            1427
crypto__DOT             1427
crypto__ETH             1427
crypto__LINK            1427
crypto__LTC             1427
crypto__POL28321-USD    1032
crypto__SOL             1427
crypto__UNI7083-USD     1427
crypto__USDC            1427
crypto__USDT            1427
crypto__XRP             1427
dtype: int64
REAL data, all features
train rows: 15525, test rows: 6864

Trend (up or down in the next 3 days). Share of up days in test: 0.463
                 model  accuracy     f1    auc
baseline (most common)    0.4634 0.6333 0.5000
              logistic    0.5001 0.5384 0.5148
         random_forest    0.5169 0.5200 0.5385
               xgboost    0.5319 0.5499 0.5535

Volatility (std of daily returns in the next 3 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01633 0.02577
                        ridge 0.01436 0.02178
                random_forest 0.01461 0.02242
                      xgboost 0.01385 0.02144
REAL data, return features only
train rows: 15525, test rows: 6864

Trend (up or down in the next 3 days). Share of up days in test: 0.463
                 model  accuracy     f1    auc
baseline (most common)    0.4634 0.6333 0.5000
              logistic    0.5036 0.5731 0.5281
         random_forest    0.5224 0.5203 0.5406
               xgboost    0.5328 0.5669 0.5624

Volatility (std of daily returns in the next 3 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01633 0.02577
                        ridge 0.01468 0.02192
                random_forest 0.01444 0.02225
                      xgboost 0.01389 0.02146

=================================================================
Regimes (KMeans, 2 clusters)
=================================================================
Share of turbulent days in test: 0.443
Average regime length: 21.9 days
Agreement with the simple volatility rule: 0.841
SYNTHETIC: pure noise (no signal)
train rows: 15936, test rows: 6864

Trend (up or down in the next 3 days). Share of up days in test: 0.503
                 model  accuracy     f1    auc
baseline (most common)    0.5028 0.6691 0.5000
              logistic    0.4968 0.5200 0.4903
         random_forest    0.5036 0.5100 0.4994
               xgboost    0.4863 0.5123 0.4851

Volatility (std of daily returns in the next 3 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01492 0.02001
                        ridge 0.01338 0.01756
                random_forest 0.01337 0.01784
                      xgboost 0.01307 0.01744
SYNTHETIC: noise + simulated trend
train rows: 15936, test rows: 6864

Trend (up or down in the next 3 days). Share of up days in test: 0.507
                 model  accuracy     f1    auc
baseline (most common)    0.5066 0.6725 0.5000
              logistic    0.5575 0.5421 0.5766
         random_forest    0.5439 0.5570 0.5555
               xgboost    0.5599 0.5795 0.5726

Volatility (std of daily returns in the next 3 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01494 0.02002
                        ridge 0.01340 0.01758
                random_forest 0.01337 0.01787
                      xgboost 0.01309 0.01745

Process finished with exit code 0



using horizon of 7 days

"/Users/orenhadri/Documents/לימודים תואר שני/project_finance_analysis/finance_markets_analysis/.venv/bin/python" /Users/orenhadri/Documents/לימודים תואר שני/project_finance_analysis/finance_markets_analysis/src/machine_learning/data_preprocess.py 
16 coins: ['crypto__BTC', 'crypto__ETH', 'crypto__SOL', 'crypto__BNB', 'crypto__XRP', 'crypto__ADA', 'crypto__DOGE', 'crypto__DOT', 'crypto__AVAX', 'crypto__POL28321-USD', 'crypto__LTC', 'crypto__LINK', 'crypto__ATOM', 'crypto__UNI7083-USD', 'crypto__USDT', 'crypto__USDC']
Dataset: (22373, 13) 2022-10-30 00:00:00 to 2026-09-21 00:00:00
coin
crypto__ADA             1423
crypto__ATOM            1423
crypto__AVAX            1423
crypto__BNB             1423
crypto__BTC             1423
crypto__DOGE            1423
crypto__DOT             1423
crypto__ETH             1423
crypto__LINK            1423
crypto__LTC             1423
crypto__POL28321-USD    1028
crypto__SOL             1423
crypto__UNI7083-USD     1423
crypto__USDC            1423
crypto__USDT            1423
crypto__XRP             1423
dtype: int64
REAL data, all features
train rows: 15429, test rows: 6832

Trend (up or down in the next 7 days). Share of up days in test: 0.445
                 model  accuracy     f1    auc
baseline (most common)    0.5546 0.0000 0.5000
              logistic    0.5085 0.4643 0.5228
         random_forest    0.5190 0.4992 0.5429
               xgboost    0.5383 0.4906 0.5540

Volatility (std of daily returns in the next 7 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01418 0.02219
                        ridge 0.01187 0.01743
                random_forest 0.01218 0.01837
                      xgboost 0.01132 0.01703
REAL data, return features only
train rows: 15429, test rows: 6832

Trend (up or down in the next 7 days). Share of up days in test: 0.445
                 model  accuracy     f1    auc
baseline (most common)    0.5546 0.0000 0.5000
              logistic    0.5347 0.4977 0.5445
         random_forest    0.5303 0.4955 0.5410
               xgboost    0.5356 0.4873 0.5490

Volatility (std of daily returns in the next 7 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01418 0.02219
                        ridge 0.01215 0.01756
                random_forest 0.01181 0.01771
                      xgboost 0.01121 0.01692
Regimes (KMeans, 2 clusters)
Share of turbulent days in test: 0.440
Average regime length: 21.7 days
Agreement with the simple volatility rule: 0.840
SYNTHETIC: pure noise (no signal)
train rows: 15824, test rows: 6848

Trend (up or down in the next 7 days). Share of up days in test: 0.505
                 model  accuracy     f1    auc
baseline (most common)    0.5050 0.6711 0.5000
              logistic    0.4990 0.5404 0.4934
         random_forest    0.5018 0.5098 0.4962
               xgboost    0.4953 0.5326 0.4952

Volatility (std of daily returns in the next 7 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01091 0.01478
                        ridge 0.00877 0.01142
                random_forest 0.00853 0.01143
                      xgboost 0.00827 0.01111
SYNTHETIC: noise + simulated trend
train rows: 15824, test rows: 6848

Trend (up or down in the next 7 days). Share of up days in test: 0.512
                 model  accuracy     f1    auc
baseline (most common)    0.5124 0.6776 0.5000
              logistic    0.5818 0.5843 0.6123
         random_forest    0.5699 0.5902 0.5920
               xgboost    0.5904 0.6090 0.6173

Volatility (std of daily returns in the next 7 days)
                        model     mae    rmse
baseline (last volatility_7d) 0.01093 0.01480
                        ridge 0.00876 0.01142
                random_forest 0.00850 0.01142
                      xgboost 0.00826 0.01112

Process finished with exit code 0



lets try to use horizon of 14 days

