

we have started with naive implementation - take input data:
each training sample (line) is one day of one coin. The input is 9 features computed only from the past up to that day (recent returns, 7-day return, 7-day and 30-day volatility, RSI, ATR and volume z-score), 
and the target is what happens in the next 7 days: whether the return is positive (trend), and how large the daily volatility is (volatility). 
All coins are stacked in one table, and the models are trained on the first 70% of the dates and tested on the later 30%.
can a model look on the past ~4 yeaers and try to predict the regime changes during the next year ? (each time it will based on the data from last month and predict the next week)


what we have seen:



lets try to improve the results 
1. we will try to predict only the next day / 2 days etc,  and see the degradation of the results. 
2. we will run only on subset of the data, e.g. only BTC, or only coins with highest correlation with BTC, and see if we can get better results.
2. train using - Time Series Cross-Validation
3. increase data resolution and try to predict the intraday regime changes (1h, 4h, 12h) and see if we can get better results.
4. use more complex models (LSTM, GRU, Transformers) and see if they can capture the temporal dependencies better than simpler models.


simulation_a
I tested whether simple ML models can predict the behavior of 16 crypto coins from daily data. Each sample is one coin on one day, and the models (Logistic Regression and Ridge, Random Forest, XGBoost) use features from the past only to predict the trend and the volatility of the next 7 days. They are trained on the earlier part of the data and tested on a later period, and I repeat the same tests on synthetic noise and on noise with a simulated trend. The trend is almost unpredictable: the AUC is 0.52-0.56, close to random guessing (0.50), while the models do find a simulated trend, so the code works. Adding RSI, ATR and volume did not give a consistent improvement. The models beat the 7-day volatility baseline by about 20%, but they also do so on pure noise, so this is probably not a real signal, and stronger baselines are needed to check it. The market regimes found by k-means last about 22 days, but part of this is expected by construction, because the features are rolling windows. Overall, the results show that there is very little learnable signal in the data, which is an acceptable negative result.
