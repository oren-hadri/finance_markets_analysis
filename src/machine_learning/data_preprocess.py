from pathlib import Path
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor  # pip install xgboost + brew install libomp


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "assets_list.ini"
DEFAULT_DATA_DIR = PROJECT_DIR / "data/raw"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/normalized"

input_dir = DEFAULT_OUTPUT_DIR
output_dir = PROJECT_DIR / "data/ml"

HORIZON = 14          # how many days ahead we predict
TRAIN_SHARE = 0.7    # first 70% of the dates are for training
MIN_DAYS = 500       # drop coins with a shortr data history

RETURN_FEATURES = ["ret_1", "ret_2", "ret_3", "ret_7", "volatility_7d", "volatility_30d"]
ALL_FEATURES = RETURN_FEATURES + ["rsi", "atr", "volume_z"]


def load_table(name):
    return pd.read_csv(input_dir / name, index_col=0, parse_dates=True)

def load_crypto():
    # lets laod the data fro the preprocessed csv files
    returns = load_table("merged_returns.csv")
    # Relative Strength Index - momentum indicator - how strong the price rises compared to falls. For each day, compute the prce change. Take average gain and the average loss over 14 days.
    # Compute RS = average gain / average loss.
    # Compute RSI = 100 − 100 / (1 + RS).
    # Above 70: price rose strongly and quickly. “overbought”.
    # Below 30: price fell strongly and quickly. “oversold”.
    # Around 50: gains and losses are balanced.
    rsi = load_table("merged_rsi.csv")
    # normalized average true range - volatility indicator (avarage of 14 days)
    # normalize by price level so ATR is comparable across assets of very
    # different price scales (e.g. BTC at $60k vs a $5 altcoin)
    atr = load_table("merged_atr.csv")
    # volume z = (volume today − mean of last 20 days) / std of last 20 days
    # is todays volume unusually high or low for this coin, compared with its past
    volume = load_table("merged_volume.csv")

    # only crypto coins that exist in all the tables
    coins = [c for c in returns.columns if c.startswith("crypto_")]
    coins = [c for c in coins if c in rsi.columns and c in atr.columns and c in volume.columns]
    print(len(coins), "coins:", coins)

    returns = returns[coins].replace([np.inf, -np.inf], np.nan) # sanity - protect if price was zero and cause return to be infinit
    extra = {"rsi": rsi[coins], "atr": atr[coins], "volume_z": volume[coins]}
    return returns, extra



# Build faetures and targets
def to_long(wide, name):
    # one row = one day and one coin
    wide = wide.rename_axis("date")
    return wide.reset_index().melt(id_vars="date", var_name="coin", value_name=name)


def make_dataset(returns, extra=None):
    # features: only data up to today
    features = {
        "ret_1": returns, # ret_today 
        "ret_2": returns.shift(1), # ret_1d_ago
        "ret_3": returns.shift(2),
        "ret_7": returns.rolling(7).sum(),
        "volatility_7d": returns.rolling(7).std(),
        "volatility_30d": returns.rolling(30).std(),
    }
    if extra:
        features.update(extra)

    # targets: the next 7 days (shift(-7) brings the future to today)
    future_ret = returns.rolling(HORIZON).sum().shift(-HORIZON)
    future_vol = returns.rolling(HORIZON).std().shift(-HORIZON)
    targets = {
        "target_trend": (future_ret > 0).astype(float).where(future_ret.notna()),
        "target_volatility": future_vol,
    }

    # we wish to creat data such that - each training sample (line) is one day of one coin. The input is 9 features computed only from the past up to that day (recent returns, 7-day return, 7-day and 30-day volatility, RSI, ATR and volume z-score), and the target is what happens in the next 7 days: whether the return is positive (trend), and how large the daily volatility is (volatility). All coins are stacked in one table, and the models are trained on the first 70% of the dates and tested on the later 30%.
    parts = [to_long(df, name) for name, df in {**features, **targets}.items()]
    data = reduce(lambda a, b: a.merge(b, on=["date", "coin"]), parts)
    data = data.dropna()

    # remove coins with a short history
    days = data.groupby("coin").size()
    data = data[data["coin"].isin(days[days >= MIN_DAYS].index)]
    return data.sort_values(["date", "coin"]).reset_index(drop=True)



# 3. Split by time (with a gap, because the targets overlap)

def split_by_time(data):
    dates = np.sort(data["date"].unique())
    num_of_days = len(dates)
    cut = dates[int(num_of_days * TRAIN_SHARE)]
    gap_start = cut - pd.Timedelta(days=HORIZON)
    train = data[data["date"] < gap_start]
    test = data[data["date"] >= cut]
    return train, test



# 4. Models

def run_trend(train, test, cols):
    scaler = StandardScaler().fit(train[cols])  # fit on train only
    # normalize features to have mean=0 and std=1, so that the models are not affected by the different scales of the features
    X_train, X_test = scaler.transform(train[cols]), scaler.transform(test[cols])
    # the target is binary: whether the return in the next 7 days is positive or negative - so no need to normalize it
    y_train, y_test = train["target_trend"], test["target_trend"]

    models = {
        "baseline (most common)": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=0),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "auc": roc_auc_score(y_test, prob),
        })
    print("\nTrend (up or down in the next %d days). Share of up days in test: %.3f"
          % (HORIZON, y_test.mean()))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    return pd.DataFrame(rows)


def run_volatility(train, test, cols):
    scaler = StandardScaler().fit(train[cols])
    X_train, X_test = scaler.transform(train[cols]), scaler.transform(test[cols])
    y_train, y_test = train["target_volatility"], test["target_volatility"]

    # baseline: tomorrow looks like the last 7 days
    rows = [{
        "model": "baseline (last volatility_7d)",
        "mae": mean_absolute_error(y_test, test["volatility_7d"]),
        "rmse": mean_squared_error(y_test, test["volatility_7d"]) ** 0.5,
    }]
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=0),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
        })
    print("\nVolatility (std of daily returns in the next %d days)" % HORIZON)
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    return pd.DataFrame(rows)


def run_all(data, cols, title):
    train, test = split_by_time(data)
    print(title)
    print("train rows: %d, test rows: %d" % (len(train), len(test)))
    trend = run_trend(train, test, cols)
    vol = run_volatility(train, test, cols)
    return trend, vol


# 5. Regimes (calm or turbulent), found by clustering
def find_regimes(data):
    train, test = split_by_time(data)
    cols = ["volatility_7d", "volatility_30d", "atr"]  # the first column is used to name the clusters

    scaler = StandardScaler().fit(train[cols])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(scaler.transform(train[cols]))

    test = test.sort_values(["coin", "date"]).copy()
    test["cluster"] = kmeans.predict(scaler.transform(test[cols]))

    # the cluster with the higher volatility is "turbulent"
    turbulent = kmeans.cluster_centers_[:, 0].argmax()
    test["turbulent"] = (test["cluster"] == turbulent).astype(int)

    # a simple rule to compare with: volatility above the training median
    rule = (test["volatility_7d"] > train["volatility_7d"].median()).astype(int)
    agreement = (test["turbulent"] == rule).mean()

    # how many days in a row does a regime last (for each coin)
    lengths = []
    for coin, g in test.groupby("coin"):
        s = g["turbulent"].values
        changes = np.flatnonzero(np.diff(s))
        lengths += np.diff(np.r_[-1, changes, len(s) - 1]).tolist()

    print("Regimes (KMeans, 2 clusters)")
    print("Share of turbulent days in test: %.3f" % test["turbulent"].mean())
    print("Average regime length: %.1f days" % np.mean(lengths))
    print("Agreement with the simple volatility rule: %.3f" % agreement)



# Synthetc data: pure noise, and noise with a trend
def make_synthetic(returns, trend_strength=0.0, seed=0):
    # Random returns with the same volatility as each real coin.
    # trend_strength = 0 means pure noise.
    # Otherwise a slow wave (period of 120 days) is added, so there is
    # a real trend that the models should be able to find.
    rng = np.random.default_rng(seed)
    t = np.arange(len(returns))
    fake = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for coin in returns.columns:
        std = returns[coin].std()
        noise = rng.normal(0, std, len(returns))
        phase = rng.uniform(0, 2 * np.pi)
        trend = trend_strength * std * np.sin(2 * np.pi * t / 120 + phase)
        fake[coin] = noise + trend
    return fake


import matplotlib.pyplot as plt

def plot_auc(results, path):
    # results = {"Real": trend_df, "Noise": ..., "Noise + trend": ...}
    models = ["logistic", "random_forest", "xgboost"]
    width = 0.25
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (name, df) in enumerate(results.items()):
        values = df.set_index("model").loc[models, "auc"]
        ax.bar(x + i * width, values, width, label=name)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)  # random guessing
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=7)
    ax.set_ylim(0.45, 0.65)
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)

# Run everything
if __name__ == "__main__":
    returns, extra = load_crypto()

    # real data
    data = make_dataset(returns, extra)
    data.to_csv(output_dir / "ml_dataset_daily.csv", index=False)
    print("Dataset:", data.shape, data["date"].min(), "to", data["date"].max())
    print(data.groupby("coin").size())

    trend_real, _  = run_all(data, ALL_FEATURES, "REAL data, all features")
    run_all(data, RETURN_FEATURES, "REAL data, return features only")
    find_regimes(data)

    # synthetic data, built only from returns
    noise = make_dataset(make_synthetic(returns, trend_strength=0.0))
    trend_noise, _ = run_all(noise, RETURN_FEATURES, "SYNTHETIC: pure noise (no signal)")

    trend = make_dataset(make_synthetic(returns, trend_strength=0.3))
    trend_sim, _  = run_all(trend, RETURN_FEATURES, "SYNTHETIC: noise + simulated trend")

    # plot AUC comparison
    plot_auc({"Real": trend_real, "Noise": trend_noise, "Noise + trend": trend_sim},
             output_dir / "auc_comparison.png")