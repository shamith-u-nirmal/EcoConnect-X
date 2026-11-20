"""
Waste-generation forecasting (per bin, next 7 days).

Model: gradient-boosted trees (HistGradientBoostingRegressor) on lag / calendar / weather /
land-use features. Evaluated on the last 21 days with a strict time split and compared with
two baselines a municipality would otherwise use (yesterday's value, same-weekday-last-week).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

LAGS = [1, 2, 3, 7, 14]
LAND_USES = ["market", "commercial", "residential", "new_layout", "institutional", "vacant"]


def build_features(daily: pd.DataFrame, bins: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    df = daily[["date", "bin_id", "generated_kg"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal, on="date", how="left").merge(bins[["bin_id", "land_use"]], on="bin_id")
    df = df.sort_values(["bin_id", "date"])
    g = df.groupby("bin_id").generated_kg
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["trend"] = df["roll7"] / df["roll28"]
    df["dow"] = df.date.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["festival_flag"] = (df.festival.fillna("") != "").astype(int)
    for lu in LAND_USES:
        df[f"lu_{lu}"] = (df.land_use == lu).astype(int)
    return df


FEATURES = [f"lag_{l}" for l in LAGS] + ["roll7", "roll28", "trend", "dow", "is_weekend",
                                         "festival_flag", "rain_mm"] + [f"lu_{lu}" for lu in LAND_USES]


def train_and_evaluate(daily, bins, cal, test_days: int = 21, seed: int = 0):
    df = build_features(daily, bins, cal).dropna(subset=FEATURES)
    cutoff = df.date.max() - pd.Timedelta(days=test_days)
    tr, te = df[df.date <= cutoff], df[df.date > cutoff]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=0.1, random_state=seed)
    model.fit(tr[FEATURES], tr.generated_kg)
    pred = model.predict(te[FEATURES])
    y = te.generated_kg.values
    mae = mean_absolute_error(y, pred)
    metrics = {
        "model": "HistGradientBoostingRegressor",
        "test_days": test_days,
        "mae_kg": round(float(mae), 2),
        "mae_naive_yesterday_kg": round(mean_absolute_error(y, te.lag_1), 2),
        "mae_naive_lastweek_kg": round(mean_absolute_error(y, te.lag_7), 2),
        "wape_pct": round(float(100 * np.abs(y - pred).sum() / y.sum()), 1),
    }
    metrics["improvement_vs_yesterday_pct"] = round(
        100 * (1 - metrics["mae_kg"] / metrics["mae_naive_yesterday_kg"]), 1)
    # refit on everything for production forecasts
    model.fit(df[FEATURES], df.generated_kg)
    return model, metrics, te.assign(pred=pred)


def forecast_next_days(model, daily, bins, cal, horizon: int = 7) -> pd.DataFrame:
    """Recursive multi-step forecast. Future calendar = weekday pattern, no rain, no festival."""
    hist = daily[["date", "bin_id", "generated_kg"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    last = hist.date.max()
    out = []
    for h in range(1, horizon + 1):
        day = last + pd.Timedelta(days=h)
        stub = pd.DataFrame({"date": day, "bin_id": bins.bin_id, "generated_kg": np.nan})
        cal_ext = pd.concat([cal, pd.DataFrame({"date": [day], "rain_mm": [0.0], "festival": [""]})])
        feats = build_features(pd.concat([hist, stub]), bins, cal_ext)
        today = feats[feats.date == day]
        today = today.assign(generated_kg=np.clip(model.predict(today[FEATURES]), 0, None))
        hist = pd.concat([hist, today[["date", "bin_id", "generated_kg"]]])
        out.append(today[["date", "bin_id", "generated_kg"]])
    fc = pd.concat(out).rename(columns={"generated_kg": "pred_kg"})
    fc["pred_kg"] = fc.pred_kg.round(2)
    return fc
