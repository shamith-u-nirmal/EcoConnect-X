"""
Forecast-driven collection routing.

Instead of a fixed beat, each morning the vehicle visits only bins predicted to cross the
fill threshold today (current reading + forecast), plus anything overflowing. The route is
solved with nearest-neighbour construction + 2-opt improvement over a haversine distance
matrix (OR-Tools / OSRM road distances drop in later without changing the interface).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_m


def _dist_matrix(lat, lon):
    return haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) * 1.3  # road detour factor


def _tour_len(D, tour):
    return float(sum(D[a, b] for a, b in zip(tour[:-1], tour[1:])))


def solve_tsp(lat, lon):
    D = _dist_matrix(np.asarray(lat), np.asarray(lon))
    n = len(D)
    tour, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt)
        left.remove(nxt)
    tour.append(0)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b, c, d = tour[i - 1], tour[i], tour[k], tour[k + 1]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-6:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    improved = True
    return tour, _tour_len(D, tour)


def plan_routes(bins: pd.DataFrame, daily: pd.DataFrame, forecast: pd.DataFrame, depot,
                threshold: float = 0.7, max_age_days: int = 4) -> tuple[pd.DataFrame, dict]:
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    now = d[d.date == d.date.max()].set_index("bin_id")
    fill_now = now.fill_kg_reading.where(~now.collected, 0)
    f1 = forecast[forecast.date == forecast.date.min()].set_index("bin_id").pred_kg
    b = bins.set_index("bin_id").copy()
    b["fill_now_kg"] = fill_now.round(1)
    b["expected_kg"] = (fill_now + f1).round(1)
    b["expected_pct"] = (100 * b.expected_kg / b.capacity_kg).clip(upper=150).round(0)
    last_pick = d[d.collected].groupby("bin_id").date.max()
    b["days_since_pickup"] = (d.date.max() - last_pick.reindex(b.index)).dt.days.fillna(99) + 1
    # visit if forecast says it will be >= threshold full, or wet waste would sit > max_age days
    b["must_visit"] = (b.expected_kg >= threshold * b.capacity_kg) | (b.days_since_pickup >= max_age_days)
    # baseline = what the fixed schedule actually collected on the most recent day
    b["fixed_beat"] = now.collected.reindex(b.index).fillna(False).astype(bool)
    stops, summary = [], {"date": str(f1.index.size and forecast.date.min().date()), "vehicles": []}
    tot_smart = tot_fixed = 0.0
    for truck, g in b.groupby("truck"):
        sel = g[g.must_visit]
        lat = np.r_[depot[0], sel.lat.values]
        lon = np.r_[depot[1], sel.lon.values]
        tour, length = solve_tsp(lat, lon) if len(sel) else ([0, 0], 0.0)
        fb = g[g.fixed_beat]
        _, fixed_len = solve_tsp(np.r_[depot[0], fb.lat.values], np.r_[depot[1], fb.lon.values])
        tot_smart += length
        tot_fixed += fixed_len
        for seq, idx in enumerate(tour[1:-1], start=1):
            r = sel.iloc[idx - 1]
            stops.append(dict(vehicle_id=truck, seq=seq, bin_id=r.name, lat=r.lat, lon=r.lon,
                              expected_pct=r.expected_pct, land_use=r.land_use,
                              reason="forecast full" if r.expected_kg >= threshold * r.capacity_kg
                              else f"{int(r.days_since_pickup)} days since pickup"))
        summary["vehicles"].append(dict(vehicle_id=truck, stops=int(len(sel)), bins_in_zone=int(len(g)), fixed_beat_stops=int(len(fb)),
                                        km=round(length / 1000, 2), fixed_beat_km=round(fixed_len / 1000, 2)))
    summary["total_km"] = round(tot_smart / 1000, 2)
    summary["fixed_beat_km"] = round(tot_fixed / 1000, 2)
    summary["km_saved_pct"] = round(100 * (1 - tot_smart / tot_fixed), 1) if tot_fixed else 0.0
    summary["est_diesel_saved_l_day"] = round((tot_fixed - tot_smart) / 1000 / 4.0, 1)  # ~4 km/l garbage truck
    return pd.DataFrame(stops), summary


def backtest_policy(bins: pd.DataFrame, daily: pd.DataFrame, preds: pd.DataFrame, depot,
                    threshold: float = 0.7, max_age_days: int = 4) -> dict:
    """Replay the last N days (where out-of-sample forecasts exist) under the smart policy and
    compare with what the fixed schedule actually did: km driven, bin visits, overflow events."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    p = preds.copy()
    p["date"] = pd.to_datetime(p.date)
    days = sorted(p.date.unique())
    b = bins.set_index("bin_id")
    cap = b.capacity_kg
    start = d[d.date == days[0] - pd.Timedelta(days=1)].set_index("bin_id")
    fill = start.fill_kg_reading.where(~start.collected, 0).clip(lower=0).reindex(b.index).fillna(0)
    age = pd.Series(1, index=b.index)
    res = {"days": len(days), "smart_km": 0.0, "fixed_km": 0.0, "smart_visits": 0, "fixed_visits": 0,
           "smart_overflow_bin_days": 0, "fixed_overflow_bin_days": 0}
    for day in days:
        today = d[d.date == day].set_index("bin_id").reindex(b.index)
        pred = p[p.date == day].set_index("bin_id").pred.reindex(b.index).fillna(0)
        visit = ((fill + pred) >= threshold * cap) | (age >= max_age_days)
        fill = fill.where(~visit, 0) + today.generated_kg
        res["smart_overflow_bin_days"] += int((fill > cap).sum())
        fill = fill.clip(upper=cap)
        age = age.where(~visit, 0) + 1
        res["fixed_overflow_bin_days"] += int((today.overflow_kg > 0).sum())
        res["smart_visits"] += int(visit.sum())
        res["fixed_visits"] += int(today.collected.sum())
        for truck, g in b.groupby("truck"):
            for key, mask in (("smart_km", visit), ("fixed_km", today.collected.astype(bool))):
                sel = g[mask.reindex(g.index)]
                if len(sel):
                    res[key] += solve_tsp(np.r_[depot[0], sel.lat.values], np.r_[depot[1], sel.lon.values])[1] / 1000
    res["smart_km"], res["fixed_km"] = round(res["smart_km"], 1), round(res["fixed_km"], 1)
    res["km_saved_pct"] = round(100 * (1 - res["smart_km"] / res["fixed_km"]), 1)
    res["visits_saved_pct"] = round(100 * (1 - res["smart_visits"] / res["fixed_visits"]), 1)
    res["overflow_reduction_pct"] = round(
        100 * (1 - res["smart_overflow_bin_days"] / max(res["fixed_overflow_bin_days"], 1)), 1)
    return res
