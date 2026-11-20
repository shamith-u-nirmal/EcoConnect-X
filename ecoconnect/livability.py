"""
Waste Livability Score (0-100) for home buyers and planners.

For every ~250 m cell the score blends five explainable signals:
    generation density      (kg/day/km2, last 28 days)              weight 0.30
    distance to a hotspot   (closer = worse)                         weight 0.20
    open-dumping reports    (citizen reports per cell)               weight 0.20
    collection reliability  (served / required visits on its streets) weight 0.15
    growth trend            (forecast vs last-28-day mean)           weight 0.15
Each signal is converted to a 0-1 percentile risk across the town; score = 100 * (1 - weighted risk).
Zones: Clean (>=60) / Moderate (40-60) / Waste-stressed (<40). A KMeans model (k=5) on the risk
profile groups cells into archetypes (e.g. "Dumping-prone area", "Fast-growing layout").
The response always lists the top reasons, so the score is never a black box.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .geo import cell_center, cell_id, haversine_m

WEIGHTS = {"density": 0.30, "hotspot": 0.20, "dumping": 0.20, "reliability": 0.15, "trend": 0.15}
ARCHETYPE = {
    "density": "High-generation core",
    "hotspot": "Hotspot fringe",
    "dumping": "Dumping-prone area",
    "reliability": "Under-served by collection",
    "trend": "Fast-growing layout",
}
REASON_TEXT = {
    "density": "high everyday waste generation nearby",
    "hotspot": "close to a garbage hotspot",
    "dumping": "frequent open-dumping complaints",
    "reliability": "collection vehicle visits less often than needed",
    "trend": "waste generation rising fast (new construction / growth)",
}


def score_cells(grid: pd.DataFrame, hotspots: pd.DataFrame, service: pd.DataFrame,
                bins: pd.DataFrame, daily: pd.DataFrame, forecast: pd.DataFrame, seed: int = 0):
    g = grid.copy().set_index("cell")
    # distance to nearest hotspot
    d = haversine_m(g.lat.values[:, None], g.lon.values[:, None],
                    hotspots.lat.values[None, :], hotspots.lon.values[None, :])
    g["hotspot_m"] = d.min(axis=1).round(0)
    # collection reliability per cell (streets whose midpoint lies in the cell)
    sv = service[service.required_per_week > 0].copy()
    sv["cell"] = cell_id(sv.mid_lat.values, sv.mid_lon.values)
    rel = (sv.visits_per_week / sv.required_per_week).clip(upper=1.5).groupby(sv.cell).mean()
    g["reliability"] = rel.reindex(g.index).fillna(1.0).round(2)
    # trend: forecast mean vs recent mean per cell
    dd = daily.copy()
    dd["date"] = pd.to_datetime(dd.date)
    recent = dd[dd.date > dd.date.max() - pd.Timedelta(days=28)].groupby("bin_id").generated_kg.mean()
    fut = forecast.groupby("bin_id").pred_kg.mean()
    b = bins.set_index("bin_id")
    bc = pd.Series(cell_id(b.lat.values, b.lon.values), index=b.index)
    tr = (fut.groupby(bc).sum() / recent.groupby(bc).sum().replace(0, np.nan))
    early = dd[dd.date <= dd.date.min() + pd.Timedelta(days=28)].groupby("bin_id").generated_kg.mean()
    long_tr = recent.groupby(bc).sum() / early.groupby(bc).sum().replace(0, np.nan)
    g["trend"] = (0.5 * tr + 0.5 * long_tr).reindex(g.index).round(3)  # NaN = no bins in cell

    risk = pd.DataFrame(index=g.index)
    risk["density"] = g.kg_day_km2.rank(pct=True)
    risk["hotspot"] = (-g.hotspot_m).rank(pct=True)
    risk["dumping"] = g.dump_reports.rank(pct=True)
    risk["reliability"] = (-g.reliability).rank(pct=True)
    risk["trend"] = g.trend.rank(pct=True).fillna(0.5)  # unknown trend -> neutral
    g["trend"] = g.trend.fillna(1.0)
    g["score"] = (100 * (1 - sum(risk[k] * w for k, w in WEIGHTS.items()))).round(0).clip(0, 100)
    g["zone"] = pd.cut(g.score, [-1, 40, 60, 101], labels=["Waste-stressed", "Moderate", "Clean"]).astype(str)
    # unsupervised archetypes: KMeans on the risk profile, each cluster named by its dominant risk
    km = KMeans(n_clusters=5, n_init=10, random_state=seed).fit(risk.values)
    prof = risk.groupby(km.labels_).mean()
    calm = prof.sum(axis=1).idxmin()
    arche = {c: ("Low-risk neighbourhood" if c == calm else ARCHETYPE[(prof.loc[c] - risk.mean()).idxmax()])
             for c in prof.index}
    g["archetype"] = [arche[l] for l in km.labels_]
    contrib = risk.mul(pd.Series(WEIGHTS))
    g["reasons"] = [
        [REASON_TEXT[k] for k in row.sort_values(ascending=False).index[:2] if risk.loc[c, k] > 0.6]
        for c, row in contrib.iterrows()]
    for k in WEIGHTS:
        g[f"risk_{k}"] = risk[k].round(2)
    return g.reset_index()


def lookup(cells: pd.DataFrame, lat: float, lon: float, radius_m: float = 500) -> dict:
    """Score for a plot location: distance-weighted mean of cells within radius_m."""
    d = haversine_m(lat, lon, cells.lat.values, cells.lon.values)
    near = cells[d <= radius_m].assign(dist=d[d <= radius_m])
    if near.empty:
        return {"found": False, "message": "Location is outside the monitored area."}
    w = 1 / (near.dist + 100)
    score = float(np.average(near.score, weights=w))
    own = near.sort_values("dist").iloc[0]
    reasons = [r for rs in near.sort_values("dist").reasons for r in rs]
    reasons = list(dict.fromkeys(reasons))[:3]
    alt = cells[(cells.score >= 70) & (haversine_m(lat, lon, cells.lat.values, cells.lon.values) < 2500)]
    alt = alt.assign(dist=haversine_m(lat, lon, alt.lat.values, alt.lon.values)).nsmallest(3, "dist")
    return {
        "found": True,
        "score": round(score),
        "zone": "Clean" if score >= 60 else "Moderate" if score >= 40 else "Waste-stressed",
        "archetype": own.archetype,
        "nearest_hotspot_m": int(own.hotspot_m),
        "kg_day_km2": round(float(np.average(near.kg_day_km2, weights=w)), 1),
        "reasons": reasons or ["no major waste risk signals nearby"],
        "cleaner_alternatives": [
            {"lat": round(r.lat, 5), "lon": round(r.lon, 5), "score": int(r.score), "distance_m": int(r.dist)}
            for r in alt.itertuples()],
    }
