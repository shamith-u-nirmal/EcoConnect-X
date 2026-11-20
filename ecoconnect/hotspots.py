"""
Where is garbage generated, and how much?

1. Grid density map: kg/day/km2 per ~250 m cell from load-cell bins.
2. Hotspot discovery: weighted DBSCAN (haversine) over bin generation + citizen dump reports.
   Clusters are ranked by total kg/day and labelled with their dominant source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .geo import CELL_AREA_KM2, cell_center, cell_id, EARTH_R_M


def density_grid(bins: pd.DataFrame, daily: pd.DataFrame, reports: pd.DataFrame, last_days: int = 28):
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    recent = d[d.date > d.date.max() - pd.Timedelta(days=last_days)]
    kg = recent.groupby("bin_id").generated_kg.mean().rename("kg_day")
    b = bins.merge(kg, left_on="bin_id", right_index=True)
    b["cell"] = cell_id(b.lat.values, b.lon.values)
    grid = b.groupby("cell").agg(kg_day=("kg_day", "sum"), bins=("bin_id", "count"),
                                 land_use=("land_use", lambda s: s.mode().iat[0]))
    r = reports.copy()
    r["cell"] = cell_id(r.lat.values, r.lon.values)
    grid = grid.join(r.groupby("cell").size().rename("dump_reports"), how="outer").fillna(
        {"kg_day": 0, "bins": 0, "dump_reports": 0, "land_use": "unknown"})
    grid["kg_day_km2"] = (grid.kg_day / CELL_AREA_KM2).round(1)
    centers = np.array([cell_center(c) for c in grid.index])
    grid["lat"], grid["lon"] = centers[:, 0].round(6), centers[:, 1].round(6)
    return grid.reset_index().rename(columns={"index": "cell"})


def _clusters(lat, lon, w, eps_m, min_w):
    X = np.radians(np.c_[lat, lon])
    db = DBSCAN(eps=eps_m / EARTH_R_M, min_samples=min_w, metric="haversine", algorithm="ball_tree")
    return db.fit_predict(X, sample_weight=w)


def find_hotspots(bins, daily, reports, gen_eps_m: float = 350, gen_min_kg: float = 150,
                  dump_eps_m: float = 300, dump_min_reports: int = 25, dump_days: int = 90):
    """Two kinds of hotspot, found with weighted DBSCAN (haversine):
       * generation hotspots - >= gen_min_kg kg/day of bin waste within gen_eps_m
       * dumping hotspots    - >= dump_min_reports citizen reports within dump_eps_m (last dump_days)
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d.date)
    kg = d[d.date > d.date.max() - pd.Timedelta(days=28)].groupby("bin_id").generated_kg.mean()
    w = bins.bin_id.map(kg).values
    rows = []
    lab = _clusters(bins.lat.values, bins.lon.values, w, gen_eps_m, gen_min_kg)
    for c in set(lab) - {-1}:
        m = lab == c
        rows.append(dict(type="generation", lat=np.average(bins.lat[m], weights=w[m]),
                         lon=np.average(bins.lon[m], weights=w[m]), kg_day=round(w[m].sum(), 1),
                         n_bins=int(m.sum()), n_reports=0,
                         radius_m=int(max(150, np.ptp(bins.lat[m]) * 111_000 / 2 + 120)),
                         land_use=bins.land_use[m].mode().iat[0]))
    r = reports.copy()
    r["date"] = pd.to_datetime(r.date)
    r = r[(r.date > r.date.max() - pd.Timedelta(days=dump_days)) & (r.category != "overflowing_bin")]
    lab = _clusters(r.lat.values, r.lon.values, np.ones(len(r)), dump_eps_m, dump_min_reports)
    for c in set(lab) - {-1}:
        m = lab == c
        rows.append(dict(type="dumping", lat=r.lat[m].mean(), lon=r.lon[m].mean(),
                         kg_day=round(m.sum() * 25.0 / dump_days, 1),  # ~25 kg per reported dump
                         n_bins=0, n_reports=int(m.sum()),
                         radius_m=int(max(150, np.ptp(r.lat[m]) * 111_000 / 2 + 80)), land_use=""))
    hs = pd.DataFrame(rows).sort_values(["type", "kg_day"], ascending=[False, False]).reset_index(drop=True)
    hs.insert(0, "hotspot_id", [f"{'G' if t == 'generation' else 'D'}{n:02d}" for t, n in
                                zip(hs.type, hs.groupby("type").cumcount() + 1)])
    return hs.round({"lat": 6, "lon": 6})
