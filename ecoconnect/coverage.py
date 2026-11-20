"""
Collection-coverage audit from vehicle GPS.

* Map-matching: every GPS ping is snapped to the nearest street segment (BallTree, haversine)
  using points sampled along each segment; pings > 35 m from any street are discarded.
* A street counts as "served" on a day only when the vehicle *stopped* on it (>= 2 stationary
  matched pings); driving through does not count as collection.
* Required frequency per street comes from the ML forecast (kg/day vs bin capacity, with a
  hygiene floor of one pickup every 4 days), so the
  panchayat sees streets driven less often than the waste actually requires.
* Trip anomaly detection: IsolationForest over per-trip features (km, hours, stops, idle time,
  streets covered) flags short trips, long idling and skipped beats for review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import BallTree

from .geo import EARTH_R_M, haversine_m

MATCH_RADIUS_M = 35
MAX_AGE_DAYS = 4


def _street_samples(streets: pd.DataFrame, step_m: float = 20):
    lat, lon, sid = [], [], []
    for s in streets.itertuples():
        n = max(2, int(s.length_m // step_m))
        f = np.linspace(0.1, 0.9, n)  # avoid junctions, which belong to several streets
        lat += list(s.lat1 + f * (s.lat2 - s.lat1))
        lon += list(s.lon1 + f * (s.lon2 - s.lon1))
        sid += [s.street_id] * n
    return np.array(lat), np.array(lon), np.array(sid)


def map_match(gps: pd.DataFrame, streets: pd.DataFrame) -> pd.DataFrame:
    lat, lon, sid = _street_samples(streets)
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    dist, idx = tree.query(np.radians(gps[["lat", "lon"]].values), k=1)
    g = gps.copy()
    g["street_id"] = sid[idx[:, 0]]
    g["match_m"] = dist[:, 0] * EARTH_R_M
    return g[g.match_m <= MATCH_RADIUS_M]


def street_service(matched: pd.DataFrame, streets: pd.DataFrame, bins: pd.DataFrame,
                   forecast: pd.DataFrame) -> pd.DataFrame:
    m = matched.copy()
    m["date"] = pd.to_datetime(m.timestamp).dt.date
    days = m.date.nunique()
    # a street is *served* only if the vehicle stopped on it (>= 2 stationary pings); passing through
    # on the way somewhere else does not count as collection
    served = m[m.speed_kmph < 1].groupby(["street_id", "date"]).size()
    served = served[served >= 2].reset_index().groupby("street_id").agg(
        visits=("date", "count"), last_visit=("date", "max"))
    s = streets.set_index("street_id")[["mid_lat", "mid_lon", "land_use", "length_m", "lat1", "lon1", "lat2", "lon2"]]
    s = s.join(served).fillna({"visits": 0})
    s["visits_per_week"] = (s.visits / days * 7).round(2)
    # required visits/week = predicted weekly kg / (0.85 * capacity), only for streets with bins
    wk = forecast.groupby("bin_id").pred_kg.mean() * 7
    b = bins.set_index("bin_id")
    need = (wk / (0.85 * b.capacity_kg)).groupby(b.street_id).sum()
    # hygiene floor: wet waste must not sit more than MAX_AGE_DAYS in a bin
    need = need.clip(lower=7 / MAX_AGE_DAYS)
    s["required_per_week"] = need.reindex(s.index).fillna(0).round(2)
    s["gap"] = (s.required_per_week - s.visits_per_week).round(2)
    s["status"] = np.select(
        [s.required_per_week == 0, s.gap > 1.0, s.gap > 0.25, s.gap < -2.5],
        ["no bin", "under-served", "watch", "over-served"], "ok")
    return s.reset_index()


def trip_features(gps: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    g = gps.copy()
    g["timestamp"] = pd.to_datetime(g.timestamp)
    g["date"] = g.timestamp.dt.date
    g = g.sort_values(["vehicle_id", "timestamp"])
    rows = []
    mm = matched.copy()
    mm["date"] = pd.to_datetime(mm.timestamp).dt.date
    cov = mm.groupby(["vehicle_id", "date"]).street_id.nunique()
    for (v, d), t in g.groupby(["vehicle_id", "date"]):
        km = haversine_m(t.lat.values[:-1], t.lon.values[:-1], t.lat.values[1:], t.lon.values[1:]).sum() / 1000
        hrs = (t.timestamp.max() - t.timestamp.min()).total_seconds() / 3600
        stopped = t.speed_kmph.values == 0
        idle_min = stopped.sum() * 0.5
        stops = int((np.diff(stopped.astype(int)) == 1).sum())
        rows.append(dict(vehicle_id=v, date=d, km=round(km, 2), hours=round(hrs, 2), stops=stops,
                         idle_min=round(idle_min, 1), streets=int(cov.get((v, d), 0)),
                         idle_per_stop=round(idle_min / max(stops, 1), 2)))
    return pd.DataFrame(rows)


def detect_trip_anomalies(trips: pd.DataFrame, contamination: float = 0.08, seed: int = 0):
    X = trips[["km", "hours", "stops", "idle_min", "streets", "idle_per_stop"]].copy()
    # compare each trip with that vehicle's own normal beat
    X = X.groupby(trips.vehicle_id).transform(lambda c: (c - c.median()) / (c.std() + 1e-6))
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    t = trips.copy()
    t["anomaly_score"] = (-iso.fit(X).score_samples(X)).round(3)
    t["is_anomaly"] = iso.predict(X) == -1
    med = trips.groupby("vehicle_id")[["streets", "idle_per_stop", "hours"]].transform("median")
    t["reason"] = np.select(
        [t.is_anomaly & (trips.streets < 0.6 * med.streets),
         t.is_anomaly & (trips.idle_per_stop > 2 * med.idle_per_stop),
         t.is_anomaly], ["short trip / beat skipped", "excessive idling", "unusual pattern"], "")
    return t
