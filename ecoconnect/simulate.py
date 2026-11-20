"""
Synthetic-but-realistic data generator for a ~5 x 5 km town (centred on Puttur, Karnataka).

Produces the four data streams EcoConnect-X consumes in production:
  streets.csv      street segments with land use               (from OpenStreetMap in production)
  bins.csv         smart-bin registry                            (from the ESP32 load-cell bins)
  bin_daily.csv    daily load-cell readings per bin              (sensor stream)
  gps.csv          collection-vehicle GPS pings (last 28 days)   (ESP32 + NEO-6M tracker / phone app)
  reports.csv      geo-tagged citizen reports of open dumping    (citizen app)

The generator deliberately plants real-world problems the ML layer must discover:
  * a market core and commercial corridor that generate far more waste
  * a fast-growing new layout (generation trending up week over week)
  * one driver who systematically skips outskirt streets (under-served zone)
  * occasional anomalous trips (short trips, long idling)
  * festival spikes and monsoon effects
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .geo import haversine_m

CENTER = (12.7590, 75.2010)  # Puttur town centre
N = 14                       # street-grid nodes per side
STEP = 0.0035                # ~390 m between junctions
DAYS = 180
GPS_DAYS = 28
END_DATE = pd.Timestamp("2026-11-19")

BASE_KG = {"market": 55, "commercial": 32, "residential": 16, "new_layout": 14, "institutional": 10, "vacant": 3}
FREQ_DAYS = {"market": 1, "commercial": 1, "residential": 2, "new_layout": 3, "institutional": 3, "vacant": 4}
BIN_CAPACITY_KG = 120.0

FESTIVALS = {  # configurable local calendar
    "Ganesh Chaturthi": ("2026-09-14", "2026-09-16", 1.35),
    "Dasara": ("2026-10-11", "2026-10-20", 1.25),
    "Deepavali": ("2026-11-08", "2026-11-10", 1.45),
}


def _land_use(lat, lon, rng):
    dlat, dlon = lat - CENTER[0], lon - CENTER[1]
    d = np.hypot(dlat, dlon)
    if d < 0.0055:
        return "market"
    if abs(dlat) < 0.0022 and abs(dlon) < 0.02:  # main road corridor
        return "commercial"
    if dlat > 0.009 and dlon > 0.006:            # new layout in the north-east
        return "new_layout"
    if d > 0.024 and dlat < -0.004:              # southern outskirts
        return "vacant" if rng.random() < 0.6 else "residential"
    if rng.random() < 0.08:
        return "institutional"
    return "residential"


def build_streets(rng):
    G = nx.Graph()
    lat0 = CENTER[0] - STEP * (N - 1) / 2
    lon0 = CENTER[1] - STEP * (N - 1) / 2
    for i in range(N):
        for j in range(N):
            G.add_node((i, j), lat=lat0 + i * STEP + rng.normal(0, 0.0004),
                       lon=lon0 + j * STEP + rng.normal(0, 0.0004))
    for i in range(N):
        for j in range(N):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b in G and (rng.random() > 0.08 or i in (0, N - 1) or j in (0, N - 1)):
                    G.add_edge(a, b)
    rows = []
    for k, (a, b) in enumerate(G.edges()):
        la, oa, lb, ob = G.nodes[a]["lat"], G.nodes[a]["lon"], G.nodes[b]["lat"], G.nodes[b]["lon"]
        mlat, mlon = (la + lb) / 2, (oa + ob) / 2
        lu = _land_use(mlat, mlon, rng)
        length = float(haversine_m(la, oa, lb, ob))
        G.edges[a, b].update(street_id=f"S{k:03d}", length=length)
        rows.append(dict(street_id=f"S{k:03d}", u=f"{a[0]}_{a[1]}", v=f"{b[0]}_{b[1]}",
                         lat1=la, lon1=oa, lat2=lb, lon2=ob, mid_lat=mlat, mid_lon=mlon,
                         length_m=round(length, 1), land_use=lu))
    return G, pd.DataFrame(rows)


def simulate(out_dir: str | Path = "data/generated", seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    G, streets = build_streets(rng)
    lu_map = dict(zip(streets.street_id, streets.land_use))

    # ---- bins: one smart bin on most street segments --------------------------------------
    has_bin = rng.random(len(streets)) < 0.7
    bins = streets[has_bin][["street_id", "mid_lat", "mid_lon", "land_use"]].copy()
    bins = bins.rename(columns={"mid_lat": "lat", "mid_lon": "lon"}).reset_index(drop=True)
    bins.insert(0, "bin_id", [f"B{i:03d}" for i in range(len(bins))])
    bins["base_kg"] = bins.land_use.map(BASE_KG) * rng.lognormal(0, 0.25, len(bins))
    bins["capacity_kg"] = BIN_CAPACITY_KG

    # ---- trucks: split the town into 3 zones by longitude -----------------------------------
    order = np.argsort(bins.lon.values)
    truck_of = np.empty(len(bins), dtype=int)
    for t, chunk in enumerate(np.array_split(order, 3)):
        truck_of[chunk] = t
    bins["truck"] = [f"KA19-T{t + 1}" for t in truck_of]
    depot = (CENTER[0] - 0.012, CENTER[1] - 0.004)

    dates = pd.date_range(END_DATE - pd.Timedelta(days=DAYS - 1), END_DATE)
    month = dates.month.values
    rain_mm = np.where(np.isin(month, [6, 7, 8, 9]), rng.gamma(2.0, 9.0, len(dates)),
                       rng.gamma(0.4, 4.0, len(dates)))
    fest = np.ones(len(dates))
    fest_name = np.array([""] * len(dates), dtype=object)
    for name, (s, e, mult) in FESTIVALS.items():
        m = (dates >= s) & (dates <= e)
        fest[m] = mult
        fest_name[m] = name

    fill = np.zeros(len(bins))
    last_pick = np.zeros(len(bins), dtype=int) - rng.integers(0, 3, len(bins))
    freq = bins.land_use.map(FREQ_DAYS).values
    lazy_truck = "KA19-T1"  # this crew systematically skips outskirt streets
    outskirt = bins.land_use.isin(["vacant", "residential"]).values & (bins.lat.values < CENTER[0] - 0.012)

    daily_rows, visits = [], {}
    for d_i, day in enumerate(dates):
        dow = day.dayofweek
        dow_f = np.where(bins.land_use.values == "market", 1.25 if dow in (5, 6) else 1.0,
                         1.12 if dow == 6 else 1.0)
        growth = np.where(bins.land_use.values == "new_layout", 1 + 0.9 * d_i / DAYS, 1.0)
        wet = 1 + 0.004 * rain_mm[d_i]
        gen = bins.base_kg.values * dow_f * fest[d_i] * growth * wet * rng.lognormal(0, 0.18, len(bins))
        fill += gen
        due = (d_i - last_pick) >= freq
        skip = rng.random(len(bins)) < 0.05
        lazy = (bins.truck.values == lazy_truck) & outskirt & (rng.random(len(bins)) < 0.8)
        collected = due & ~skip & ~lazy
        overflow = np.clip(fill - bins.capacity_kg.values, 0, None)
        pickup = np.where(collected, np.minimum(fill, bins.capacity_kg.values), 0.0)
        reading = np.minimum(fill, bins.capacity_kg.values) + rng.normal(0, 1.5, len(bins))  # load-cell noise
        for k in range(len(bins)):
            daily_rows.append((day.date(), bins.bin_id[k], round(gen[k], 2), round(max(reading[k], 0), 2),
                               bool(collected[k]), round(pickup[k], 2), round(overflow[k], 2)))
        fill = np.where(collected, 0.0, np.minimum(fill, bins.capacity_kg.values))
        last_pick = np.where(collected, d_i, last_pick)
        visits[day] = bins.bin_id[collected].tolist()

    bin_daily = pd.DataFrame(daily_rows, columns=["date", "bin_id", "generated_kg", "fill_kg_reading",
                                                  "collected", "pickup_kg", "overflow_kg"])
    weather = pd.DataFrame({"date": dates.date, "rain_mm": rain_mm.round(1), "festival": fest_name})

    # ---- citizen reports of open dumping ------------------------------------------------------
    ov = bin_daily.groupby("bin_id").overflow_kg.mean()
    rep_rows = []
    for _, s in streets.iterrows():
        b = bins[bins.street_id == s.street_id]
        lam = 0.02 + (0.12 if s.land_use == "vacant" else 0.0)
        if len(b):
            lam += 0.004 * ov.get(b.bin_id.iloc[0], 0)
        else:
            lam += 0.04  # no bin on the street -> people dump
        n = rng.poisson(lam * DAYS)
        for _ in range(n):
            rep_rows.append(dict(report_id=f"R{len(rep_rows):04d}",
                                 date=dates[rng.integers(0, DAYS)].date(),
                                 lat=s.mid_lat + rng.normal(0, 0.0005), lon=s.mid_lon + rng.normal(0, 0.0005),
                                 category=rng.choice(["open_dump", "overflowing_bin", "burning"], p=[.6, .3, .1])))
    reports = pd.DataFrame(rep_rows)

    # ---- GPS traces for the last 28 days --------------------------------------------------------
    node_xy = {n: (G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in G.nodes}
    depot_node = min(G.nodes, key=lambda n: (node_xy[n][0] - depot[0]) ** 2 + (node_xy[n][1] - depot[1]) ** 2)
    street_nodes = {r.street_id: tuple(map(int, r.u.split("_"))) for r in streets.itertuples()}
    street_other = {r.street_id: tuple(map(int, r.v.split("_"))) for r in streets.itertuples()}
    bin_street = dict(zip(bins.bin_id, bins.street_id))
    bin_truck = dict(zip(bins.bin_id, bins.truck))
    gps_rows = []
    for day in dates[-GPS_DAYS:]:
        for truck in sorted(bins.truck.unique()):
            todo = [street_nodes[bin_street[b]] for b in visits[day] if bin_truck[b] == truck]
            other = {street_nodes[bin_street[b]]: street_other[bin_street[b]] for b in visits[day]
                     if bin_truck[b] == truck}
            if not todo:
                continue
            anomaly = rng.random() < 0.08
            if anomaly:
                todo = todo[: max(1, len(todo) // 4)]  # short trip, crew leaves early
            ts = pd.Timestamp(day) + pd.Timedelta(hours=6, minutes=int(rng.integers(0, 30)))
            cur, path = depot_node, [depot_node]
            remaining = set(todo)
            while remaining:
                nxt = min(remaining, key=lambda n: np.hypot(node_xy[n][0] - node_xy[cur][0],
                                                             node_xy[n][1] - node_xy[cur][1]))
                path += nx.shortest_path(G, cur, nxt, weight="length")[1:]
                # drive the street segment that holds the bin
                path.append(other[nxt])  # drive down the bin's street and back
                path.append(nxt)
                cur = nxt
                remaining.discard(nxt)
            path += nx.shortest_path(G, cur, depot_node, weight="length")[1:]
            stopped = set()
            for a, b in zip(path[:-1], path[1:]):
                (la, oa), (lb, ob) = node_xy[a], node_xy[b]
                dist = float(haversine_m(la, oa, lb, ob))
                n_pts = max(2, int(dist // 40))
                speed = rng.uniform(10, 22)
                is_bin_street = a in other and other[a] == b and a not in stopped
                for f in np.linspace(0, 1, n_pts, endpoint=False):
                    ts += pd.Timedelta(seconds=float(dist / n_pts / (speed / 3.6)))
                    gps_rows.append((truck, ts, la + f * (lb - la) + rng.normal(0, 0.00006),
                                     oa + f * (ob - oa) + rng.normal(0, 0.00006), round(speed, 1)))
                    if is_bin_street and f >= 0.45:  # stop at the bin (street midpoint) to collect
                        stopped.add(a)
                        is_bin_street = False
                        stop = rng.uniform(90, 240) + (rng.uniform(1800, 3600) if anomaly and rng.random() < .3 else 0)
                        mlat, mlon = la + 0.5 * (lb - la), oa + 0.5 * (ob - oa)
                        for _ in range(int(stop // 30)):
                            ts += pd.Timedelta(seconds=30)
                            gps_rows.append((truck, ts, mlat + rng.normal(0, 0.00004),
                                             mlon + rng.normal(0, 0.00004), 0.0))
    gps = pd.DataFrame(gps_rows, columns=["vehicle_id", "timestamp", "lat", "lon", "speed_kmph"])
    gps["timestamp"] = gps.timestamp.dt.floor("s")

    streets.to_csv(out / "streets.csv", index=False)
    bins.round(6).to_csv(out / "bins.csv", index=False)
    bin_daily.to_csv(out / "bin_daily.csv", index=False)
    weather.to_csv(out / "weather_calendar.csv", index=False)
    reports.round(6).to_csv(out / "reports.csv", index=False)
    gps.round({"lat": 6, "lon": 6}).to_csv(out / "gps.csv", index=False)
    return dict(streets=len(streets), bins=len(bins), bin_days=len(bin_daily), gps_points=len(gps),
                reports=len(reports))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(simulate(a.out, a.seed))
