"""
EcoConnect-X: AI-Powered Environmental & Smart City Intelligence Platform (Flask API + dashboard).

    python app/app.py           # http://localhost:5000
Runs the ML pipeline on first start if outputs/ is empty.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoconnect import livability, pipeline  # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "data" / "generated"
LIVE = ROOT / "data" / "live"

app = Flask(__name__)
_cache: dict = {}


def _load():
    if not (OUT / "summary.json").exists():
        pipeline.run(DATA, OUT, verbose=False)
    _cache.clear()
    _cache["summary"] = json.loads((OUT / "summary.json").read_text())
    for name in ("hotspots", "density_grid", "street_service", "trips", "route_plan", "forecast_7d",
                 "forecast_backtest"):
        _cache[name] = pd.read_csv(OUT / f"{name}.csv")
    _cache["cells"] = pd.read_json(OUT / "livability_cells.json")
    _cache["bins"] = pd.read_csv(DATA / "bins.csv")
    _cache["daily"] = pd.read_csv(DATA / "bin_daily.csv")


def C(name):
    if not _cache:
        _load()
    return _cache[name]


def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/summary")
def summary():
    return jsonify(C("summary"))


@app.get("/api/hotspots")
def hotspots():
    return jsonify(records(C("hotspots")))


@app.get("/api/density")
def density():
    return jsonify(records(C("density_grid")))


@app.get("/api/streets")
def streets():
    s = C("street_service")
    if st := request.args.get("status"):
        s = s[s.status == st]
    return jsonify(records(s))


@app.get("/api/trips")
def trips():
    t = C("trips")
    if request.args.get("anomalies") == "1":
        t = t[t.is_anomaly]
    return jsonify(records(t.sort_values("anomaly_score", ascending=False)))


@app.get("/api/route")
def route():
    r = C("route_plan")
    if v := request.args.get("vehicle"):
        r = r[r.vehicle_id == v]
    return jsonify({"depot": pipeline.DEPOT, "stops": records(r), "summary": C("summary")["routing"]})


@app.get("/api/forecast")
def forecast():
    fc = C("forecast_7d")
    d = C("daily")
    bin_id = request.args.get("bin_id")
    if bin_id:
        hist = d[d.bin_id == bin_id].tail(28)[["date", "generated_kg"]]
        f = fc[fc.bin_id == bin_id][["date", "pred_kg"]]
    else:
        hist = d.groupby("date", as_index=False).generated_kg.sum().tail(28)
        f = fc.groupby("date", as_index=False).pred_kg.sum()
    return jsonify({"bin_id": bin_id, "history": records(hist.round(1)), "forecast": records(f.round(1))})


@app.get("/api/livability")
def livability_lookup():
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return jsonify({"error": "pass ?lat=..&lon=.."}), 400
    return jsonify(livability.lookup(C("cells"), lat, lon))


@app.get("/api/livability/cells")
def livability_cells():
    cols = ["cell", "lat", "lon", "score", "zone", "archetype", "kg_day_km2", "dump_reports", "reasons"]
    return jsonify(records(C("cells")[cols]))


# ---------------------------------------------------------------- ingestion (IoT + citizen app)
def _append(name, row: dict):
    LIVE.mkdir(parents=True, exist_ok=True)
    f = LIVE / f"{name}.csv"
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


@app.post("/api/ingest/gps")
def ingest_gps():
    """ESP32 + NEO-6M tracker (or driver phone) posts {vehicle_id, lat, lon, speed_kmph[, timestamp]}."""
    pings = request.get_json(force=True)
    pings = pings if isinstance(pings, list) else [pings]
    for p in pings:
        _append("gps", {"vehicle_id": p["vehicle_id"], "timestamp": p.get("timestamp", datetime.now().isoformat()),
                        "lat": float(p["lat"]), "lon": float(p["lon"]), "speed_kmph": float(p.get("speed_kmph", 0))})
    return jsonify({"ok": True, "received": len(pings)})


@app.post("/api/ingest/bin")
def ingest_bin():
    """ESP32 + HX711 load-cell bin posts {bin_id, weight_kg}."""
    p = request.get_json(force=True)
    _append("bin_readings", {"bin_id": p["bin_id"], "timestamp": datetime.now().isoformat(),
                             "weight_kg": float(p["weight_kg"])})
    return jsonify({"ok": True})


@app.post("/api/report")
def report():
    """Citizen app posts a geo-tagged open-dumping report {lat, lon, category}."""
    p = request.get_json(force=True)
    _append("reports", {"timestamp": datetime.now().isoformat(), "lat": float(p["lat"]), "lon": float(p["lon"]),
                        "category": p.get("category", "open_dump")})
    return jsonify({"ok": True})


@app.post("/api/retrain")
def retrain():
    s = pipeline.run(DATA, OUT, verbose=False)
    _load()
    return jsonify({"ok": True, "forecast": s["forecast"], "runtime_s": s["runtime_s"]})


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
