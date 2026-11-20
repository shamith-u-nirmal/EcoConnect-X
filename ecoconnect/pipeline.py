"""
End-to-end pipeline: raw streams -> ML models -> JSON/CSV outputs served by the dashboard.

    python -m ecoconnect.pipeline            # uses data/generated (creates it if missing)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from . import coverage, forecast, hotspots, livability, routing
from .simulate import CENTER, simulate

DEPOT = (CENTER[0] - 0.012, CENTER[1] - 0.004)


def load(data_dir: Path):
    r = lambda n: pd.read_csv(data_dir / n)
    return dict(streets=r("streets.csv"), bins=r("bins.csv"), daily=r("bin_daily.csv"),
                cal=r("weather_calendar.csv").fillna({"festival": ""}), reports=r("reports.csv"),
                gps=r("gps.csv"))


def run(data_dir="data/generated", out_dir="outputs", verbose=True) -> dict:
    data_dir, out = Path(data_dir), Path(out_dir)
    if not (data_dir / "bins.csv").exists():
        simulate(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    D = load(data_dir)
    t0 = time.time()

    log("[1/5] Forecasting waste generation ...")
    model, fmetrics, test = forecast.train_and_evaluate(D["daily"], D["bins"], D["cal"])
    fc = forecast.forecast_next_days(model, D["daily"], D["bins"], D["cal"], horizon=7)
    joblib.dump(model, out / "forecast_model.joblib")
    log("      ", fmetrics)

    log("[2/5] Mapping generation density & hotspots ...")
    grid = hotspots.density_grid(D["bins"], D["daily"], D["reports"])
    hs = hotspots.find_hotspots(D["bins"], D["daily"], D["reports"])
    log(f"       {len(grid)} cells, hotspots:", hs.type.value_counts().to_dict())

    log("[3/5] Auditing collection coverage from GPS ...")
    matched = coverage.map_match(D["gps"], D["streets"])
    service = coverage.street_service(matched, D["streets"], D["bins"], fc)
    trips = coverage.detect_trip_anomalies(coverage.trip_features(D["gps"], matched))
    log("      ", service.status.value_counts().to_dict(), f"| anomalous trips: {int(trips.is_anomaly.sum())}")

    log("[4/5] Planning forecast-driven routes ...")
    stops, rsum = routing.plan_routes(D["bins"], D["daily"], fc, DEPOT)
    log("      ", {k: v for k, v in rsum.items() if k != "vehicles"})
    bt = routing.backtest_policy(D["bins"], D["daily"], test, DEPOT)
    log("       backtest:", bt)

    log("[5/5] Scoring livability for home buyers ...")
    cells = livability.score_cells(grid, hs, service, D["bins"], D["daily"], fc)
    log("      ", cells.zone.value_counts().to_dict())

    # ---- persist ---------------------------------------------------------------------------
    fc.assign(date=fc.date.dt.date.astype(str)).to_csv(out / "forecast_7d.csv", index=False)
    grid.to_csv(out / "density_grid.csv", index=False)
    hs.to_csv(out / "hotspots.csv", index=False)
    service.to_csv(out / "street_service.csv", index=False)
    trips.to_csv(out / "trips.csv", index=False)
    stops.to_csv(out / "route_plan.csv", index=False)
    cells.to_json(out / "livability_cells.json", orient="records")
    test[["date", "bin_id", "generated_kg", "pred"]].to_csv(out / "forecast_backtest.csv", index=False)

    d = D["daily"]
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "town": {"bins": int(len(D["bins"])), "streets": int(len(D["streets"])),
                 "gps_points": int(len(D["gps"])), "citizen_reports": int(len(D["reports"])),
                 "days_of_data": int(d.date.nunique())},
        "waste_today_kg": round(float(d[d.date == d.date.max()].generated_kg.sum()), 0),
        "forecast": fmetrics,
        "forecast_tomorrow_kg": round(float(fc[fc.date == fc.date.min()].pred_kg.sum()), 0),
        "hotspots": int(len(hs)),
        "coverage": {k: int(v) for k, v in service.status.value_counts().items()},
        "anomalous_trips": int(trips.is_anomaly.sum()),
        "trips_audited": int(len(trips)),
        "routing": rsum,
        "routing_backtest": bt,
        "livability_zones": {k: int(v) for k, v in cells.zone.value_counts().items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Done in {summary['runtime_s']} s -> {out}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/generated")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    run(a.data, a.out)
