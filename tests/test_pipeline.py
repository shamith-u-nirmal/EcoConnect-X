import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecoconnect import pipeline, simulate  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("ecoconnect")
    simulate.simulate(d / "data", seed=7)
    return pipeline.run(d / "data", d / "out", verbose=False), d


def test_forecast_beats_naive_baseline(run):
    s, _ = run
    assert s["forecast"]["mae_kg"] < s["forecast"]["mae_naive_yesterday_kg"]


def test_hotspots_found(run):
    s, _ = run
    assert s["hotspots"] >= 2


def test_smart_routes_save_distance_without_more_overflow(run):
    bt = run[0]["routing_backtest"]
    assert bt["smart_km"] < bt["fixed_km"]
    assert bt["smart_overflow_bin_days"] <= bt["fixed_overflow_bin_days"]


def test_coverage_finds_under_served_streets(run):
    assert run[0]["coverage"].get("under-served", 0) > 0


def test_livability_lookup(run):
    import pandas as pd
    from ecoconnect import livability
    _, d = run
    cells = pd.read_json(d / "out" / "livability_cells.json")
    r = livability.lookup(cells, 12.759, 75.201)
    assert r["found"] and 0 <= r["score"] <= 100
    assert not livability.lookup(cells, 13.5, 76.5)["found"]


def test_api(run, monkeypatch):
    sys.path.insert(0, str(ROOT / "app"))
    import app as appmod
    _, d = run
    monkeypatch.setattr(appmod, "OUT", d / "out")
    monkeypatch.setattr(appmod, "DATA", d / "data")
    monkeypatch.setattr(appmod, "LIVE", d / "live")
    appmod._cache.clear()
    c = appmod.app.test_client()
    assert c.get("/api/summary").status_code == 200
    assert len(c.get("/api/hotspots").get_json()) >= 2
    assert c.get("/api/livability?lat=12.759&lon=75.201").get_json()["found"]
    assert c.post("/api/ingest/gps", data=json.dumps({"vehicle_id": "T1", "lat": 12.75, "lon": 75.2})).get_json()["ok"]
