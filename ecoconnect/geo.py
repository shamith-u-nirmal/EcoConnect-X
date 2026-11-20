"""Small geo helpers shared by every module (no heavy GIS dependency needed)."""
import numpy as np

EARTH_R_M = 6_371_000.0
CELL_DEG = 0.00225  # ~250 m grid cell at this latitude


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


def cell_id(lat, lon):
    """Snap a coordinate to a ~250 m grid cell id like 'r5670_c33422'."""
    r = np.floor(np.asarray(lat) / CELL_DEG).astype(int)
    c = np.floor(np.asarray(lon) / CELL_DEG).astype(int)
    if np.ndim(r) == 0:
        return f"r{r}_c{c}"
    return np.char.add(np.char.add("r", r.astype(str)), np.char.add("_c", c.astype(str)))


def cell_center(cid):
    r, c = cid[1:].split("_c")
    return (int(r) + 0.5) * CELL_DEG, (int(c) + 0.5) * CELL_DEG


CELL_AREA_KM2 = (CELL_DEG * 111.32) * (CELL_DEG * 111.32 * np.cos(np.radians(12.76)))
