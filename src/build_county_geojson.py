r"""
Build app/data/counties_va_nc_md.geojson: simplified county boundaries for the app's map (Milestone 15).

Source: U.S. Census Bureau, 2023 Cartographic Boundary File, counties, 1:5,000,000 scale
(cb_2023_us_county_5m), free and public:
    https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip
The download goes to data/raw/census_boundaries/ (not in Git). Cartographic boundary files are
already simplified for small-scale maps; this script keeps only Virginia (51), North Carolina (37)
and Maryland (24), rounds coordinates to 3 decimal places (about 100 m, invisible at map scale),
drops points that become duplicates and tiny islands, and writes GeoJSON with each feature's
"id" = the 5-digit county FIPS code, so the app can join it to its county table.

Run from the project folder (needs pyshp, a small pure-Python shapefile reader: dev-only, the app
itself doesn't need it):
    .venv\Scripts\python.exe src\build_county_geojson.py
"""

import io
import json
import zipfile
from pathlib import Path

import shapefile  # pyshp

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "raw" / "census_boundaries" / "cb_2023_us_county_5m.zip"
OUT = ROOT / "app" / "data" / "counties_va_nc_md.geojson"
STATES = {"51", "37", "24"}
DECIMALS = 3
MIN_RING_POINTS = 4          # a closed ring needs at least 4 points (first = last)


def ring_area(r):
    """Shoelace area in squared degrees (only used to drop specks)."""
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(r, r[1:]))) / 2


def clean_ring(ring):
    out = []
    for x, y in ring:
        p = (round(x, DECIMALS), round(y, DECIMALS))
        if not out or out[-1] != p:
            out.append(p)
    if out and out[0] != out[-1]:
        out.append(out[0])
    return out if len(out) >= MIN_RING_POINTS else None


def polygons(shape):
    """pyshp gives one list of points with 'parts'; split into rings, then group into polygons
    (an outer ring followed by its holes; in shapefiles outer rings run clockwise)."""
    pts, parts = shape.points, list(shape.parts) + [len(shape.points)]
    polys = []
    for a, b in zip(parts[:-1], parts[1:]):
        ring = clean_ring(pts[a:b])
        if ring is None:
            continue
        signed = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:]))
        if signed < 0 or not polys:          # clockwise = outer ring
            if ring_area(ring) < 1e-5 and polys:
                continue                     # a speck of an island: invisible on this map
            polys.append([ring])
        else:
            polys[-1].append(ring)
    return polys


def main():
    z = zipfile.ZipFile(ZIP)
    base = "cb_2023_us_county_5m"
    r = shapefile.Reader(shp=io.BytesIO(z.read(f"{base}.shp")), dbf=io.BytesIO(z.read(f"{base}.dbf")),
                         shx=io.BytesIO(z.read(f"{base}.shx")))
    feats = []
    for sr in r.iterShapeRecords():
        rec = sr.record.as_dict()
        if rec["STATEFP"] not in STATES:
            continue
        polys = polygons(sr.shape)
        if not polys:
            continue
        geom = ({"type": "Polygon", "coordinates": [[list(p) for p in ring] for ring in polys[0]]}
                if len(polys) == 1 else
                {"type": "MultiPolygon", "coordinates": [[[list(p) for p in ring] for ring in poly] for poly in polys]})
        feats.append({"type": "Feature", "id": rec["GEOID"], "properties": {"name": rec["NAMELSAD"]},
                      "geometry": geom})
    feats.sort(key=lambda f: f["id"])
    gj = {"type": "FeatureCollection",
          "source": "U.S. Census Bureau, 2023 Cartographic Boundary File, counties 1:5,000,000 (cb_2023_us_county_5m), "
                    "simplified (coordinates rounded to 3 decimals)",
          "features": feats}
    OUT.write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")
    by_state = {s: sum(f["id"].startswith(s) for f in feats) for s in sorted(STATES)}
    print(f"wrote {OUT.relative_to(ROOT)}: {len(feats)} counties {by_state}, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
