"""
Existing rooftop solar check for health center sites.

For each site (public HRSA info only: site number, name, address, point):
  1. Building outline: an outline given in the sites file (footprint_wkt, lon/lat WKT), otherwise the
     Overture Maps building that contains the point, otherwise the largest Overture building within 40 m.
  2. Imagery: the newest USDA NAIP aerial photo covering the building (Microsoft Planetary Computer).
  3. Detection: open-source solar panel segmentation model (Kleebauer et al. 2023, DeepLabV3-ResNet101,
     trained on 0.1-3.2 m imagery; weights CC BY 4.0, Zenodo record 10036926). Run on the photo as is
     (~0.6 m) and enlarged 2x (~0.3 m), since panels at 0.6 m are only a few pixels wide.
  4. Per site: panel area on the roof and next to it (carports, ground mounts), share of roof covered,
     model confidence, photo date, has_solar flag. Plus a picture of each site (results/chips/) for
     checking flagged sites by eye.

Usage:  python detect_solar.py sites/test_sites.csv results/test
        (split runs: env SHARD / NSHARDS, then python detect_solar.py combine results/<name>)
Sites file columns: site_id, site_name, address, lat, lon (optional: footprint_wkt, expected_roof_solar).
"""
import math
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import shapely
from shapely import wkb, wkt
from shapely.geometry import Point, mapping

SQFT = 10.7639
NEAR_M = 40            # point not inside a building: largest building within this distance
NEARBY_M = 30          # "next to the building" ring for carports / ground mounts
MARGIN_M = 35          # photo crop margin around the building
PROB = 0.5             # pixel counts as panel at this model probability
MIN_ROOF_M2 = float(os.environ.get("MIN_ROOF_M2", "15"))   # ~8 panels; has_solar needs at least this on the roof
MIN_ROOF_SHARE = float(os.environ.get("MIN_ROOF_SHARE", "0.02"))
WEIGHTS_URL = "https://zenodo.org/records/10036926/files/PV-Segmentation-deeplabv3.pt?download=1"
WEIGHTS = os.environ.get("WEIGHTS_PATH", "weights/PV-Segmentation-deeplabv3.pt")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
t0 = time.time()


def reproject(geom, transformer):
    """Shapely geometry through a pyproj transform function (always_xy)."""
    return shapely.transform(geom, lambda c: np.column_stack(transformer(c[:, 0], c[:, 1])))


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- building outlines (Overture)
def overture_buildings(points):
    """{site_id: (building polygon in lon/lat, rule, source, [other buildings nearby])} for (site_id, lat, lon)."""
    import duckdb
    with urllib.request.urlopen("https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/"
                                "?list-type=2&prefix=release/&delimiter=/", timeout=60) as r:
        rel = sorted(re.findall(r"<Prefix>release/([^/<]+)/</Prefix>", r.read().decode()))[-1]
    con = duckdb.connect()
    for s in ["INSTALL httpfs", "LOAD httpfs", "SET s3_region='us-west-2'",
              "SET enable_object_cache=true", "SET enable_geoparquet_conversion=false"]:
        try:
            con.execute(s)
        except duckdb.Error:
            pass
    files = [f for (f,) in con.execute(
        f"SELECT file FROM glob('s3://overturemaps-us-west-2/release/{rel}/theme=buildings/type=building/*')").fetchall()]
    meta = con.execute(f"SELECT file_name, path_in_schema, stats_min, stats_max FROM parquet_metadata({files!r})").df()
    meta["col"] = meta.path_in_schema.str.replace(" ", "")
    ext = {}
    for f, g in meta[meta.col.isin(["bbox,xmin", "bbox,xmax", "bbox,ymin", "bbox,ymax"])].groupby("file_name"):
        v = lambda c, s: pd.to_numeric(g.loc[g.col == c, s], errors="coerce")
        ext[f] = (v("bbox,xmin", "stats_min").min(), v("bbox,xmax", "stats_max").max(),
                  v("bbox,ymin", "stats_min").min(), v("bbox,ymax", "stats_max").max())
    log(f"Overture release {rel}: {len(files)} building files")
    out, d = {}, 0.0012   # ~130 m box: the site building plus its neighbours
    for sid, lat, lon in points:
        cand = [f for f, (x0, x1, y0, y1) in ext.items()
                if x0 <= lon + d and x1 >= lon - d and y0 <= lat + d and y1 >= lat - d]
        if not cand:
            continue
        rows = con.execute(f"""SELECT geometry, sources[1].dataset FROM read_parquet({cand!r})
                 WHERE bbox.xmin < {lon + d} AND bbox.xmax > {lon - d} AND bbox.ymin < {lat + d} AND bbox.ymax > {lat - d}""").fetchall()
        kx, ky = 111320 * math.cos(math.radians(lat)), 110540
        bs = []
        for g, src in rows:
            try:
                geom = wkb.loads(bytes(g))
            except Exception:
                continue
            local = shapely.transform(geom, lambda c: (c - [lon, lat]) * [kx, ky])
            bs.append((geom, src, local.area, local.distance(Point(0, 0)), local.contains(Point(0, 0))))
        cont = sorted([b for b in bs if b[4]], key=lambda b: -b[2])
        near = sorted([b for b in bs if b[3] <= NEAR_M], key=lambda b: -b[2])
        pick = (cont[0], "Overture: contains point") if cont else (
            (near[0], f"Overture: largest within {NEAR_M} m") if near else (None, None))
        others = [b[0] for b in bs if pick[0] is None or b is not pick[0]]
        out[sid] = (pick[0][0], pick[1], pick[0][1], others) if pick[0] else (None, None, None, others)
    return out


# ---------------------------------------------------------------- NAIP photo
def naip_crop(catalog, geom_ll):
    """Newest NAIP photo over the building: (rgb uint8 HxWx3, window transform, crs, date, gsd_m, item id)."""
    import planetary_computer
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    items = list(catalog.search(collections=["naip"], intersects=mapping(geom_ll.centroid)).items())
    if not items:
        return None
    items.sort(key=lambda it: it.datetime, reverse=True)
    for it in items[:3]:                      # newest first; skip a photo that does not cover the whole crop
        href = planetary_computer.sign(it.assets["image"].href)
        with rasterio.open(href) as src:
            to_r = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
            g = reproject(geom_ll, to_r)
            x0, y0, x1, y1 = g.buffer(MARGIN_M).bounds
            sb = src.bounds
            if x0 < sb.left or x1 > sb.right or y0 < sb.bottom or y1 > sb.top:
                continue
            win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            rgb = src.read([1, 2, 3], window=win)
            return (np.moveaxis(rgb, 0, -1), src.window_transform(win), src.crs,
                    it.datetime.date().isoformat(), abs(src.transform.a), it.id)
    return None


# ---------------------------------------------------------------- model
def load_model():
    import torch
    import torchvision
    if not os.path.exists(WEIGHTS):
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        log("downloading model weights (245 MB, Zenodo 10036926)")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS + ".part")
        os.replace(WEIGHTS + ".part", WEIGHTS)
    obj = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:                                     # a state dict: rebuild the network and load it
        sd = obj.get("state_dict", obj.get("model_state_dict", obj)) if isinstance(obj, dict) else obj
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        aux = any(k.startswith("aux_classifier") for k in sd)
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None, weights_backbone=None, num_classes=1, aux_loss=aux)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"state dict loaded; missing {len(missing)}, unexpected {len(unexpected)}")
    model.eval()
    torch.set_num_threads(os.cpu_count() or 2)
    log(f"model ready ({type(model).__name__}, torch {torch.__version__}, torchvision {torchvision.__version__})")
    return model


def predict(model, rgb, scale):
    """Panel probability per photo pixel. scale=2 enlarges the photo 2x before the model."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)   # same as ToTensor()
    h, w = x.shape[-2:]
    if scale != 1:
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    H, W = x.shape[-2:]
    ph, pw = (-H) % 16, (-W) % 16
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        out = model(x)
        out = out["out"] if isinstance(out, dict) else out
    p = torch.sigmoid(out[:, :1, :H, :W])
    if scale != 1:
        p = F.interpolate(p, size=(h, w), mode="area")
    return p[0, 0].numpy()


# ---------------------------------------------------------------- per-site numbers and picture
def site_numbers(prob, roof_mask, near_mask, gsd):
    px = gsd * gsd
    hit = prob >= PROB
    roof_px = int(roof_mask.sum())
    roof_hit, near_hit = hit & roof_mask, hit & near_mask
    return dict(roof_pv_m2=round(roof_hit.sum() * px, 1),
                roof_pv_share=round(roof_hit.sum() / roof_px, 4) if roof_px else np.nan,
                roof_pv_conf=round(float(prob[roof_hit].mean()), 3) if roof_hit.any() else np.nan,
                nearby_pv_m2=round(near_hit.sum() * px, 1),
                roof_m2=round(roof_px * px, 1))


def save_chip(path, title, rgb, prob1, prob2, roof_ring, near_ring):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for a, (p, name) in zip(ax, [(None, "NAIP photo"), (prob1, "panels, photo as is"), (prob2, "panels, photo 2x")]):
        a.imshow(rgb)
        if p is not None:
            a.imshow(np.ma.masked_less(p, PROB), cmap="autumn", vmin=0, vmax=1, alpha=0.6)
        for ring, c in [(roof_ring, "cyan"), (near_ring, "white")]:
            for xs, ys in ring:
                a.plot(xs, ys, color=c, lw=1.2, ls="-" if c == "cyan" else ":")
        a.set_title(name)
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def pixel_rings(geom, inv):
    gs = getattr(geom, "geoms", [geom])
    rings = []
    for g in gs:
        if g.is_empty:
            continue
        xs, ys = g.exterior.coords.xy
        cols, rows = zip(*[inv * (x, y) for x, y in zip(xs, ys)])
        rings.append((cols, rows))
    return rings


def score(out):
    """Test summary when the sites file has expected_roof_solar (yes / no; other values not scored)."""
    if "expected_roof_solar" not in out:
        return ""
    e = out.expected_roof_solar.astype(str).str.strip().str.lower()
    ok = out[(out.status == "ok") & e.isin(["yes", "no"])]
    y = e[ok.index].eq("yes")
    lines = [f"Scored sites: {len(ok)} ({int(y.sum())} with solar, {int((~y).sum())} without); "
             f"not scored: {int((out.status != 'ok').sum())} without a result, {int((~e.isin(['yes', 'no'])).sum())} unlabeled"]
    rules = {"roof only (has_solar)": ok.has_solar.astype(bool),
             "roof or nearby (has_solar or solar_nearby)": ok.has_solar.astype(bool) | ok.solar_nearby.astype(bool)}
    for name, f in rules.items():
        tp, fp, fn, tn = int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum()), int((~f & ~y).sum())
        lines.append(f"{name}: found {tp} of {tp + fn} solar sites (recall {tp / max(tp + fn, 1):.0%}); "
                     f"{fp} false alarms among {fp + tn} no-solar sites; precision {tp / max(tp + fp, 1):.0%}")
    return "\n".join(lines)


def main(sites_csv, out_prefix):
    import pystac_client
    from pyproj import Transformer
    from rasterio.features import rasterize

    sites = pd.read_csv(sites_csv, dtype={"site_id": str})
    shard, nshards = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    sites = sites.iloc[shard::nshards].reset_index(drop=True)
    log(f"{len(sites)} sites from {sites_csv} (part {shard + 1} of {nshards})")
    # pictures: every site for small lists; for big lists only flagged sites and test misses
    all_chips = os.environ.get("CHIPS", "auto") == "all" or (os.environ.get("CHIPS", "auto") == "auto" and len(sites) * nshards <= 60)
    name = os.path.basename(out_prefix)
    chips = os.path.join(os.path.dirname(out_prefix) or ".", "chips", name)
    os.makedirs(chips, exist_ok=True)

    bldg = overture_buildings([(r.site_id, r.lat, r.lon) for r in sites.itertuples()])
    if "footprint_wkt" in sites:   # a given outline replaces Overture's pick; Overture neighbours are kept
        for r in sites.dropna(subset=["footprint_wkt"]).itertuples():
            g = wkt.loads(r.footprint_wkt)
            others = [o for o in bldg.get(r.site_id, (None,) * 4)[3] or [] if not o.intersects(g)]
            bldg[r.site_id] = (g, "given outline", "sites file", others)
    log(f"building outlines: {sum(1 for v in bldg.values() if v[0] is not None)} of {len(sites)}")

    catalog = pystac_client.Client.open(STAC)
    model = load_model()
    recs = []
    for i, r in enumerate(sites.itertuples(), 1):
        rec = dict(site_id=r.site_id, site_name=r.site_name, address=r.address, lat=r.lat, lon=r.lon)
        expected = str(getattr(r, "expected_roof_solar", "")).strip().lower()
        try:
            geom_ll, rule, src, others = bldg.get(r.site_id, (None, None, None, []))
            if geom_ll is None:   # no outline: a 15 m circle around the point, flagged as such
                kx = 111320 * math.cos(math.radians(r.lat))
                geom_ll = shapely.transform(Point(0, 0).buffer(15), lambda c: c / [kx, 110540] + [r.lon, r.lat])
                rule, src = "no building: 15 m circle", ""
            rec.update(building_rule=rule, building_source=src)
            got = naip_crop(catalog, geom_ll)
            if got is None:
                rec.update(status="no NAIP photo")
                recs.append(rec)
                continue
            rgb, tr, crs, date, gsd, item = got
            to_r = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
            g = reproject(geom_ll, to_r)
            # nearby ring: open ground around the building (parking, carports, ground mounts); other
            # buildings' roofs are left out, since panels there are most likely a neighbour's
            other_r = shapely.union_all([reproject(o, to_r) for o in others]) if others else None
            near = g.buffer(NEARBY_M).difference(g)
            if other_r is not None and not other_r.is_empty:
                near = near.difference(other_r.difference(g))
            shape = rgb.shape[:2]
            roof_mask = rasterize([g], out_shape=shape, transform=tr).astype(bool)
            near_mask = rasterize([near], out_shape=shape, transform=tr).astype(bool) if not near.is_empty else np.zeros(shape, bool)
            p1, p2 = predict(model, rgb, 1), predict(model, rgb, 2)
            n1, n2 = site_numbers(p1, roof_mask, near_mask, gsd), site_numbers(p2, roof_mask, near_mask, gsd)
            rec.update(naip_date=date, naip_item=item, naip_gsd_m=round(gsd, 2), roof_sqft=round(n1["roof_m2"] * SQFT))
            for k in ["roof_pv_m2", "roof_pv_share", "roof_pv_conf", "nearby_pv_m2"]:
                rec[f"{k}_1x"], rec[f"{k}_2x"] = n1[k], n2[k]
            best = max((n1, n2), key=lambda n: n["roof_pv_m2"])
            rec["has_solar"] = bool(best["roof_pv_m2"] >= MIN_ROOF_M2 and best["roof_pv_share"] >= MIN_ROOF_SHARE)
            rec["solar_nearby"] = bool(max(n1["nearby_pv_m2"], n2["nearby_pv_m2"]) >= 4 * MIN_ROOF_M2)
            rec["status"] = "ok"
            miss = expected in ("yes", "no") and rec["has_solar"] != (expected == "yes")
            if all_chips or rec["has_solar"] or rec["solar_nearby"] or miss or expected == "yes":
                inv = ~tr
                fn = re.sub(r"[^A-Za-z0-9_-]", "_", r.site_id) + ".png"
                save_chip(os.path.join(chips, fn),
                          f"{r.site_id}  {str(r.site_name)[:55]}  |  NAIP {date}  |  has_solar={rec['has_solar']}"
                          + (f" (expected {expected})" if expected in ("yes", "no") else "")
                          + f"  roof panels {rec['roof_pv_m2_1x']}/{rec['roof_pv_m2_2x']} m2 (1x/2x)",
                          rgb, p1, p2, pixel_rings(g, inv), pixel_rings(g.buffer(NEARBY_M), inv))
                rec["chip"] = f"chips/{name}/{fn}"
            log(f"[{i}/{len(sites)}] {r.site_id}: NAIP {date} ({gsd:.2f} m), roof {rec['roof_sqft']} sqft, roof panels "
                f"{n1['roof_pv_m2']}/{n2['roof_pv_m2']} m2, nearby {n1['nearby_pv_m2']}/{n2['nearby_pv_m2']} m2 "
                f"-> has_solar={rec['has_solar']}" + (f" (expected {expected})" if expected else ""))
        except Exception as e:
            rec["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
            log(f"[{i}/{len(sites)}] {r.site_id}: {rec['status']}")
        recs.append(rec)

    out = pd.DataFrame(recs)
    extra = [c for c in sites.columns if c not in out.columns and c not in ("footprint_wkt",)]
    out = out.merge(sites[["site_id"] + extra], on="site_id", how="left")
    if "expected_roof_solar" in out:
        e = out.expected_roof_solar.astype(str).str.strip().str.lower()
        out["matches_expected"] = [(bool(h) == (x == "yes")) if st == "ok" and x in ("yes", "no") else None
                                   for h, x, st in zip(out.get("has_solar", [None] * len(out)), e, out.status)]
    path = out_prefix + (f"_part{shard}.csv" if nshards > 1 else "_results.csv")
    out.to_csv(path, index=False)
    log(f"wrote {path}")
    if nshards == 1 and (summary := score(out)):
        open(out_prefix + "_summary.txt", "w").write(summary + "\n")
        log("\n" + summary)


def combine(out_prefix):
    """Merge the parts of a split run into one results file and summary."""
    import glob
    parts = sorted(glob.glob(out_prefix + "_part*.csv"))
    out = pd.concat([pd.read_csv(p, dtype={"site_id": str}) for p in parts], ignore_index=True)
    out.to_csv(out_prefix + "_results.csv", index=False)
    for p in parts:
        os.remove(p)
    summary = score(out)
    if summary:
        open(out_prefix + "_summary.txt", "w").write(summary + "\n")
    log(f"combined {len(parts)} parts, {len(out)} sites\n{summary}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "combine":
        combine(sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "sites/test_sites.csv",
             sys.argv[2] if len(sys.argv) > 2 else "results/test")
