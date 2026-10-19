"""Build 1 cm field maps (FIELD frame, meters) from the tessellated field triangles.

Memory-safe rewrite: triangles are point-sampled (barycentric grid, <=7 mm spacing) in
batches capped by a point budget (not a triangle count), so huge triangles or the 1.6 M hub
triangles never blow up memory.

Outputs (in /workspace/frc/field/):
  field_maps.npz : floor_z, clearance, obstacle_category, zone, floor_slope + metadata
  regions.json   : human readable summary
Run:  .venv/bin/python tools/build_maps.py
"""
import json, sys, time, numpy as np
from scipy import ndimage

D = "/workspace/frc/field"
HALF_L, HALF_W = 8.270, 4.035            # inner wall faces, 651.2 in x 317.7 in
RES = 0.01
STEP = 0.007                              # sample spacing (m) < cell size
PT_BUDGET = 3_000_000                     # max sample points per batch
REACH_CLEAR = 0.25                        # flood-fill passability threshold (m)
LOW_PLATE = 0.035                         # obstacle geometry <= floor+3.5 cm is treated as drivable plate
FLOOR_CATS = {'bump', 'tower_base', 'depot'}
SKIP = {'carpet', 'tape', 'fuel'}
CATS = ['none', 'hub', 'tower', 'trench', 'outpost', 'perimeter', 'scoring_table', 'other']
ZONES = ['blue_alliance', 'red_alliance', 'neutral', 'barrier_band_blue', 'barrier_band_red']

# cad (mm) -> field (m): 180 deg about Z + shift
CAD_TO_FIELD = np.array([[-1e-3, 0, 0, HALF_L],
                         [0, -1e-3, 0, HALF_W],
                         [0, 0, 1e-3, 0.0],
                         [0, 0, 0, 1.0]])

def to_field(p):
    q = np.empty(p.shape, np.float64)
    q[..., 0] = HALF_L - p[..., 0] / 1000.0
    q[..., 1] = HALF_W - p[..., 1] / 1000.0
    q[..., 2] = p[..., 2] / 1000.0
    return q

_bary = {}
def bary(k):
    if k not in _bary:
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij'); m = i + j <= k
        _bary[k] = ((i[m] / k)[:, None], (j[m] / k)[:, None])
    return _bary[k]

def sample_batches(tris):
    """yield (N,3) point arrays, each <= ~PT_BUDGET points"""
    e = np.linalg.norm(tris[:, [1, 2, 0]] - tris, axis=2).max(1)
    k = np.clip(np.ceil(e / STEP).astype(np.int64), 1, 4000)
    for kk in np.unique(k):
        idx = np.nonzero(k == kk)[0]
        a, b = bary(int(kk)) if kk <= 400 else (None, None)
        npt = (kk + 1) * (kk + 2) // 2
        if kk <= 400:
            per = max(1, PT_BUDGET // npt)
            for s in range(0, len(idx), per):
                t = tris[idx[s:s + per]]
                p = t[:, None, 0] + (t[:, None, 1] - t[:, None, 0]) * a[None] + (t[:, None, 2] - t[:, None, 0]) * b[None]
                yield p.reshape(-1, 3)
        else:   # gigantic triangle: sample row by row
            for ti in idx:
                t = tris[ti]
                for i in range(0, kk + 1, 50):
                    ii = np.arange(i, min(i + 50, kk + 1))
                    I, J = np.meshgrid(ii, np.arange(kk + 1), indexing='ij'); m = I + J <= kk
                    aa = (I[m] / kk)[:, None]; bb = (J[m] / kk)[:, None]
                    yield t[0] + (t[1] - t[0]) * aa + (t[2] - t[0]) * bb

def main():
    t0 = time.time()
    meta = json.load(open(f"{D}/field_tris_meta.json")); Z = np.load(f"{D}/field_tris_cad_mm.npz")
    nx, ny = int(round(2 * HALF_L / RES)), int(round(2 * HALF_W / RES))
    floor = np.zeros(nx * ny, np.float32)                 # carpet = 0
    obs = np.full(nx * ny, np.inf, np.float32)            # lowest obstacle underside (abs z)
    obs_cat = np.zeros(nx * ny, np.int8)

    def cells(p):
        ix = np.floor(p[:, 0] / RES).astype(np.int64); iy = np.floor(p[:, 1] / RES).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return ix[ok] * ny + iy[ok], p[ok, 2].astype(np.float32)

    order = sorted(meta, key=lambda m: m['category'] not in FLOOR_CATS)   # floor pass first
    for m in order:
        cat = m['category']
        if cat in SKIP: continue
        tris = Z[m['key']]
        npts = 0
        for s in range(0, len(tris), 50000):
            tf = to_field(tris[s:s + 50000].astype(np.float64))
            # quick reject: triangles fully outside field XY
            mn = tf[..., :2].min(1); mx = tf[..., :2].max(1)
            keep = (mx[:, 0] >= 0) & (mn[:, 0] <= 2 * HALF_L) & (mx[:, 1] >= 0) & (mn[:, 1] <= 2 * HALF_W)
            tf = tf[keep]
            if not len(tf): continue
            for p in sample_batches(tf):
                lin, z = cells(p); npts += len(z)
                if not len(z): continue
                if cat in FLOOR_CATS:
                    np.maximum.at(floor, lin, z)
                else:
                    low = z <= floor[lin] + LOW_PLATE
                    if low.any(): np.maximum.at(floor, lin[low], z[low])
                    lin, z = lin[~low], z[~low]
                    if not len(z): continue
                    # per-batch min via sort, then merge (keeps category of lowest point)
                    o = np.lexsort((z, lin)); lin, z = lin[o], z[o]
                    first = np.r_[True, lin[1:] != lin[:-1]]
                    lin, z = lin[first], z[first]
                    better = z < obs[lin]
                    obs[lin[better]] = z[better]
                    obs_cat[lin[better]] = CATS.index(cat) if cat in CATS else CATS.index('other')
        print(f"{time.time()-t0:7.1f}s {m['key']:>4} {cat:14s} tris={m['tris']:8d} pts={npts}", file=sys.stderr, flush=True)

    floor = floor.reshape(nx, ny); obs = obs.reshape(nx, ny); obs_cat = obs_cat.reshape(nx, ny)
    # close sampling pinholes conservatively: obstacles grow 1 cell, raised floor grows 1 cell
    obs_f = ndimage.minimum_filter(np.where(np.isfinite(obs), obs, 1e9), size=3)
    cat_f = ndimage.maximum_filter(obs_cat, size=3)  # only used where obs was empty
    newly = ~np.isfinite(obs) & (obs_f < 1e8)
    obs_cat = np.where(newly, cat_f, obs_cat).astype(np.int8)
    obs = np.where(obs_f < 1e8, obs_f, np.inf).astype(np.float32)
    floor = ndimage.maximum_filter(floor, size=3).astype(np.float32)
    clear = np.maximum(obs - floor, 0).astype(np.float32)        # inf where open
    obs_cat[~np.isfinite(clear)] = 0
    # reachability: cells enclosed by walls (e.g. hub interior funnel) are not reachable from the open
    # carpet -> mark blocked. Passable for flood fill = clearance > REACH_CLEAR (lowest plausible robot).
    lab, n = ndimage.label(clear > REACH_CLEAR)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    main_lab = 1 + int(np.argmax(sizes))
    reachable = lab == main_lab
    enclosed = ~reachable & (clear > REACH_CLEAR)
    print(f"components={n}, enclosed cells set blocked={int(enclosed.sum())}", file=sys.stderr)
    clear[enclosed] = 0.0

    # slope magnitude (rise/run) of floor, smoothed over 3 cm
    fs = ndimage.uniform_filter(floor.astype(np.float64), 3)
    gx, gy = np.gradient(fs, RES)
    slope = np.hypot(gx, gy).astype(np.float32)

    # zones (field frame, along X). Alliance zone 158.6 in from each wall; barrier band = bump/trench/hub band
    L = 2 * HALF_L; AZ = 4.028
    BB_IN, BB_OUT = HALF_L - 4.258, HALF_L - 3.048           # 4.012 .. 5.222 (blue side)
    x = (np.arange(nx) + 0.5) * RES
    zone = np.full((nx, ny), 2, np.int8)
    zone[x < AZ, :] = 0
    zone[x > L - AZ, :] = 1
    zone[(x >= AZ) & (x <= BB_OUT), :] = 3
    zone[(x <= L - AZ) & (x >= L - BB_OUT), :] = 4

    np.savez_compressed(f"{D}/field_maps.npz", floor_z=floor, clearance=clear, obstacle_category=obs_cat,
                        zone=zone, floor_slope=slope, reachable=reachable, res=np.float32(RES), size=np.array([L, 2 * HALF_W]),
                        origin=np.array([0.0, 0.0]), cad_to_field=CAD_TO_FIELD,
                        obstacle_category_names=np.array(CATS), zone_names=np.array(ZONES),
                        index_convention=np.array("array[ix, iy]; cell (ix,iy) covers x in [ix*res,(ix+1)*res), y in [iy*res,(iy+1)*res)"))
    print(f"saved maps {time.time()-t0:.1f}s; obstructed frac {np.isfinite(clear).mean():.3f}", file=sys.stderr)
    write_regions(floor, clear, obs_cat, zone, L, AZ, BB_IN, BB_OUT)

def bbox_cad_to_field(b):
    xs = sorted([HALF_L - b[0] / 1000, HALF_L - b[3] / 1000]); ys = sorted([HALF_W - b[1] / 1000, HALF_W - b[4] / 1000])
    return {"x": [round(xs[0], 4), round(xs[1], 4)], "y": [round(ys[0], 4), round(ys[1], 4)],
            "z": [round(b[2] / 1000, 4), round(b[5] / 1000, 4)]}

def write_regions(floor, clear, obs_cat, zone, L, AZ, BB_IN, BB_OUT):
    tree = json.load(open("/workspace/frc/tools/field_tree.json"))
    structs = []
    for e in tree:
        if e['depth'] != 1: continue
        n = e['name']
        for key in ['Hub', 'Bump', 'Trench', 'Tower', 'Outpost', 'Depot', 'SCORING TABLE', 'Welded Perimeter']:
            if key in n:
                f = bbox_cad_to_field(e['bbox']); cx = (f['x'][0] + f['x'][1]) / 2
                side = 'red' if cx > L / 2 else 'blue'
                if 'SCORING' in key or 'Perimeter' in key: side = 'n/a'
                structs.append({"name": n, "type": key.lower().replace('welded ', ''), "alliance_side": side, "bbox_field_m": f})
    fuel = []
    for e in tree:
        if e['depth'] == 1 and 'GE-26900: Fuel' in e['name']:
            b = e['bbox']; c = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2]
            x, y, z = HALF_L - c[0] / 1000, HALF_W - c[1] / 1000, c[2] / 1000
            inf = bool(0 < x < L and 0 < y < 2 * HALF_W)
            loc = 'off_field_outpost'
            if inf:
                loc = 'depot' if (x < 0.75 or x > L - 0.75) else ('neutral_pile' if abs(x - L / 2) < 2.5 else 'other')
            fuel.append({"x": round(x, 4), "y": round(y, 4), "z": round(z, 4), "in_field": inf, "location": loc})
    from collections import Counter
    R = {
        "frame": {"units": "m", "up": "+Z",
                  "origin": "blue alliance wall, right-hand corner as seen by blue drivers",
                  "axes": "+X toward red alliance wall, +Y to the left (blue drivers' view)",
                  "cad_to_field_4x4_mm_to_m": CAD_TO_FIELD.tolist(),
                  "formula": "x_f=(8270-x_cad)/1000, y_f=(4035-y_cad)/1000, z_f=z_cad/1000"},
        "field_size_m": [L, 2 * HALF_W],
        "map": {"file": "field_maps.npz", "res_m": RES, "shape": list(floor.shape),
                "layers": {"floor_z": "drivable surface height (m) above carpet",
                           "clearance": "vertical clearance (m) from floor_z to lowest obstacle underside; inf = open",
                           "obstacle_category": "index into obstacle_category_names (category of lowest obstacle in cell)",
                           "zone": "index into zone_names",
                           "floor_slope": "|grad floor_z| (rise/run)",
                           "reachable": "bool: connected (4-nbr) to the main open carpet through cells with clearance > 0.25 m; enclosed pockets (hub interior) are set to clearance 0"}},
        "zones_field_m": {
            "blue_alliance": {"x": [0, AZ], "y": [0, 2 * HALF_W]},
            "barrier_band_blue": {"x": [AZ, BB_OUT], "y": [0, 2 * HALF_W], "note": "bumps, trenches, hub; hub itself spans x 4.012..5.498"},
            "neutral": {"x": [BB_OUT, L - BB_OUT], "y": [0, 2 * HALF_W]},
            "barrier_band_red": {"x": [L - BB_OUT, L - AZ], "y": [0, 2 * HALF_W]},
            "red_alliance": {"x": [L - AZ, L], "y": [0, 2 * HALF_W]}},
        "hub_centers_field_m": {"blue": [round(HALF_L - 3.515, 4), HALF_W, 1.83], "red": [round(HALF_L + 3.515, 4), HALF_W, 1.83],
                                "note": "z = approx top opening height used for shot targeting"},
        "trench_rule": {"clearance_m": 0.5652, "clearance_in": 22.25, "opening_width_m": 1.279,
                        "rule": "robot may occupy a cell only if clearance > robot_height + 0.02 m; so only robots <= ~0.545 m fit under trench arms"},
        "structures": structs,
        "fuel_staging": {"diameter_m": 0.150, "count_total": len(fuel), "count_by_location": dict(Counter(f['location'] for f in fuel)),
                         "balls": fuel},
        "map_stats": {"obstructed_fraction": float(np.isfinite(clear).mean()),
                      "trench_clearance_median_m": float(np.median(clear[(obs_cat == 3) & np.isfinite(clear) & (clear > 0.3)])) if ((obs_cat == 3) & (clear > 0.3)).any() else None,
                      "max_floor_z_m": float(floor.max())},
    }
    json.dump(R, open(f"{D}/regions.json", "w"), indent=1)
    print("regions written", R['map_stats'], R['fuel_staging']['count_by_location'], file=sys.stderr)

if __name__ == "__main__":
    main()
