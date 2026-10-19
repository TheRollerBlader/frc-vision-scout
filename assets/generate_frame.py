#!/usr/bin/env python3
"""FRC 2026 REBUILT synthetic frame randomizer.

Writes one JSON per frame (schema "frc-frame/v1"): 6 robots (3 red, 3 blue) with legal poses
on the field map, randomized ground + airborne fuel. Field frame: meters, Z-up, origin at the
blue alliance wall right-hand corner (blue drivers' view), +X toward red, +Y left.

Usage:
  .venv/bin/python tools/generate_frame.py --n-frames 5 --seed 1 [--plot]
Frame k (0-based) uses seed = --seed + k, so a single frame can be reproduced with
  --seed <frame seed> --n-frames 1
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import shapely
from shapely.geometry import Polygon, box, LineString
from shapely import affinity
from shapely.ops import unary_union
from scipy import ndimage

ROOT = "/workspace/frc"
SCHEMA = "frc-frame/v1"

DEFAULTS = {
    "field_model": f"{ROOT}/models/field/rebuilt_field.step",
    "maps": f"{ROOT}/field/field_maps.npz",
    "regions": f"{ROOT}/field/regions.json",
    "robots": {
        "count_per_alliance": 3,
        "bumper_depth": 0.085,            # m, added around the frame outline
        "height_margin": 0.02,            # clearance must exceed height + margin
        "team_number_range": [1, 11999],
        "zone_weights": {"own": 0.35, "neutral": 0.45, "opponent": 0.15, "barrier": 0.05},
        # when the "barrier" zone is drawn: where in the bump/trench/hub band to aim the robot center
        "barrier_targets": {"trench": 0.35, "bump": 0.45, "any": 0.20},
        "min_robot_gap": 0.005,
        "trench_yaw_sigma_deg": 4.0,
        "zone_redraw_every": 40,          # attempts before the target zone is redrawn           # m between bumper outlines
        "max_plane_rms": 0.03,            # reject pose if RMS residual of floor plane fit > this (m)
        "attempts_per_robot": 600,
        "frame_attempts": 4,              # full re-draws of the robot set if a robot cannot be placed
        "edge_margin": 0.0,
        "placeholder": {"length": [0.55, 0.85], "width": [0.55, 0.85],
                        "short_prob": 0.5, "short_height": [0.45, 0.56], "tall_height": [0.56, 1.15]},
        "velocity": {"stationary_prob": 0.25, "speed_triangular": [0.3, 2.5, 4.5],
                     "swerve_prob": 0.7, "reverse_prob": 0.2, "wz_sigma": 2.0, "wz_clip": 6.0,
                     "bump_dir_sigma_deg": 10.0, "bump_yaw_align_prob": 0.7, "bump_yaw_sigma_deg": 15.0,
                     "wall_lookahead_s": 0.3, "obstacle_lookahead_s": 0.4},
    },
    "phase": {"match_start_prob": 0.10},
    "match_start": {"fuel_jitter": 0.006, "fuel_jitter_max": 0.0012, "include_offfield_staged": False, "robot_moving_prob": 0.0,
                    "line_gap": [0.0, 0.8], "yaw_sigma_deg": 15.0, "yaw_random_cardinal_prob": 0.3},
    "fuel": {
        "keep_prob": 0.6,                 # --fuel-prob: per-candidate keep probability
        "max": 300,                       # --max-fuel: max candidate count
        "candidate_frac": [0.4, 1.0],     # candidates = round(max * U(frac))
        "radius": 0.075, "min_center_dist": 0.15,
        "min_clearance": 0.16, "max_slope": 0.05,
        "mixture": {"cluster": 0.55, "wall": 0.20, "uniform": 0.25},
        "clusters": {"count": [3, 8], "sigma": [0.2, 0.8],
                     "center_weights": {"hub_exit": 0.30, "depot": 0.15, "center_pile": 0.10,
                                        "neutral": 0.30, "alliance": 0.15}},
        "wall": {"piles": [1, 4], "length_sigma": [0.3, 1.5], "offset_sigma": 0.12,
                 "wall_weights": {"guardrail": 0.75, "alliance_wall": 0.25}},
        "nudge_retries": 4, "nudge_sigma": 0.08,
        "rolling_prob": 0.10, "rolling_speed": [0.1, 1.5],
    },
    "airborne": {"shoot_prob": 0.30, "balls": [1, 6], "launch_z": 0.5, "g": 9.81,
                 "hub_target_z": 1.83, "hub_apex_z": [2.4, 3.4], "max_hub_dist": 7.5,
                 "pass_prob": 0.5,       # robots in neutral zone may lob fuel into own alliance zone
                 "pass_apex_z": [2.5, 4.0], "spacing_s": [0.08, 0.25]},
}

# ----------------------------------------------------------------------------------------- utils
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v
    return d

def r4(x): return float(round(float(x), 4)) + 0.0   # +0.0 turns -0.0 into 0.0

class Field:
    def __init__(self, cfg):
        M = np.load(cfg["maps"])
        self.floor = M["floor_z"]; self.clear = M["clearance"]; self.cat = M["obstacle_category"]
        self.zone = M["zone"]; self.slope = M["floor_slope"]; self.reach = M["reachable"]
        self.res = float(M["res"]); self.L, self.W = [float(v) for v in M["size"]]
        self.cad_to_field = M["cad_to_field"].tolist()
        self.cat_names = [str(s) for s in M["obstacle_category_names"]]
        self.zone_names = [str(s) for s in M["zone_names"]]
        self.nx, self.ny = self.floor.shape
        R = json.load(open(cfg["regions"]))
        self.regions = R
        z = R["zones_field_m"]
        self.AZ = z["blue_alliance"]["x"][1]; self.BB = z["barrier_band_blue"]["x"][1]
        self.hub = {"blue": np.array(R["hub_centers_field_m"]["blue"][:2]), "red": np.array(R["hub_centers_field_m"]["red"][:2])}
        self.depots = {}
        for s in R["structures"]:
            if s["type"] == "depot":
                self.depots[s["alliance_side"]] = s["bbox_field_m"]
        self.hub_half = 0.743
        # rectangles (x-range, y-range) used to aim barrier-band robots: trench openings and bump tops
        tw = R["trench_rule"]["opening_width_m"]; self.rects = {"trench": [], "bump": []}
        for s in R["structures"]:
            b = s["bbox_field_m"]
            if s["type"] == "trench":
                self.rects["trench"] += [(tuple(b["x"]), (0.0, tw)), (tuple(b["x"]), (self.W - tw, self.W))]
            elif s["type"] == "bump":
                self.rects["bump"].append((tuple(b["x"]), tuple(b["y"])))
        self.staged = [b for b in R["fuel_staging"]["balls"]]
        f = cfg["fuel"]
        ok = (self.clear > f["min_clearance"]) & (self.slope < f["max_slope"]) & self.reach
        rc = int(math.ceil(f["radius"] / self.res))
        yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1]
        self.fuel_ok = ndimage.binary_erosion(ok, structure=(xx ** 2 + yy ** 2) <= rc * rc, border_value=0)

    def idx(self, x, y):
        return np.floor(np.asarray(x) / self.res).astype(int), np.floor(np.asarray(y) / self.res).astype(int)

    def inside(self, x, y):
        ix, iy = self.idx(x, y)
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def zone_rects(self, name):
        L, W, AZ, BB = self.L, self.W, self.AZ, self.BB
        return {"blue_alliance": [(0, AZ)], "red_alliance": [(L - AZ, L)], "neutral": [(BB, L - BB)],
                "barrier": [(AZ, BB), (L - BB, L - AZ)]}[name]

    def cells_in(self, poly):
        x0, y0, x1, y1 = poly.bounds
        if x0 < 0 or y0 < 0 or x1 > self.L or y1 > self.W: return None
        ix0, iy0 = int(x0 / self.res), int(y0 / self.res)
        ix1, iy1 = min(int(x1 / self.res) + 1, self.nx), min(int(y1 / self.res) + 1, self.ny)
        IX, IY = np.meshgrid(np.arange(ix0, ix1), np.arange(iy0, iy1), indexing="ij")
        xc = (IX + 0.5) * self.res; yc = (IY + 0.5) * self.res
        m = shapely.contains_xy(poly, xc, yc)
        return IX[m], IY[m]

# ----------------------------------------------------------------------------------------- robots
def load_manifest(path):
    """Return list of robot specs (full robots only) or None if no manifest."""
    if not path or not os.path.exists(path): return None
    M = json.load(open(path))
    items = M.get("robots", M) if isinstance(M, dict) else M
    if isinstance(items, dict): items = [dict(v, key=k) if isinstance(v, dict) else None for k, v in items.items()]
    out = []
    def g(e, *names):
        for n in names:
            if n in e and e[n] is not None: return e[n]
        return None
    for e in items:
        if not isinstance(e, dict): continue
        if e.get("partial") is True or e.get("complete") is False or e.get("full_robot") is False or str(e.get("status", "")).lower() == "partial":
            continue
        L = g(e, "frame_length_m", "frame_length", "length_m", "length")
        W = g(e, "frame_width_m", "frame_width", "width_m", "width")
        H = g(e, "height_m", "height", "max_height_m")
        bb = g(e, "bbox_m", "bbox")
        outline = g(e, "frame_outline_m", "frame_outline", "footprint_outline_m", "footprint_m")
        if (L is None or W is None) and outline:
            xs = [p[0] for p in outline]; ys = [p[1] for p in outline]
            L = L or max(xs) - min(xs); W = W or max(ys) - min(ys)
        mn = mx = None
        if isinstance(bb, dict) and "min" in bb: mn, mx = bb["min"], bb["max"]
        elif isinstance(bb, (list, tuple)) and len(bb) == 2 and len(bb[0]) == 3: mn, mx = bb
        elif isinstance(bb, (list, tuple)) and len(bb) == 6: mn, mx = bb[:3], bb[3:]
        if mn is not None and (L is None or W is None or H is None):
            L = L or mx[0] - mn[0]; W = W or mx[1] - mn[1]; H = H or mx[2] - mn[2]
        if L is None or W is None or H is None: continue
        out.append({"key": str(g(e, "key", "name", "id") or "robot"),
                    "model_file": g(e, "model_file", "glb", "file", "path"),
                    "team_source": g(e, "team", "team_number"),
                    "length": float(L), "width": float(W), "height": float(H),
                    "outline": outline, "bumper_depth": g(e, "bumper_depth_m", "bumper_depth")})
    return out

def placeholder_robot(rng, P):
    h = rng.uniform(*P["short_height"]) if rng.random() < P["short_prob"] else rng.uniform(*P["tall_height"])
    return {"key": "placeholder", "model_file": None, "length": float(rng.uniform(*P["length"])),
            "width": float(rng.uniform(*P["width"])), "height": float(h), "outline": None, "bumper_depth": None}

def local_footprint(spec, bumper):
    if spec.get("outline"):
        poly = Polygon(spec["outline"])
        if not poly.is_valid: poly = poly.buffer(0)
    else:
        poly = box(-spec["length"] / 2, -spec["width"] / 2, spec["length"] / 2, spec["width"] / 2)
    return poly.buffer(bumper, join_style=2)   # mitre: keeps rectangle corners square

def place(poly_local, x, y, yaw):
    return affinity.translate(affinity.rotate(poly_local, yaw, origin=(0, 0), use_radians=True), x, y)

def fit_pose(F, cells, x, y, yaw):
    ix, iy = cells
    fz = F.floor[ix, iy].astype(np.float64)
    if np.ptp(fz) < 1e-4:
        return float(fz.mean()), 0.0, 0.0, 0.0, 0.0, 0.0
    px = (ix + 0.5) * F.res - x; py = (iy + 0.5) * F.res - y
    A = np.c_[px, py, np.ones_like(px)]
    (a, b, c), *_ = np.linalg.lstsq(A, fz, rcond=None)
    res = fz - A @ np.array([a, b, c])
    rms = float(np.sqrt(np.mean(res ** 2)))
    z = c + max(0.0, float(res.max()))       # rest on the highest support, never sink into the floor
    sf = a * math.cos(yaw) + b * math.sin(yaw)        # slope along heading
    sl = -a * math.sin(yaw) + b * math.cos(yaw)       # slope to the left
    N = math.sqrt(1 + a * a + b * b)
    pitch = math.atan(-sf); roll = math.asin(max(-1, min(1, sl / N)))
    return z, roll, pitch, rms, a, b

def sample_robot_velocity(rng, V, yaw, on_bump, slope_ab, x, y, F):
    """on_bump=True also used for robots under a trench: velocity mostly along +-X (crossing the band)."""
    if rng.random() < V["stationary_prob"]:
        return [0.0, 0.0, 0.0, 0.0]
    a, b, c = V["speed_triangular"]
    sp = rng.triangular(a, b, c)
    if on_bump:
        d = (0.0 if rng.random() < 0.5 else math.pi) + math.radians(rng.normal(0, V["bump_dir_sigma_deg"]))
    elif rng.random() < V["swerve_prob"]:
        d = rng.uniform(-math.pi, math.pi)
    else:
        d = yaw + (math.pi if rng.random() < V["reverse_prob"] else 0.0)
    vx, vy = sp * math.cos(d), sp * math.sin(d)
    # don't drive straight into the field boundary next instant: flip the offending component
    la = V["wall_lookahead_s"]
    if not (0.3 < x + vx * la < F.L - 0.3): vx = -vx
    if not (0.3 < y + vy * la < F.W - 0.3): vy = -vy
    vz = vx * slope_ab[0] + vy * slope_ab[1]
    wz = float(np.clip(rng.normal(0, V["wz_sigma"]), -V["wz_clip"], V["wz_clip"]))
    return [vx, vy, vz, wz]

def try_place_robot(rng, F, cfg, spec, alliance, placed_polys, phase):
    R = cfg["robots"]; bumper = spec["bumper_depth"] if spec.get("bumper_depth") is not None else R["bumper_depth"]
    poly_local = local_footprint(spec, bumper)
    need = spec["height"] + R["height_margin"]
    zw = R["zone_weights"]; names = ["own", "neutral", "opponent", "barrier"]
    p = np.array([zw[n] for n in names], float); p /= p.sum()
    own = f"{alliance}_alliance"; opp = ("red" if alliance == "blue" else "blue") + "_alliance"
    MS = cfg["match_start"]
    rmax = math.hypot(*np.ptp(np.array(poly_local.exterior.coords), axis=0)) / 2
    stats = {"tries": 0, "outside": 0, "blocked": 0, "overlap": 0, "uneven": 0}
    for _ in range(R["attempts_per_robot"]):
        stats["tries"] += 1
        if phase == "match_start":
            fwd = 0.0 if alliance == "blue" else math.pi
            if rng.random() < MS["yaw_random_cardinal_prob"]: yaw = fwd + rng.integers(0, 4) * math.pi / 2
            else: yaw = fwd + math.radians(rng.normal(0, MS["yaw_sigma_deg"]))
            pl = affinity.rotate(poly_local, yaw, origin=(0, 0), use_radians=True)
            hx = max(abs(pl.bounds[0]), abs(pl.bounds[2]))
            gap = rng.uniform(*MS["line_gap"])
            x = F.AZ - hx - gap if alliance == "blue" else F.L - F.AZ + hx + gap
            y = rng.uniform(0, F.W); zname = own
        else:
            if (stats["tries"] - 1) % R["zone_redraw_every"] == 0:   # sticky zone choice so weights are realized
                zc = names[rng.choice(4, p=p)]
                bt = R["barrier_targets"]; bn = list(bt); bp = np.array([bt[k] for k in bn], float); bp /= bp.sum()
                tgt = bn[rng.choice(len(bn), p=bp)]
            zname = {"own": own, "opponent": opp, "neutral": "neutral", "barrier": "barrier"}[zc]
            rects = F.zone_rects(zname); x0, x1 = rects[rng.integers(len(rects))]
            x = rng.uniform(x0, x1); y = rng.uniform(0, F.W)
            yaw = rng.uniform(-math.pi, math.pi)
            if zc == "barrier":
                if tgt in ("trench", "bump"):
                    rr = F.rects[tgt][rng.integers(len(F.rects[tgt]))]
                    x = rng.uniform(*rr[0]); y = rng.uniform(*rr[1])
                    if tgt == "trench":
                        y = rng.uniform(rr[1][0] + 0.35, rr[1][1] - 0.35)   # driving through the trench: square to the field axes
                        yaw = rng.integers(0, 4) * math.pi / 2 + math.radians(rng.normal(0, R["trench_yaw_sigma_deg"]))
                zname = "barrier:" + tgt
        poly = place(poly_local, x, y, yaw)
        cells = F.cells_in(poly)
        if cells is None or len(cells[0]) == 0: stats["outside"] += 1; continue
        if not np.all(F.clear[cells] > need): stats["blocked"] += 1; continue
        if any(poly.distance(q) < R["min_robot_gap"] for q in placed_polys): stats["overlap"] += 1; continue
        z, roll, pitch, rms, a, b = fit_pose(F, cells, x, y, yaw)
        if rms > R["max_plane_rms"]: stats["uneven"] += 1; continue
        fz = F.floor[cells]
        cc = F.cat[cells]
        under_trench = bool(np.any((cc == F.cat_names.index("trench")) & np.isfinite(F.clear[cells])))
        on_bump = bool(fz.max() > 0.04 and np.any(np.isin(F.zone[cells], (3, 4))))
        # bump crossings: robots mostly square to the bump (optional re-yaw, re-checked)
        if on_bump and phase != "match_start" and rng.random() < R["velocity"]["bump_yaw_align_prob"]:
            yaw2 = (0.0 if rng.random() < 0.5 else math.pi) + math.radians(rng.normal(0, R["velocity"]["bump_yaw_sigma_deg"]))
            poly2 = place(poly_local, x, y, yaw2); cells2 = F.cells_in(poly2)
            if cells2 is not None and np.all(F.clear[cells2] > need) and not any(poly2.distance(q) < R["min_robot_gap"] for q in placed_polys):
                z2, roll2, pitch2, rms2, a2, b2 = fit_pose(F, cells2, x, y, yaw2)
                if rms2 <= R["max_plane_rms"]:
                    yaw, poly, cells, z, roll, pitch, rms, a, b = yaw2, poly2, cells2, z2, roll2, pitch2, rms2, a2, b2

        ixc, iyc = F.idx(x, y)
        zone_name = F.zone_names[int(F.zone[ixc, iyc])]
        if phase == "match_start" and rng.random() >= MS["robot_moving_prob"]:
            vel = [0.0, 0.0, 0.0, 0.0]
        else:
            # resample velocity until the footprint `lookahead` seconds ahead is still legal (no driving into hubs/walls)
            la = R["velocity"]["obstacle_lookahead_s"]
            for _v in range(8):
                vel = sample_robot_velocity(rng, R["velocity"], yaw, on_bump or under_trench, (a, b), x, y, F)
                if vel[0] == 0 and vel[1] == 0: break
                ahead = F.cells_in(affinity.translate(poly, vel[0] * la, vel[1] * la))
                if ahead is not None and len(ahead[0]) and np.all(F.clear[ahead] > need): break
            else:
                vel = [0.0, 0.0, 0.0, 0.0]
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        return {"poly": poly, "x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw, "rms": rms,
                "vel": vel, "zone": zone_name, "on_bump": on_bump, "under_trench": under_trench,
                "bumper": bumper, "sampled_zone": zname}, stats
    return None, stats

# ----------------------------------------------------------------------------------------- fuel
class FuelSet:
    def __init__(self, dmin):
        self.d = dmin; self.grid = {}; self.items = []
    def ok(self, x, y):
        gx, gy = int(x // self.d), int(y // self.d)
        for i in (gx - 1, gx, gx + 1):
            for j in (gy - 1, gy, gy + 1):
                for k in self.grid.get((i, j), ()):
                    px, py = self.items[k][:2]
                    if (px - x) ** 2 + (py - y) ** 2 < self.d * self.d - 1e-9: return False
        return True
    def add(self, x, y, z, **kw):
        self.grid.setdefault((int(x // self.d), int(y // self.d)), []).append(len(self.items))
        self.items.append((x, y, z, kw))

def fuel_candidates(rng, F, cfg, n):
    f = cfg["fuel"]; mix = f["mixture"]; L, W = F.L, F.W
    comp = np.array([mix["cluster"], mix["wall"], mix["uniform"]], float); comp /= comp.sum()
    nc, nw, nu = rng.multinomial(n, comp)
    pts = []
    # gaussian clusters
    C = f["clusters"]; k = int(rng.integers(C["count"][0], C["count"][1] + 1))
    cw = C["center_weights"]; cn = list(cw); cp = np.array([cw[c] for c in cn], float); cp /= cp.sum()
    centers = []
    for _ in range(k):
        t = cn[rng.choice(len(cn), p=cp)]
        if t == "hub_exit":     # neutral-zone side of a hub
            side = rng.choice(["blue", "red"]); hx, hy = F.hub[side]
            s = 1 if side == "blue" else -1
            cx = hx + s * (F.hub_half + rng.uniform(0.3, 1.6)); cy = hy + rng.normal(0, 0.9)
        elif t == "depot":
            side = rng.choice(["blue", "red"]); bb = F.depots[side]
            cx = rng.uniform(*bb["x"]); cy = rng.uniform(*bb["y"])
        elif t == "center_pile":
            cx = L / 2 + rng.normal(0, 0.5); cy = W / 2 + rng.normal(0, 1.2)
        elif t == "neutral":
            cx = rng.uniform(F.BB, L - F.BB); cy = rng.uniform(0, W)
        else:
            x0, x1 = (0, F.AZ) if rng.random() < 0.5 else (L - F.AZ, L)
            cx = rng.uniform(x0, x1); cy = rng.uniform(0, W)
        sig = rng.uniform(*C["sigma"]) * (0.5 if t == "depot" else 1.0)
        centers.append((cx, cy, sig, t))
    if nc:
        wts = rng.dirichlet(np.ones(k))
        for ci in rng.choice(k, size=nc, p=wts):
            cx, cy, sig, _ = centers[ci]
            pts.append((cx + rng.normal(0, sig), cy + rng.normal(0, sig)))
    # piles along walls
    Wc = f["wall"]; npile = int(rng.integers(Wc["piles"][0], Wc["piles"][1] + 1)); piles = []
    for _ in range(npile):
        if rng.random() < Wc["wall_weights"]["guardrail"] / sum(Wc["wall_weights"].values()):
            wall = rng.choice(["y0", "yW"]); along = rng.uniform(0.5, L - 0.5)
        else:
            wall = rng.choice(["x0", "xL"]); along = rng.uniform(0.5, W - 0.5)
        piles.append((wall, along, rng.uniform(*Wc["length_sigma"])))
    for pi in rng.integers(0, npile, size=nw):
        wall, along, ls = piles[pi]
        a = along + rng.normal(0, ls); off = f["radius"] + abs(rng.normal(0, Wc["offset_sigma"]))
        pts.append({"y0": (a, off), "yW": (a, W - off), "x0": (off, a), "xL": (L - off, a)}[wall])
    for _ in range(nu):
        pts.append((rng.uniform(0, L), rng.uniform(0, W)))
    return pts, centers, piles

def ballistic(p0, target, apex_z, g):
    """velocity so a ball from p0 peaks at apex_z and descends through target (x,y,z)."""
    z0 = p0[2]; zt = target[2]
    apex_z = max(apex_z, z0 + 0.2, zt + 0.2)
    vz0 = math.sqrt(2 * g * (apex_z - z0)); T = vz0 / g + math.sqrt(2 * (apex_z - zt) / g)
    dx, dy = target[0] - p0[0], target[1] - p0[1]
    return np.array([dx / T, dy / T, vz0]), T

def airborne_fuel(rng, F, cfg, robots, fs, warn):
    A = cfg["airborne"]; g = A["g"]; out = []
    for r in robots:
        if rng.random() >= A["shoot_prob"] or r["under_trench"]: continue
        al = r["alliance"]; hx, hy = F.hub[al]
        behind = (r["x"] < hx - 0.3) if al == "blue" else (r["x"] > hx + 0.3)   # alliance-wall side of own hub (net blocks the other side)
        dist = math.hypot(r["x"] - hx, r["y"] - hy)
        if behind and dist <= A["max_hub_dist"]:
            kind = "hub_shot"; target = (hx, hy, A["hub_target_z"]); apex = rng.uniform(*A["hub_apex_z"])
        elif r["zone"] == "neutral" and rng.random() < A["pass_prob"]:
            kind = "pass"
            x0, x1 = (0.6, F.AZ - 0.5) if al == "blue" else (F.L - F.AZ + 0.5, F.L - 0.6)
            hubbox = box(hx - F.hub_half - 0.3, hy - F.hub_half - 0.3, hx + F.hub_half + 0.3, hy + F.hub_half + 0.3)
            target = None
            for _t in range(8):   # lob over the bumps/trench, not through the hub net
                ty = rng.uniform(0.6, 2.8) if rng.random() < 0.5 else rng.uniform(F.W - 2.8, F.W - 0.6)
                tx = rng.uniform(x0, x1)
                if not LineString([(r["x"], r["y"]), (tx, ty)]).intersects(hubbox):
                    target = (tx, ty, cfg["fuel"]["radius"]); break
            if target is None: continue
            apex = rng.uniform(*A["pass_apex_z"])
        else:
            continue
        n = int(rng.integers(A["balls"][0], A["balls"][1] + 1))
        rv = np.array(r["velocity"][:2])
        _, Tref = ballistic(np.array([r["x"], r["y"], r["z"] + A["launch_z"]]), target, apex, g)
        t_last = rng.uniform(0.1, 0.92) * Tref
        ts = [t_last - i * rng.uniform(*A["spacing_s"]) for i in range(n)]
        for t in ts:
            if t <= 0.03: continue
            # ball was launched t seconds ago from where the robot was then
            p0 = np.array([r["x"] - rv[0] * t, r["y"] - rv[1] * t, r["z"] + A["launch_z"]])
            v0, T = ballistic(p0, target, apex, g)
            if t >= T: continue
            p = p0 + v0 * t + np.array([0, 0, -0.5 * g * t * t]); v = v0 + np.array([0, 0, -g * t])
            if not (0 < p[0] < F.L and 0 < p[1] < F.W) or p[2] < F.floor[F.idx(p[0], p[1])] + 0.075: continue
            out.append({"x": p[0], "y": p[1], "z": p[2], "state": "airborne", "vx": v[0], "vy": v[1], "vz": v[2],
                        "shooter": r["id"], "shot": kind})
    return out

# ----------------------------------------------------------------------------------------- frame
def make_frame(seed, F, cfg, manifest, PP=None):
    rng = np.random.default_rng(seed); warn = []
    phase = "match_start" if rng.random() < cfg["phase"]["match_start_prob"] else "teleop"
    R = cfg["robots"]; nper = R["count_per_alliance"]
    teams = rng.choice(np.arange(R["team_number_range"][0], R["team_number_range"][1] + 1), size=2 * nper, replace=False)
    alliances = ["red"] * nper + ["blue"] * nper
    robots = None
    for fa in range(R["frame_attempts"]):
        if manifest:
            pick = rng.choice(len(manifest), size=2 * nper, replace=len(manifest) < 2 * nper)
            specs = [manifest[i] for i in pick]
        else:
            specs = [placeholder_robot(rng, R["placeholder"]) for _ in range(2 * nper)]
        placed, polys, fail = [], [], None
        order = rng.permutation(2 * nper)
        for i in order:
            res, st = try_place_robot(rng, F, cfg, specs[i], alliances[i], polys, phase)
            if res is None: fail = (i, st); break
            res["i"] = int(i); placed.append(res); polys.append(res["poly"])
        if fail is None:
            robots = sorted(placed, key=lambda d: d["i"]); break
        warn.append(f"frame attempt {fa}: could not place robot {fail[0]} ({alliances[fail[0]]}, h={specs[fail[0]]['height']:.2f}) stats={fail[1]}")
    if robots is None:
        raise RuntimeError(f"seed {seed}: failed to place robots: {warn}")
    rob_out = []
    cnt = {"red": 0, "blue": 0}
    for r in robots:
        i = r["i"]; s = specs[i]; al = alliances[i]; cnt[al] += 1
        r["alliance"] = al; r["id"] = f"{al}_{cnt[al]}"; r["velocity"] = r["vel"]
        rob_out.append({
            "id": r["id"], "alliance": al, "bumper_color": al, "team_number": int(teams[i]),
            "manifest_key": s["key"], "model_file": s["model_file"] or "placeholder",
            "height": r4(s["height"]),
            "footprint": {"length": r4(s["length"]), "width": r4(s["width"]), "bumper_depth": r4(r["bumper"]),
                          "outline_xy": [[r4(a), r4(b)] for a, b in list(r["poly"].exterior.coords)[:-1]]},
            "pose": {"x": r4(r["x"]), "y": r4(r["y"]), "z": r4(r["z"]), "roll": r4(r["roll"]), "pitch": r4(r["pitch"]), "yaw": r4(r["yaw"])},
            "velocity": {"vx": r4(r["vel"][0]), "vy": r4(r["vel"][1]), "vz": r4(r["vel"][2]), "wz": r4(r["vel"][3])},
            "zone": r["zone"], "on_bump": r["on_bump"], "under_trench": r["under_trench"],
            "floor_fit_rms": r4(r["rms"]),
        })
    # ---- fuel
    f = cfg["fuel"]; fs = FuelSet(f["min_center_dist"])
    excl = unary_union([r["poly"] for r in robots]).buffer(f["radius"])
    stats = {}
    if phase == "match_start":
        MS = cfg["match_start"]; kept = 0
        for b in F.staged:
            if not b["in_field"] and not MS["include_offfield_staged"]: continue
            x0, y0, z0 = b["x"], b["y"], b["z"]
            if b["in_field"]:
                d = rng.normal(0, MS["fuel_jitter"], 2); nd = float(np.hypot(*d))
                if nd > MS["fuel_jitter_max"]: d *= MS["fuel_jitter_max"] / nd   # staged grid is 152.4 mm: keep balls from overlapping
                x, y = x0 + d[0], y0 + d[1]
                if not fs.ok(x, y): x, y = x0, y0
                if shapely.contains_xy(excl, x, y) or not fs.ok(x, y): continue
                z = float(F.floor[F.idx(x, y)]) + f["radius"]
            else:
                x, y, z = x0, y0, z0
            fs.add(x, y, z, state="ground", vel=(0.0, 0.0, 0.0), src="staged_" + b["location"]); kept += 1
        stats = {"staged_used": kept}
    else:
        n = int(round(f["max"] * rng.uniform(*f["candidate_frac"])))
        cands, centers, piles = fuel_candidates(rng, F, cfg, n)
        keep = rng.random(len(cands)) < f["keep_prob"]
        rej = {"invalid_surface": 0, "in_robot": 0, "overlap": 0}
        for (x, y), k in zip(cands, keep):
            if not k: continue
            done = False
            for t in range(f["nudge_retries"] + 1):
                xx, yy = (x, y) if t == 0 else (x + rng.normal(0, f["nudge_sigma"]), y + rng.normal(0, f["nudge_sigma"]))
                if not F.inside(xx, yy) or not F.fuel_ok[F.idx(xx, yy)]: why = "invalid_surface"; continue
                if shapely.contains_xy(excl, xx, yy): why = "in_robot"; continue
                if not fs.ok(xx, yy): why = "overlap"; continue
                vel = (0.0, 0.0, 0.0)
                if rng.random() < f["rolling_prob"]:
                    sp = rng.uniform(*f["rolling_speed"]); d = rng.uniform(-math.pi, math.pi); vel = (sp * math.cos(d), sp * math.sin(d), 0.0)
                fs.add(xx, yy, float(F.floor[F.idx(xx, yy)]) + f["radius"], state="ground", vel=vel, src="teleop")
                done = True; break
            if not done: rej[why] += 1
        stats = {"candidates": n, "kept_by_prob": int(keep.sum()), "rejected": rej,
                 "clusters": [{"x": r4(c[0]), "y": r4(c[1]), "sigma": r4(c[2]), "type": c[3]} for c in centers],
                 "wall_piles": [{"wall": str(p[0]), "along": r4(p[1]), "sigma": r4(p[2])} for p in piles]}
    fuel = []
    for x, y, z, kw in fs.items:
        fuel.append({"id": len(fuel), "x": r4(x), "y": r4(y), "z": r4(z), "state": kw["state"],
                     "vx": r4(kw["vel"][0]), "vy": r4(kw["vel"][1]), "vz": r4(kw["vel"][2]), "source": kw["src"]})
    if phase == "teleop":
        for b in airborne_fuel(rng, F, cfg, robots, fs, warn):
            fuel.append({"id": len(fuel), **{k: (r4(v) if isinstance(v, (float, np.floating)) else v) for k, v in b.items()}})
    stats["ground"] = sum(1 for b in fuel if b["state"] == "ground")
    stats["airborne"] = sum(1 for b in fuel if b["state"] == "airborne")
    extra = {}
    if PP is not None:
        import people_venue as pv
        r2 = np.random.default_rng([int(seed), 7])     # separate stream: robots/fuel unchanged
        venue = pv.make_venue(r2, cfg.get("venue"))
        extra = {"venue": venue, "people": pv.make_people(r2, rob_out, F.L, F.W, venue, PP, cfg.get("people"))}
    return {
        "schema": SCHEMA, "frame_id": f"frame_{seed:06d}", "seed": int(seed), "phase": phase,
        "coordinate_frame": {
            "units": "m", "angles": "rad", "up": "+Z",
            "origin": "blue alliance wall, right-hand corner as seen by blue drivers (floor level)",
            "axes": "+X toward red alliance wall, +Y to the left from blue driver view",
            "field_size": [F.L, F.W],
            "rotation_convention": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (Blender rotation_mode 'XYZ'); robot +X = front",
            "robot_origin": "center of frame footprint on the floor; pose.z is the height of that origin"},
        "field": {"model": cfg["field_model"], "model_units": "mm", "cad_to_field": F.cad_to_field,
                  "maps": cfg["maps"]},
        "robots": rob_out, "fuel": fuel, "fuel_stats": stats, "warnings": warn,
        **extra,
        "config": cfg,
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=f"{ROOT}/frames")
    ap.add_argument("--manifest", default=f"{ROOT}/manifest.json")
    ap.add_argument("--fuel-prob", type=float, default=None)
    ap.add_argument("--max-fuel", type=int, default=None)
    ap.add_argument("--config", default=None, help="JSON file with overrides of DEFAULTS (deep-merged)")
    ap.add_argument("--plot", action="store_true", help="also write <frame_id>.png top-down check plot")
    ap.add_argument("--dump-default-config", action="store_true")
    ap.add_argument("--no-people", action="store_true", help="skip people + venue (stadium) randomization")
    a = ap.parse_args(argv)
    cfg = copy.deepcopy(DEFAULTS)
    if a.dump_default_config: print(json.dumps(cfg, indent=1)); return
    if a.config: deep_update(cfg, json.load(open(a.config)))
    if a.fuel_prob is not None: cfg["fuel"]["keep_prob"] = a.fuel_prob
    if a.max_fuel is not None: cfg["fuel"]["max"] = a.max_fuel
    t0 = time.time(); F = Field(cfg); t_load = time.time() - t0
    manifest = load_manifest(a.manifest)
    cfg["robot_source"] = "manifest" if manifest else "placeholder"
    if manifest is not None and len(manifest) == 0:
        print("WARNING: manifest has no usable full robots; using placeholders", file=sys.stderr); manifest = None
        cfg["robot_source"] = "placeholder"
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"field maps loaded in {t_load:.2f}s; robots from {cfg['robot_source']}", file=sys.stderr)
    PP = None
    if not a.no_people:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import people_venue as pv; PP = pv.People()
    for k in range(a.n_frames):
        seed = a.seed + k; t1 = time.time()
        fr = make_frame(seed, F, cfg, manifest, PP)
        p = os.path.join(a.out_dir, fr["frame_id"] + ".json")
        json.dump(fr, open(p, "w"), indent=1)
        dt = time.time() - t1
        msg = (f"{fr['frame_id']} phase={fr['phase']:11s} ground={fr['fuel_stats']['ground']:3d} "
               f"air={fr['fuel_stats']['airborne']:2d} warnings={len(fr['warnings'])} {dt:.2f}s")
        if a.plot:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import plot_frame
            plot_frame.plot(p, F=F)
        print(msg, file=sys.stderr)

if __name__ == "__main__":
    main()
