#!/usr/bin/env python3
"""Check manifest.json against Eddie's loader (tools/generate_frame.py) + our conventions."""
import json, sys, os
sys.path.insert(0, "/workspace/frc/tools")
ROOT = "/workspace/frc"
path = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/manifest.json"
M = json.load(open(path))
assert isinstance(M, dict) and isinstance(M.get("robots"), list), "top-level must be {robots:[...]}"
from shapely.geometry import Polygon
import trimesh, numpy as np
probs = 0
for e in M["robots"]:
    k = e.get("key"); P = []
    for f in ["key", "team", "year", "model_file", "full_robot", "height_m", "footprint_m", "bbox_m", "bumper_meshes",
              "frame_length_m", "frame_width_m", "partial"]:
        if f not in e: P.append(f"missing {f}")
    if not isinstance(e.get("team"), int) or not isinstance(e.get("year"), int): P.append("team/year not int")
    if e.get("full_robot"):
        bb = e["bbox_m"]
        if not (len(bb) == 2 and len(bb[0]) == 3 and all(bb[0][i] <= bb[1][i] for i in range(3))): P.append("bbox_m not [[min],[max]]")
        if abs(bb[0][2]) > 0.003: P.append(f"zmin {bb[0][2]}")
        if abs(e["height_m"] - (bb[1][2] - bb[0][2])) > 0.005: P.append("height_m != bbox z extent")
        poly = Polygon(e["footprint_m"])
        if not poly.is_valid or poly.area < 0.1: P.append(f"footprint invalid/small area={poly.area:.3f}")
        b = poly.bounds; c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if max(abs(c[0]), abs(c[1])) > 0.01: P.append(f"footprint not centred {c}")
        if not (0.4 <= e["frame_length_m"] <= 1.3 and 0.4 <= e["frame_width_m"] <= 1.3): P.append(f"frame size {e['frame_length_m']}x{e['frame_width_m']}")
        if e["height_m"] > 1.6 or e["height_m"] < 0.25: P.append(f"height {e['height_m']}")
        mf = os.path.join(ROOT, e["model_file"])
        if os.path.isabs(e["model_file"]) or not os.path.exists(mf): P.append("model_file missing/abs")
        elif "--glb" in sys.argv:
            s = trimesh.load(mf, force="scene")
            names = set(s.graph.nodes)
            miss = [n for n in e["bumper_meshes"] if n not in names]
            if miss: P.append(f"bumper nodes missing in GLB: {miss[:3]}")
            lo, hi = s.bounds   # glTF Y-up: robot (x,y,z) -> (x,z,-y)
            rb = [[lo[0], -hi[2], lo[1]], [hi[0], -lo[2], hi[1]]]
            d = np.array(rb) - np.array(bb); d[1][2] = max(0.0, d[1][2])   # GLB top may sit below the conservative CAD height
            if np.abs(d).max() > 0.01: P.append(f"GLB bounds {np.round(rb,3).tolist()} != bbox_m")
    if P: probs += 1
    print(("FAIL " if P else "ok   ") + f"{k:28s} full={e.get('full_robot')} H={e.get('height_m')} L×W={e.get('frame_length_m')}×{e.get('frame_width_m')} "
          f"bumpers={len(e.get('bumper_meshes') or [])} note={'Y' if e.get('height_note') else '-'} " + "; ".join(P))
# loader round-trip
import importlib.util
spec = importlib.util.spec_from_file_location("gf", f"{ROOT}/tools/generate_frame.py"); gf = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(gf); L = gf.load_manifest(path)
    print(f"loader: {len(L)} usable robots of {len(M['robots'])} entries")
except Exception as ex:
    print("loader import failed:", ex)
print("problems:", probs)
