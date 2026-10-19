"""People + venue randomization for frc-frame JSON (added Oct 8).

people: refs, drive teams, human players, field staff, emcee, camera-side crowd, placed OUTSIDE the field
        in field-frame meters (same frame as robots). Each person: avatar GLB (feet at origin, faces +X),
        scale, pose (x, y, yaw), clip + time, velocity (walk only), outfit for the SHIRT material.
venue:  which stadium, and the 4x4 field->stadium transform (anchor + per-frame jitter/flip).
Uses its own RNG stream (seed, 7) so robot/fuel output for a seed is unchanged.
"""
import json, math, os
import numpy as np

ROOT = "/workspace/frc"
PEOPLE_JSON = f"{ROOT}/models/people/people.json"
ANCHOR_DIR = f"{ROOT}/models/stadiums/anchors"
FIELD_CENTER = (8.27, 4.035)
CARPET_T = 0.006          # field + carpet sit on a 6 mm carpet over the arena floor
CARPET = json.load(open(f"{ROOT}/models/field/carpet_placeholder.json"))

PEOPLE_DEFAULTS = {
    "min_spacing": 0.55,
    "refs": [3, 6],
    "drive_team_per_station": [2, 4],      # driver, operator, coach, (technician)
    "human_player_prob": 0.9,
    "staff": [1, 4],
    "emcee_prob": 0.6,
    "crowd_foreground": [0, 8],
    "student_adult_avatar_prob": 0.75,      # students mostly use adult avatars scaled down (Rocketbox kids are 1.43 m)
    "student_adult_scale": [0.90, 1.02], "student_child_scale": [1.10, 1.22], "adult_scale": [0.96, 1.04],
    "safety_glasses_roles": ["driver", "operator", "coach", "technician", "human_player", "ref", "staff"],
    "palettes": {
        "team": [[0.80, 0.10, 0.10], [0.05, 0.20, 0.60], [0.95, 0.55, 0.05], [0.10, 0.55, 0.20], [0.45, 0.10, 0.55],
                 [0.95, 0.85, 0.10], [0.05, 0.05, 0.05], [0.90, 0.90, 0.90], [0.00, 0.60, 0.65], [0.55, 0.05, 0.15]],
        "staff": [[0.10, 0.15, 0.35], [0.35, 0.35, 0.38], [0.05, 0.05, 0.05], [0.10, 0.30, 0.70], [0.75, 0.15, 0.15]],
        "emcee": [[0.05, 0.05, 0.05], [0.15, 0.15, 0.20], [0.85, 0.85, 0.85], [0.20, 0.30, 0.55]],
    },
    "clips": {
        "ref": {"idle_neutral_01": 3, "idle_neutral_02": 3, "idle_look_around": 2, "idle_waiting": 1, "point_left": 1, "point_right": 1, "walk": 1},
        "driver": {"idle_neutral_01": 2, "idle_look_around": 2, "talk_excited": 1, "point_left": 1, "point_right": 1, "cheer_03": 0.5},
        "operator": {"idle_neutral_01": 2, "idle_look_around": 2, "talk_excited": 1, "point_left": 1, "point_right": 1, "cheer_03": 0.5},
        "coach": {"talk_excited": 3, "point_left": 2, "point_right": 2, "idle_look_around": 1, "cheer_03": 0.5},
        "technician": {"idle_waiting": 2, "idle_neutral_02": 2, "clap": 1, "cheer_01": 1},
        "human_player": {"idle_neutral_01": 2, "idle_look_around": 2, "wave": 1, "cheer_03": 1, "point_left": 0.5},
        "staff": {"idle_neutral_01": 2, "idle_waiting": 2, "talk_neutral": 2, "walk": 1.5, "point_right": 0.5},
        "emcee": {"emcee_moderate": 4, "talk_excited": 2, "wave": 1},
        "crowd_foreground": {"cheer_01": 2, "cheer_03": 1, "clap": 2, "idle_look_around": 1, "wave": 1},
    },
}
VENUE_DEFAULTS = {"stadium_weights": None, "flip_prob": 0.5, "camera_side_prob_neg_y": 0.5}

def _wchoice(rng, d):
    k = list(d); w = np.array([d[x] for x in k], float); return k[int(rng.choice(len(k), p=w / w.sum()))]

def _r(x): return float(round(float(x), 4)) + 0.0

class People:
    def __init__(self):
        P = json.load(open(PEOPLE_JSON))
        self.avatars = P["avatars"]
        self.adults = [a for a in self.avatars if a["apparent_age"] == "adult"]
        self.kids = [a for a in self.avatars if a["apparent_age"] != "adult"]
        self.clipinfo = {c["name"]: c for c in self.avatars[0]["clips"]}

def _yaw_to(x, y, tx, ty): return math.atan2(ty - y, tx - x)

def make_venue(rng, cfgv=None):
    cfgv = {**VENUE_DEFAULTS, **(cfgv or {})}
    idx = json.load(open(f"{ANCHOR_DIR}/index.json"))["stadiums"]
    w = cfgv["stadium_weights"] or {n: 1.0 for n in idx}
    name = _wchoice(rng, {n: w.get(n, 0) for n in idx})
    A = json.load(open(f"{ANCHOR_DIR}/{name}.json"))
    ja, jc = A["jitter_m"]["along"], A["jitter_m"]["across"]
    da, dc = rng.uniform(-ja, ja), rng.uniform(-jc, jc)
    flip = bool(rng.random() < cfgv["flip_prob"])
    yaw = A["yaw"] + math.radians(rng.uniform(-A["jitter_deg"], A["jitter_deg"])) + (math.pi if flip else 0.0)
    ca, sa = math.cos(A["yaw"]), math.sin(A["yaw"])
    cx = A["center_xy"][0] + da * ca - dc * sa; cy = A["center_xy"][1] + da * sa + dc * ca
    c, s = math.cos(yaw), math.sin(yaw)
    # p_st = R(yaw) (p_field - (8.27, 4.035, 0)) + (cx, cy, floor_z)
    tx = cx - (c * FIELD_CENTER[0] - s * FIELD_CENTER[1]); ty = cy - (s * FIELD_CENTER[0] + c * FIELD_CENTER[1])
    M = [[c, -s, 0, tx], [s, c, 0, ty], [0, 0, 1, A["floor_z"] + CARPET_T], [0, 0, 0, 1]]
    return {"stadium": name, "glb": A["glb"], "anchor": f"models/stadiums/anchors/{name}.json",
            "field_to_stadium": [[_r(v) for v in row] for row in M],
            "jitter": {"along_m": _r(da), "across_m": _r(dc), "yaw_rad": _r(yaw - A["yaw"]), "flipped": flip},
            "clear_box": A["clear_box"],
            "carpet": {"glb": "models/field/carpet_placeholder.glb", "material": CARPET["material_name"],
                       "extent_field_m": CARPET["extent_field_m"], "thickness_m": CARPET_T,
                       "clear_stadium_inside_extent_plus_m": 0.5},
            "camera_side": "-y" if rng.random() < cfgv["camera_side_prob_neg_y"] else "+y",
            "render_seed": int(rng.integers(0, 2**31 - 1))}

def make_people(rng, robots, L, W, venue, PP, cfg=None):
    C = {**PEOPLE_DEFAULTS, **(cfg or {})}
    out, pts = [], []
    def free(x, y):
        if -0.45 <= x <= L + 0.45 and -0.45 <= y <= W + 0.45: return False   # never on the field / inside the perimeter
        if 5.0 <= x <= 11.55 and -2.45 <= y <= -1.2: return False             # scoring table (field CAD m3)
        return all((x - a) ** 2 + (y - b) ** 2 >= C["min_spacing"] ** 2 for a, b in pts)
    def avatar(student):
        if student and rng.random() >= C["student_adult_avatar_prob"] and PP.kids:
            a = PP.kids[rng.integers(len(PP.kids))]; sc = rng.uniform(*C["student_child_scale"])
        else:
            a = PP.adults[rng.integers(len(PP.adults))]
            sc = rng.uniform(*(C["student_adult_scale"] if student else C["adult_scale"]))
        return a, sc
    def outfit(role, team=None):
        pal = C["palettes"]
        if role == "ref": sh = {"pattern": "stripes", "colors_srgb": [[0.02, 0.02, 0.02], [0.92, 0.92, 0.92]], "stripe_width_m": 0.03}
        elif role in ("driver", "operator", "coach", "technician", "human_player"):
            sh = {"pattern": "solid", "color_srgb": pal["team"][rng.integers(len(pal["team"]))], "number": team}
        elif role == "staff": sh = {"pattern": "solid", "color_srgb": pal["staff"][rng.integers(len(pal["staff"]))]}
        elif role == "emcee": sh = {"pattern": "solid", "color_srgb": pal["emcee"][rng.integers(len(pal["emcee"]))]}
        else: sh = {"pattern": "keep_original" if rng.random() < 0.4 else "solid",
                    "color_srgb": [float(v) for v in rng.uniform(0.03, 0.9, 3)]}
        if "color_srgb" in sh: sh["color_srgb"] = [_r(v) for v in sh["color_srgb"]]
        return {"shirt": sh, "safety_glasses": role in C["safety_glasses_roles"]}
    def add(role, x, y, yaw, student=False, team=None, alliance=None, station=None, tries=12, jit=0.25):
        for t in range(tries):
            xx, yy = (x, y) if t == 0 else (x + rng.normal(0, jit), y + rng.normal(0, jit))
            if free(xx, yy): break
        else: return None
        a, sc = avatar(student)
        clip = _wchoice(rng, C["clips"][role]); ci = PP.clipinfo[clip]
        yw = yaw + rng.normal(0, 0.35)
        v = (0.0, 0.0)
        if clip == "walk":
            sp = ci["walk_speed_mps"] * sc; v = (sp * math.cos(yw), sp * math.sin(yw))
        pts.append((xx, yy))
        p = {"id": f"{role}_{sum(1 for o in out if o['role'] == role) + 1}", "role": role, "avatar": a["name"],
             "model_file": f"models/people/{a['file']}", "scale": _r(sc), "height_m": _r(a["height_m"] * sc),
             "pose": {"x": _r(xx), "y": _r(yy), "z": 0.0, "yaw": _r(math.atan2(math.sin(yw), math.cos(yw)))},
             "clip": clip, "clip_time_s": _r(rng.uniform(0, ci["duration_s"])),
             "velocity": {"vx": _r(v[0]), "vy": _r(v[1]), "vz": 0.0, "wz": 0.0},
             "outfit": outfit(role, team), "shirt_material_prefix": "SHIRT"}
        if alliance: p["alliance"] = alliance
        if station: p["station"] = station
        out.append(p); return p
    # drive teams: 3 stations per alliance behind the alliance wall; team numbers follow the robots
    for al, x0, face in (("blue", -0.95, 0.0), ("red", L + 0.95, math.pi)):
        teams = [r["team_number"] for r in robots if r["alliance"] == al]
        order = rng.permutation(len(teams))
        for k, yc in enumerate((W / 2 + 2.2, W / 2, W / 2 - 2.2) if al == "blue" else (W / 2 - 2.2, W / 2, W / 2 + 2.2)):
            team = int(teams[order[k]]) if k < len(teams) else None
            n = int(rng.integers(C["drive_team_per_station"][0], C["drive_team_per_station"][1] + 1))
            for j, role in enumerate(["driver", "operator", "coach", "technician"][:n]):
                dx = 0.0 if role != "technician" else rng.uniform(0.9, 1.6)
                xx = x0 - dx if al == "blue" else x0 + dx
                add(role, xx + rng.normal(0, 0.08), yc + rng.uniform(-0.7, 0.7), face, student=(role != "coach" or rng.random() < 0.3),
                    team=team, alliance=al, station=f"{al}{k + 1}")
    # human players behind their outposts (blue outpost near y=0.6, red near y=W-0.6)
    for al, x, y, face in (("blue", -1.4, 0.6, 0.0), ("red", L + 1.4, W - 0.6, math.pi)):
        if rng.random() < C["human_player_prob"]:
            teams = [r["team_number"] for r in robots if r["alliance"] == al]
            add("human_player", x + rng.normal(0, 0.15), y + rng.normal(0, 0.2), face, student=True,
                team=int(teams[rng.integers(len(teams))]) if teams else None, alliance=al)
    # refs: corners + long sides, 0.5-1.6 m outside the guardrail, facing the field
    for _ in range(int(rng.integers(C["refs"][0], C["refs"][1] + 1))):
        if rng.random() < 0.4:
            cx, cy = rng.choice([0.0, L]), rng.choice([0.0, W])
            x = cx + (rng.uniform(0.3, 1.2) if cx > 0 else -rng.uniform(0.3, 1.2)) * (rng.random() < 0.5)
            y = cy + (rng.uniform(0.65, 1.4) if cy > 0 else -rng.uniform(0.65, 1.4))
        else:
            x = rng.uniform(1.0, L - 1.0); y = rng.choice([-1, 1]) * rng.uniform(0.65, 1.6); y = y if y < 0 else W + y
        add("ref", x, y, _yaw_to(x, y, *FIELD_CENTER) + rng.normal(0, 0.4))
    # scoring table is part of the field CAD: x 5.22-11.32, y -2.21..-1.45; staff sit/stand behind it, emcee near an end
    st_side, sy, sx = -1, -1.83, 8.27
    for _ in range(int(rng.integers(C["staff"][0], C["staff"][1] + 1))):
        x, y = rng.uniform(5.4, 11.1), rng.uniform(-3.2, -2.6)
        add("staff", x, y, _yaw_to(x, y, *FIELD_CENTER) + rng.normal(0, 0.6))
    if rng.random() < C["emcee_prob"]:
        x, y = sx + rng.choice([-1, 1]) * rng.uniform(3.6, 6.0), sy + rng.uniform(-0.8, 0.6)
        add("emcee", x, y, _yaw_to(x, y, *FIELD_CENTER) + rng.normal(0, 0.5))
    # camera-side foreground crowd (renderer may drop those outside the camera view)
    cs = -1 if venue["camera_side"] == "-y" else 1
    for _ in range(int(rng.integers(C["crowd_foreground"][0], C["crowd_foreground"][1] + 1))):
        x = rng.uniform(-1.0, L + 1.0); y = (-rng.uniform(2.2, 5.0)) if cs < 0 else W + rng.uniform(2.2, 5.0)
        add("crowd_foreground", x, y, _yaw_to(x, y, x, W / 2) + rng.normal(0, 0.5))
    return out
