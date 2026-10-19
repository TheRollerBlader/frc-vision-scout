#!/usr/bin/env python3
"""FRC robot CAD (STEP) -> simplified GLB + manifest for Eddie's Blender bumper dataset.

Layout (agreed with Eddie):
  /workspace/frc/models/robots/<year>_<team>_<slug>.glb
  /workspace/frc/manifest.json
Sources (downloaded from Drive by an agent with the Drive connector; this script never talks to Drive):
  /workspace/frc/src/robots.json                 GrabCAD master index (Drive id 1DPsAYvEzdCDpjmhvYUFU12w-74KS8Cx5)
  /workspace/frc/src/grabcad/<robot id>/<drive_name>   GrabCAD files (STEP, .zip, .zip.001/.002, bundle zips)
  /workspace/frc/src/onshape_inventory.tsv       Onshape exports listed from Drive (file_id, year, folder, folder_id, title, bytes)
  /workspace/frc/src/onshape/<file_id>.step      downloaded Onshape exports
Commands:
  plan                      build work/plan.json (one job per robot) and print a summary
  todo [N] [--source X]     list jobs whose source files are not downloaded yet (Drive file ids + target paths)
  run [--workers 2] [--timeout 900] [--only k1,k2] [--retry-failed] [--force] [--watch]
                            convert every job whose sources are present, skipping finished ones (resumable);
                            each robot runs in its own subprocess with a timeout and an RSS cap; manifest is rewritten after each robot
  one KEY                   (internal) convert a single job
  manifest                  rebuild /workspace/frc/manifest.json from work/results/*.json
  sheet OUT.png k1,k2,...|all|pilot   contact sheet of per-robot previews
  status                    counts
"""
import json, os, re, sys, time, math, shutil, subprocess, zipfile, glob, datetime, signal, traceback
from pathlib import Path

ROOT = Path("/workspace/frc")
SRC = ROOT / "src"
WORK = ROOT / "work"
RES = WORK / "results"
TMP = WORK / "tmp"
LOGS = ROOT / "logs"
MODELS = ROOT / "models" / "robots"
PREV = ROOT / "previews" / "robots"
MANIFEST = Path(os.environ.get("FRC_MANIFEST", str(ROOT / "manifest.json")))   # set FRC_MANIFEST to stage elsewhere
PLAN = WORK / "plan.json"
SELF = Path(__file__).resolve()

TARGET_TRIS = 80000
SMALL_PART_DIAG = 0.025        # m: drop screws/nuts/small hardware before decimation
TRENCH_H = 0.565
OUTLINE_Z = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)   # m above floor, sliced and unioned
RSS_CAP_GB = float(os.environ.get("FRC_RSS_CAP_GB", "6.0"))
MIN_AVAIL_GB = float(os.environ.get("FRC_MIN_AVAIL_GB", "2.5"))

# --------------------------------------------------------------------------- helpers

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(p)


def slugify(s, n=28):
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:n].strip("_") or "robot"


# --------------------------------------------------------------------------- planning

GRABCAD_OVERRIDES = {
    # robots.json notes: files are a different team's robot / different year than the folder says
    "2026-2026-r36": {"team": 3255, "team_name": "SuperNURDs", "note": "listed as 2026 Senai, but files are Team 3255 SuperNURDs 2026 (robots.json mismatch flag)"},
    "2024-3522-r266": {"team": 3255, "team_name": "SuperNURDs", "note": "listed as 3522 CBoT S4, but files are Team 3255 SuperNURDs 2024 (robots.json mismatch flag)"},
    "2019-2360-r577": {"team": 2630, "team_name_from": "2019-2630-r583", "note": "listed as 2360, but GrabCAD model is 'FRC 2630 Grimlock' (robots.json mismatch flag)"},
    "2014-3339-r862": {"year": 2013, "note": "listed as 2014, but model/file is the 2013 robot (robots.json year mismatch)"},
    "2013-45-r866": {"year": 2012, "note": "listed as 2013, but GrabCAD model is titled 2012 (robots.json year mismatch)"},
    "unknown_year-250-r997": {"year": 2023, "note": "year unknown on source page; model titled 2023 (robots.json likely_year)"},
}
# Onshape folders holding more than one robot: variant label per file title keyword
ONSHAPE_VARIANTS = [("Roman I -", "roman1"), ("Roman II", "roman2"), ("Robot Worlds", "worlds"), ("Robot Regionals", "regionals"),
                    ("ReBlitz", "reblitz"), ("2026 Blitz", "blitz"), ("MONARCH V1", "v1"), ("MONARCH V2", "v2"),
                    ("359 2026 V1", "v1"), ("359 2026 V2", "v2"), ("A-190A", "a"), ("A-190B", "b"),
                    ("Lemon Puff", "lemonpuff"), ("Croquembouche", "croquembouche")]
ONSHAPE_SKIP_TITLES = [("Parametric Bumpers 451", "bumpers-only model (no robot)")]

PARTIAL_NOTE_RE = re.compile(r"components only|electronics only|partial - the only STEP|not the full robot|may be components only", re.I)


def build_plan():
    jobs = []
    rj = load(SRC / "robots.json", {"robots": []})
    by_id = {r["id"]: r for r in rj["robots"]}
    for r in rj["robots"]:
        if r.get("status") not in ("uploaded", "uploaded_partial"):
            continue
        ov = GRABCAD_OVERRIDES.get(r["id"], {})
        year = ov.get("year") or r.get("likely_year") or r.get("year")
        team = ov.get("team", r.get("team_number"))
        tname = ov.get("team_name") or (by_id.get(ov["team_name_from"], {}).get("team_name") if ov.get("team_name_from") else None) or r.get("team_name") or ""
        notes = list(r.get("notes") or [])
        flags = []
        if r.get("step_may_be_partial") or any(PARTIAL_NOTE_RE.search(n) for n in notes):
            flags.append("robots.json: STEP may be partial / components only")
        jobs.append({
            "id": r["id"], "source": "grabcad", "year": int(year) if year else 0, "team": int(team), "team_name": tname,
            "listed_team": r.get("team_number"), "listed_year": r.get("year"),
            "robot_name": r.get("grabcad_model_name") or r.get("robot_name"),
            "files": [{"name": f["drive_name"], "id": f["drive_file_id"], "bytes": f["drive_bytes"], "kind": f.get("kind"),
                       "original_name": f.get("original_name"), "original_bytes": f.get("original_bytes"),
                       "local": str(SRC / "grabcad" / r["id"] / f["drive_name"])} for f in r["files"]],
            "index_flags": flags, "override_note": ov.get("note"), "drive_folder": r.get("drive_folder"),
            "needs_attention": bool(r.get("needs_attention")),
        })
    inv = SRC / "onshape_inventory.tsv"
    if inv.exists():
        for line in inv.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fid, year, folder, folder_id, title, nbytes = line.split("\t")
            m = re.match(r"(\d+)_(.*)", folder)
            team, tname = int(m.group(1)), re.sub(r"__r\d+$", "", m.group(2)).replace("_", " ").strip(" -")
            skip = next((why for k, why in ONSHAPE_SKIP_TITLES if k in title), None)
            jobs.append({
                "id": "onshape-" + fid, "source": "onshape", "year": int(year), "team": team, "team_name": tname,
                "robot_name": title.rsplit(".", 1)[0], "variant": next((v for k, v in ONSHAPE_VARIANTS if k in title), None),
                "files": [{"name": title, "id": fid, "bytes": int(nbytes), "kind": "step", "local": str(SRC / "onshape" / f"{fid}.step")}],
                "index_flags": [], "drive_folder": f"Complete Robot CAD Files/{year}/{folder}", "skip": skip,
            })
    # mark which files are actually needed (best-scoring STEP unit; all parts of a split zip)
    for j in jobs:
        units = {}
        for f in j["files"]:
            m = re.match(r"(.*\.zip)\.(\d{3})$", f["name"])
            base = m.group(1) if m else f["name"]
            est = (f.get("original_bytes") or f["bytes"] * 4) if (m or f["name"].lower().endswith(".zip")) else f["bytes"]
            u = units.setdefault(base, {"files": [], "est": 0})
            u["files"].append(f); u["est"] += est
        def uname(b):
            b2 = re.sub(r"\.zip$", "", b, flags=re.I)
            return b2
        best = max(units.items(), key=lambda kv: unit_score(uname(kv[0]), kv[1]["est"]))[0] if units else None
        bundle = any("_STEP_files" in f["name"] for f in j["files"])
        for f in j["files"]:
            m = re.match(r"(.*\.zip)\.(\d{3})$", f["name"])
            base = m.group(1) if m else f["name"]
            f["needed"] = (base == best) or bundle
    # keys: <year>_<team>, suffix when several robots share it
    from collections import defaultdict
    groups = defaultdict(list)
    for j in jobs:
        groups[f"{j['year']}_{j['team']}"].append(j)
    for base, js in groups.items():
        for j in js:
            if len(js) == 1:
                j["key"] = base
            else:
                suf = j.get("variant") or (j["id"].rsplit("-", 1)[-1] if j["source"] == "grabcad" else j["files"][0]["id"][:6].lower())
                j["key"] = f"{base}_{suf}"
            j["slug"] = slugify(j["team_name"]) + (f"_{j['variant']}" if j.get("variant") and len(js) > 1 else "")
            if len(js) > 1 and j["source"] == "grabcad":
                j["slug"] += "_" + j["id"].rsplit("-", 1)[-1]
            j["glb"] = f"models/robots/{j['year']}_{j['team']}_{j['slug']}.glb"
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys)), "duplicate keys"
    save(PLAN, {"generated_at": now(), "jobs": jobs})
    return jobs


def get_plan():
    p = load(PLAN)
    if not p:
        return build_plan()
    return p["jobs"]


def sources_present(j):
    return all(Path(f["local"]).exists() and Path(f["local"]).stat().st_size == f["bytes"] for f in j["files"] if f.get("needed", True))


# --------------------------------------------------------------------------- source selection

COTS_RE = re.compile(r"\bam-|\bam14|rev-|vex|versa|talon|falcon|kraken|neo\b|cim\b|motor|gear|sprocket|pulley|hub\b|shaft|wcp-|sds |mk4|"
                     r"module|bearing|churro|hinge|wheel|breaker|bridge|sidecar|power distribution|snow blower|217-|bag\b|775", re.I)
MAIN_RE = re.compile(r"robot|assy|assem|asm|full|top|main|complete|final|comp", re.I)


def unit_score(name, nbytes):
    s = math.log10(max(nbytes, 1))
    if MAIN_RE.search(name):
        s += 1.5
    if COTS_RE.search(name):
        s -= 2.0
    if re.search(r"intake|climber|arm\b|elevator|shooter|drivetrain|drive ?base|chassis|swerve", name, re.I) and not re.search(r"robot", name, re.I):
        s -= 1.0
    return s


def resolve_source(j, tmpdir):
    """Return (step_path, chosen_name, notes) extracting zips as needed."""
    notes = []
    units = []  # (name, kind, paths, est_bytes)
    split = {}
    for f in j["files"]:
        p = Path(f["local"])
        if not (p.exists() and p.stat().st_size == f["bytes"]):
            continue
        m = re.match(r"(.*\.zip)\.(\d{3})$", f["name"])
        if m:
            split.setdefault(m.group(1), []).append(p)
        elif f["name"].lower().endswith(".zip"):
            units.append((f.get("original_name") or f["name"][:-4], "zip", [p], f.get("original_bytes") or f["bytes"] * 4))
        else:
            units.append((f["name"], "step", [p], f["bytes"]))
    for base, parts in split.items():
        units.append((base[:-4], "split", sorted(parts), sum(x.stat().st_size for x in parts) * 4))
    # bundle zips: expand members as candidate units
    expanded = []
    for u in units:
        name, kind, paths, nb = u
        if kind == "zip":
            try:
                zf = zipfile.ZipFile(paths[0])
                mem = [i for i in zf.infolist() if re.search(r"\.(step|stp)$", i.filename, re.I)]
                if len(mem) > 1 or (len(mem) == 1 and "_STEP_files" in paths[0].name):
                    for i in mem:
                        expanded.append((Path(i.filename).name, "member", [paths[0], i.filename], i.file_size))
                    continue
            except zipfile.BadZipFile:
                notes.append(f"bad zip {paths[0].name}")
                continue
        expanded.append(u)
    if not expanded:
        raise RuntimeError("no STEP unit found")
    expanded.sort(key=lambda u: unit_score(u[0], u[3]), reverse=True)
    name, kind, paths, nb = expanded[0]
    if len(expanded) > 1:
        notes.append(f"chose '{name}' among {len(expanded)} STEP units: " + ", ".join(u[0] for u in expanded[1:6]))
    tmpdir.mkdir(parents=True, exist_ok=True)
    if kind == "step":
        return str(paths[0]), name, notes
    if kind == "split":
        joined = tmpdir / "joined.zip"
        with open(joined, "wb") as out:
            for p in paths:
                with open(p, "rb") as inp:
                    shutil.copyfileobj(inp, out, 1 << 24)
        paths = [joined]; kind = "zip"
    zf = zipfile.ZipFile(paths[0])
    if kind == "member":
        member = paths[1]
    else:
        mem = [i for i in zf.infolist() if re.search(r"\.(step|stp)$", i.filename, re.I)] or zf.infolist()
        member = max(mem, key=lambda i: i.file_size).filename
    out = tmpdir / ("src_" + re.sub(r"[^A-Za-z0-9._-]", "_", Path(member).name))
    with zf.open(member) as inp, open(out, "wb") as o:
        shutil.copyfileobj(inp, o, 1 << 24)
    if paths[0].name == "joined.zip":
        paths[0].unlink()
    return str(out), name, notes


def step_declared_units(path):
    try:
        head = ""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(32 << 20)
                if not chunk:
                    break
                i = chunk.find(b"LENGTH_UNIT")
                if i >= 0:
                    head = chunk[max(0, i - 600): i + 600].decode("latin-1", "ignore")
                    break
        if re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'INCH'", head, re.I):
            return "inch"
        if re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'(FOOT|FEET)'", head, re.I):
            return "foot"
        m = re.search(r"SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.METRE\.", head)
        if m:
            return {".MILLI.": "mm", ".CENTI.": "cm", "$": "m"}.get(m.group(1), m.group(1))
        sys_ = re.search(r"FILE_DESCRIPTION|FILE_NAME\s*\([^;]*", head)
    except Exception:
        pass
    return "unknown"


def step_originating_system(path):
    try:
        with open(path, "rb") as f:
            head = f.read(20000).decode("latin-1", "ignore")
        m = re.search(r"FILE_NAME\s*\((.*?)\);", head, re.S)
        return re.sub(r"\s+", " ", m.group(1))[:300] if m else ""
    except Exception:
        return ""


# --------------------------------------------------------------------------- stage 1: STEP -> raw GLB (OpenCascade, meters, CAD axes)

def step_to_raw_glb(step_path, out_glb, lin_defl_mm=1.0, ang_defl=0.5):
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.RWGltf import RWGltf_CafWriter
    from OCP.RWMesh import RWMesh_NameFormat, RWMesh_CoordinateSystem
    from OCP.TColStd import TColStd_IndexedDataMapOfStringString
    from OCP.Message import Message_ProgressRange
    from OCP.IFSelect import IFSelect_ReturnStatus
    t = time.time()
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    r = STEPCAFControl_Reader(); r.SetNameMode(True); r.SetColorMode(True); r.SetLayerMode(False)
    st = r.ReadFile(step_path)
    if st != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {st}")
    t_read = time.time() - t
    r.Transfer(doc, Message_ProgressRange())
    t_tr = time.time() - t
    stt = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence(); stt.GetFreeShapes(roots)
    if roots.Length() == 0:
        raise RuntimeError("no shapes in STEP")
    for i in range(1, roots.Length() + 1):
        BRepMesh_IncrementalMesh(stt.GetShape_s(roots.Value(i)), lin_defl_mm, False, ang_defl, True)
    t_m = time.time() - t
    t_mabs = time.time()
    w = RWGltf_CafWriter(TCollection_AsciiString(str(out_glb)), True)
    w.SetNodeNameFormat(RWMesh_NameFormat.RWMesh_NameFormat_ProductOrInstance)
    w.SetMeshNameFormat(RWMesh_NameFormat.RWMesh_NameFormat_Product)
    w.SetParallel(True); w.SetMergeFaces(True)
    w.ChangeCoordinateSystemConverter().SetOutputCoordinateSystem(RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup)
    if not w.Perform(doc, TColStd_IndexedDataMapOfStringString(), Message_ProgressRange()):
        raise RuntimeError("glTF write failed")
    return {"read_s": round(t_read, 1), "transfer_s": round(t_tr - t_read, 1), "mesh_s": round(t_m - t_tr, 1), "write_s": round(time.time() - t_mabs, 1)}


# --------------------------------------------------------------------------- stage 2: analysis + simplification (numpy / trimesh)

BUMPER_NAME_RE = re.compile(r"bumper|bmpr|noodle|cordura|fabric|\bbump\b|bumpr", re.I)
WHEEL_NAME_RE = re.compile(r"wheel|colson|omni|mecanum|traction|tread|tire|\bmk4|\bmk5|swerve ?(x|module|drive)|thrifty ?swerve|maxswerve|caster|pneumatic wheel", re.I)
INTAKE_NAME_RE = re.compile(r"intake|collector|ground ?pick|roller ?claw|harvester|acquisition", re.I)
BATTERY_NAME_RE = re.compile(r"battery|mk es17|es17-12|sla\b", re.I)
GAMEPIECE_RE = re.compile(r"\bcone\b|game ?piece|gamepiece|power ?cell|\bcargo\b|\bnote\b|\bcoral\b|\balgae\b|\btote\b|\bfuel\b|"
                          r"\bball\b(?! ?bearing| ?joint| ?detent| ?plunger| ?screw| ?transfer| ?lock)|inner ?tube|yoga ?ball|frisbee|\bdisc\b|hatch ?panel", re.I)


def load_parts(raw_glb):
    import numpy as np, trimesh
    s = trimesh.load(raw_glb, force="scene", process=False)
    par = s.graph.transforms.parents
    root = s.graph.base_frame
    parts = []
    for node in s.graph.nodes_geometry:
        T, gname = s.graph[node]
        g = s.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        v = np.asarray(g.vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
            continue
        v = v @ T[:3, :3].T + T[:3, 3]
        f = np.asarray(g.faces, dtype=np.int64)
        path = []
        n = node
        while n in par and par[n] != root:
            n = par[n]; path.append(str(n))
        path = path[::-1]  # top -> leaf parent
        col = None
        try:
            mat = g.visual.material
            bc = getattr(mat, "baseColorFactor", None)
            if bc is None and hasattr(mat, "diffuse"):
                bc = mat.diffuse
            if bc is not None:
                bc = np.asarray(bc, dtype=float)
                if bc.max() > 1.0:
                    bc = bc / 255.0
                col = [float(x) for x in bc[:4]] + ([1.0] if len(bc) == 3 else [])
        except Exception:
            pass
        parts.append({"name": str(node), "path": path, "v": v, "f": f, "color": col})
    return parts


def bbox(v):
    import numpy as np
    return v.min(0), v.max(0)


def rot_to_z(d):
    """rotation matrix taking unit axis d (one of +-X,+-Y,+-Z) to +Z, keeping a right-handed frame."""
    import numpy as np
    d = np.asarray(d, float)
    z = np.array([0, 0, 1.0])
    if np.allclose(d, z):
        return np.eye(3)
    if np.allclose(d, -z):
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(d, z); ax /= np.linalg.norm(ax)
    c = float(np.dot(d, z)); s = math.sqrt(max(0.0, 1 - c * c))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def rotz(theta):
    import numpy as np
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def tri_area(v, f):
    import numpy as np
    a = v[f[:, 1]] - v[f[:, 0]]; b = v[f[:, 2]] - v[f[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def slice_points(v, f, zs):
    """intersection points of triangle edges with horizontal planes z in zs (vectorized)."""
    import numpy as np
    out = []
    tz = v[f][:, :, 2]
    for z in zs:
        m = (tz.min(1) <= z) & (tz.max(1) >= z)
        if not m.any():
            continue
        tri = v[f[m]]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = tri[:, a], tri[:, b]
            za, zb = pa[:, 2], pb[:, 2]
            k = ((za - z) * (zb - z) <= 0) & (np.abs(zb - za) > 1e-12)
            if k.any():
                t = ((z - za[k]) / (zb[k] - za[k]))[:, None]
                out.append((pa[k] + t * (pb[k] - pa[k]))[:, :2])
    return np.concatenate(out) if out else np.zeros((0, 2))


def cluster_faces(v, f, cell):
    """vertex-clustering simplification: snap vertices to a grid of size cell, drop degenerate/duplicate faces."""
    import numpy as np
    q = np.floor(v / cell).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv, minlength=len(uq)).astype(float)
    nv = np.zeros((len(uq), 3))
    for k in range(3):
        nv[:, k] = np.bincount(inv, weights=v[:, k], minlength=len(uq)) / cnt
    nf = inv[f]
    ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[ok]
    if len(nf):
        key = np.sort(nf, axis=1)
        _, idx = np.unique(key, axis=0, return_index=True)
        nf = nf[np.sort(idx)]
    used = np.unique(nf)
    remap = -np.ones(len(nv), np.int64); remap[used] = np.arange(len(used))
    return nv[used], remap[nf]


def drop_detached(parts, margin=0.03):
    """Union parts whose bounding boxes touch (with margin); keep the main cluster plus any other cluster that is big
    (diag >= 0.35 m) AND close (<= 0.25 m gap) to it. Small floaters outside the main cluster's box are dropped."""
    import numpy as np
    n = len(parts)
    if n < 5:
        return parts, 0
    mn = np.array([p["mn"] for p in parts]) - margin; mx = np.array([p["mx"] for p in parts]) + margin
    par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    CH = 512
    for i0 in range(0, n, CH):
        a_mn, a_mx = mn[i0:i0 + CH, None, :], mx[i0:i0 + CH, None, :]
        ov = np.all((a_mn <= mx[None, :, :]) & (mn[None, :, :] <= a_mx), axis=2)
        ii, jj = np.nonzero(ov)
        for a, b in zip(ii + i0, jj):
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    par[ra] = rb
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    if len(comps) == 1:
        return parts, 0
    def weight(ix):
        return sum(len(parts[i]["f"]) for i in ix) + 1e3 * sum(float(np.prod(np.maximum(parts[i]["mx"] - parts[i]["mn"], 1e-3))) ** (1 / 3) for i in ix)
    main = max(comps.values(), key=weight)
    Mmn = np.min([parts[i]["mn"] for i in main], axis=0); Mmx = np.max([parts[i]["mx"] for i in main], axis=0)
    keep = set(main)
    for ix in comps.values():
        if ix is main:
            continue
        cmn = np.min([parts[i]["mn"] for i in ix], axis=0); cmx = np.max([parts[i]["mx"] for i in ix], axis=0)
        gap = float(np.linalg.norm(np.maximum(0, np.maximum(Mmn - cmx, cmn - Mmx))))
        diag = float(np.linalg.norm(cmx - cmn))
        if gap <= 1e-6 or (diag >= 0.35 and gap <= 0.25):
            keep.update(ix)
    out = [parts[i] for i in range(n) if i in keep]
    return out, n - len(out)


def convert_one(j):
    import numpy as np, trimesh, fast_simplification
    from shapely.geometry import MultiPoint, Polygon
    key = j["key"]
    tmpdir = TMP / key
    res = {k: j.get(k) for k in ("key", "id", "source", "year", "team", "team_name", "robot_name", "glb", "drive_folder", "listed_team", "listed_year")}
    res.update(status="running", started_at=now(), notes=[], warnings=[])
    if j.get("override_note"):
        res["notes"].append(j["override_note"])
    t0 = time.time()
    step_path, chosen, snotes = resolve_source(j, tmpdir)
    res["notes"] += snotes
    res["source_file"] = f"{j['drive_folder']}/{chosen}" if j.get("drive_folder") else chosen
    res["source_file_ids"] = [f["id"] for f in j["files"]]
    res["units_declared"] = step_declared_units(step_path)
    origin = step_originating_system(step_path)
    res["step_header"] = origin
    raw = tmpdir / "raw.glb"
    res["timing"] = step_to_raw_glb(step_path, raw)
    if step_path.startswith(str(tmpdir)):
        os.unlink(step_path)
    parts = load_parts(str(raw))
    raw.unlink()
    if not parts:
        raise RuntimeError("no meshed geometry")
    res["parts_raw"] = len(parts)
    res["tris_raw"] = int(sum(len(p["f"]) for p in parts))

    # ---- junk / scale
    allv = np.concatenate([p["v"][:: max(1, len(p["v"]) // 200)] for p in parts])
    lo, hi = np.percentile(allv, 0.5, 0), np.percentile(allv, 99.5, 0)
    ext = float((hi - lo).max())
    scale = 1.0
    if not (0.3 <= ext <= 2.6):
        best = None
        for fct, lab in ((25.4, "values were inches"), (1000.0, "values were metres-as-mm"), (0.001, "values were mm-as-m"), (10.0, "cm"), (0.1, "x0.1"), (304.8, "feet")):
            e = ext * fct
            if 0.4 <= e <= 2.2:
                sc = abs(math.log(e / 0.95))
                if best is None or sc < best[0]:
                    best = (sc, fct, lab)
        if best:
            scale = best[1]
            res["warnings"].append(f"unit correction x{scale} ({best[2]}); raw extent {ext:.3f} m")
        else:
            res["warnings"].append(f"implausible raw extent {ext:.3f} m, no unit fix found")
    res["units_detected"] = {"mm": "mm", "inch": "inch", "m": "m", "cm": "cm", "foot": "foot"}.get(res["units_declared"], res["units_declared"])
    if scale != 1.0:
        res["units_detected"] += f" (declared) -> corrected x{scale}"
    for p in parts:
        p["v"] = p["v"] * scale
        mn, mx = bbox(p["v"]); p["mn"], p["mx"] = mn, mx; p["diag"] = float(np.linalg.norm(mx - mn))
        p["c"] = (mn + mx) / 2
        p["full"] = " / ".join(p["path"][-4:] + [p["name"]])
    # drop junk: origin markers, far outliers, huge environment pieces
    cs = np.array([p["c"] for p in parts]); w = np.array([len(p["f"]) for p in parts], float)
    med = np.array([np.median(cs[:, k]) for k in range(3)])
    keep, dropped = [], {"junk_name": 0, "outlier": 0, "huge": 0, "small": 0}
    for p in parts:
        nm = p["name"] + " " + (p["path"][-1] if p["path"] else "")
        hd = p["mx"] - p["mn"]
        if re.search(r"origin cube|^origin\b", nm, re.I) or re.search(r"\bfield\b|carpet|\bfloor\b", p["full"], re.I):
            dropped["junk_name"] += 1; continue
        if GAMEPIECE_RE.search(nm) and not re.search(r"intake|roller|shooter|hopper|indexer|holder|guide|stop|sensor|pusher|mech", nm, re.I):
            dropped["game_piece"] = dropped.get("game_piece", 0) + 1; continue
        if len(p["f"]) <= 12 and max(hd) <= 0.11 and (max(hd) - min(hd)) <= 0.02 * max(max(hd), 1e-9):
            dropped["junk_name"] += 1; continue   # origin / reference marker cubes
        if len(p["f"]) < 12 and np.linalg.norm(hd) > 0.2:
            dropped["junk_name"] += 1; continue   # big pyramids/tetrahedra = camera FOV / indicator geometry, not parts
        if len(p["f"]) <= 40 and max(hd) > 0.2:
            vc = p["v"] - p["v"].mean(0)
            try:
                sv = np.linalg.svd(vc[:: max(1, len(vc) // 500)], compute_uv=False)
                if sv[-1] / max(sv[0], 1e-12) < 1e-4:
                    dropped["junk_name"] += 1; continue   # zero-thickness construction surface / sketch plane
            except Exception:
                pass
        if np.linalg.norm(p["c"] - med) > 2.5:
            dropped["outlier"] += 1; continue
        if max(hd) > 2.3:
            dropped["huge"] += 1; continue
        keep.append(p)
    parts = keep
    if not parts:
        raise RuntimeError("nothing left after junk filter")

    # ---- up axis
    allv = np.concatenate([p["v"][:: max(1, len(p["v"]) // 300)] for p in parts])
    wheels = [p for p in parts if WHEEL_NAME_RE.search(p["full"]) and p["diag"] > 0.04]
    wc = np.array([p["c"] for p in wheels]) if wheels else None
    src_prior = {"onshape": {(0, 0, 1): 0.25}, "grabcad": {(0, 1, 0): 0.15, (0, 0, 1): 0.15}}[j["source"]]
    cands = []
    for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        R = rot_to_z(d)
        q = allv @ R.T
        z = q[:, 2] - q[:, 2].min(); h = float(z.max())
        lo2, hi2 = np.percentile(q[:, :2], 1, 0), np.percentile(q[:, :2], 99, 0)
        W, L = (hi2 - lo2)
        sc = {"axis": d, "height": h, "W": float(W), "L": float(L)}
        # spread of lowest points (wheels on the floor spread over the footprint in both directions)
        low = q[z < 0.02][:, :2]
        if len(low) >= 3:
            sp = (np.ptp(low[:, 0]) / max(W, 1e-6)) * (np.ptp(low[:, 1]) / max(L, 1e-6))
        else:
            sp = 0.0
        sc["low_spread"] = float(min(sp, 1.0))
        slab = q[z < 0.12][:, :2]
        try:
            cov = MultiPoint(slab[:: max(1, len(slab) // 4000)]).convex_hull.area / max(W * L, 1e-6) if len(slab) > 3 else 0
        except Exception:
            cov = 0
        sc["slab_cov"] = float(min(cov, 1.0))
        sc["mass_low"] = float(1 - np.clip(np.mean(z) / max(h, 1e-6), 0, 1))
        wf = None
        if wc is not None and len(wc) >= 2:
            wq = wc @ R.T
            wf = float(np.mean((wq[:, 2] - q[:, 2].min()) / max(h, 1e-6)))
        sc["wheel_frac"] = wf
        s = 1.5 * sc["low_spread"] + 1.0 * sc["slab_cov"] + 1.0 * sc["mass_low"] + src_prior.get(d, 0)
        if wf is not None:
            s += 2.0 * (1 - min(1, wf * 2))
        if h > 2.0 or max(W, L) > 1.8:
            s -= 1.0
        sc["score"] = round(float(s), 3)
        cands.append(sc)
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    margin = best["score"] - cands[1]["score"]
    axname = {(1, 0, 0): "+X", (-1, 0, 0): "-X", (0, 1, 0): "+Y", (0, -1, 0): "-Y", (0, 0, 1): "+Z", (0, 0, -1): "-Z"}
    res["up_axis_detected"] = axname[tuple(best["axis"])]
    res["up_confidence"] = "high" if margin > 0.6 else ("medium" if margin > 0.25 else "low")
    res["up_scores"] = {axname[tuple(c["axis"])]: c["score"] for c in cands}
    R = rot_to_z(best["axis"])
    for p in parts:
        p["v"] = p["v"] @ R.T
    if wc is not None:
        wc = wc @ R.T

    # ---- yaw alignment of the frame to the axes (min-area rectangle of the low slab)
    allv = np.concatenate([p["v"][:: max(1, len(p["v"]) // 300)] for p in parts])
    zmin = float(np.percentile(allv[:, 2], 0.05)) if len(allv) > 2000 else float(allv[:, 2].min())
    zmin = min(zmin, float(min(p["mn"][2] if False else p["v"][:, 2].min() for p in parts)))
    slab = allv[(allv[:, 2] - zmin) < 0.15][:, :2]
    yaw = 0.0
    try:
        rect = MultiPoint(slab[:: max(1, len(slab) // 5000)]).minimum_rotated_rectangle
        xy = np.array(rect.exterior.coords)[:4]
        e = xy[1] - xy[0]
        ang = math.degrees(math.atan2(e[1], e[0])) % 90
        off = ang if ang <= 45 else ang - 90
        if 2.0 < abs(off) < 44:
            yaw = -math.radians(off)
            res["warnings"].append(f"frame rotated {off:.1f} deg in CAD; re-aligned")
    except Exception:
        pass
    Rz = rotz(yaw)
    for p in parts:
        p["v"] = p["v"] @ Rz.T
        p["v"][:, 2] -= zmin
        mn, mx = bbox(p["v"]); p["mn"], p["mx"], p["c"] = mn, mx, (mn + mx) / 2
    if wc is not None:
        wc = wc @ Rz.T

    # ---- small-part removal
    big = [p for p in parts if p["diag"] >= SMALL_PART_DIAG]
    if len(big) >= 5:
        dropped["small"] = len(parts) - len(big)
        parts = big
    # ---- detached stray geometry (exploded fasteners, loose game pieces, reference bodies floating next to the robot)
    parts, nstray = drop_detached(parts)
    if nstray:
        dropped["detached"] = nstray
    z0 = float(min(p["v"][:, 2].min() for p in parts))
    if abs(z0) > 1e-6:
        for p in parts:
            p["v"][:, 2] -= z0
            p["mn"], p["mx"] = bbox(p["v"]); p["c"] = (p["mn"] + p["mx"]) / 2
    res["parts_dropped"] = dropped

    # ---- bumpers: names, then geometry
    low_pts = np.concatenate([p["v"][p["v"][:, 2] < 0.25][:, :2] for p in parts if (p["v"][:, 2] < 0.25).any()] or [np.zeros((0, 2))])
    if len(low_pts) > 10:
        P_lo, P_hi = np.percentile(low_pts, 0.1, 0), np.percentile(low_pts, 99.9, 0)
    else:
        P_lo, P_hi = np.array([-1, -1.0]), np.array([1, 1.0])
    PW, PL = P_hi - P_lo
    named_bumpers = 0
    for p in parts:
        p["bumper"] = None
        if BUMPER_NAME_RE.search(p["full"]) and p["mx"][2] < 0.35 and not re.search(r"bumper (mount|bracket|clip|pin|latch)", p["full"], re.I):
            p["bumper"] = "name"; named_bumpers += 1
    noodle_cands = []
    for p in parts:
        if p["bumper"] or named_bumpers:   # names are reliable: only fall back to geometry when no bumper names exist
            continue
        mn, mx = p["mn"], p["mx"]; d = mx - mn; h = d[2]
        if mn[2] < -0.01 or mx[2] > 0.28 or h < 0.05 or h > 0.2:
            continue
        lng, sht = max(d[0], d[1]), min(d[0], d[1])
        alongx = d[0] >= d[1]
        tol = 0.03
        if alongx:
            outer = (abs(mn[1] - P_lo[1]) < tol) or (abs(mx[1] - P_hi[1]) < tol)
        else:
            outer = (abs(mn[0] - P_lo[0]) < tol) or (abs(mx[0] - P_hi[0]) < tol)
        if d[0] >= 0.6 * PW and d[1] >= 0.6 * PL and h >= 0.07:
            inner_lo = P_lo + 0.2 * (P_hi - P_lo); inner_hi = P_hi - 0.2 * (P_hi - P_lo)
            vin = ((p["v"][:, 0] > inner_lo[0]) & (p["v"][:, 0] < inner_hi[0]) & (p["v"][:, 1] > inner_lo[1]) & (p["v"][:, 1] < inner_hi[1])).mean()
            if vin < 0.03:
                p["bumper"] = "geom-ring"
            continue
        if lng >= 0.25 and 0.05 <= sht <= 0.14 and outer:
            if h >= 0.07:
                p["bumper"] = "geom-segment"
            elif 0.055 <= h <= 0.095 and abs(h - sht) / h < 0.35:
                noodle_cands.append(p)
    if len(noodle_cands) >= 2:
        for p in noodle_cands:
            p["bumper"] = "geom-noodle"
    bump = [p for p in parts if p["bumper"]]
    geo_bumpers = sum(1 for p in bump if p["bumper"].startswith("geom"))
    res["has_cad_bumpers"] = bool(bump)
    res["bumper_detection"] = {"by_name": named_bumpers, "by_geometry": geo_bumpers}

    # ---- frame outline (bumpers excluded), recentre
    nb = [p for p in parts if not p["bumper"]]
    pts = np.concatenate([slice_points(p["v"], p["f"], OUTLINE_Z) for p in nb] or [np.zeros((0, 2))])
    if len(pts) < 3:
        raise RuntimeError("no geometry at bumper height")
    hull = MultiPoint(pts).convex_hull
    # trim deployed intakes etc. that stick out at bumper height:
    #  (a) never beyond the CAD bumpers' outer box, (b) at most 0.12 m beyond the floor-contact points (wheels)
    from shapely.geometry import box as sbox
    clip = None
    if bump:
        bmn = np.min([p["mn"] for p in bump], axis=0); bmx = np.max([p["mx"] for p in bump], axis=0)
        clip = sbox(bmn[0], bmn[1], bmx[0], bmx[1])
    contact = np.concatenate([p["v"][p["v"][:, 2] < 0.03][:, :2] for p in nb if (p["v"][:, 2] < 0.03).any()] or [np.zeros((0, 2))])
    if len(contact) >= 3:
        cmn, cmx = contact.min(0) - 0.12, contact.max(0) + 0.12
        cb = sbox(cmn[0], cmn[1], cmx[0], cmx[1])
        clip = cb if clip is None else clip.intersection(cb)
    if clip is not None and not clip.is_empty:
        h2 = hull.intersection(clip)
        if not h2.is_empty and h2.area > 0.25 * hull.area:
            if hull.area - h2.area > 0.01:
                hb, cbnd = hull.bounds, h2.bounds
                res["warnings"].append(f"outline trimmed from {hb[2]-hb[0]:.2f}x{hb[3]-hb[1]:.2f} to {cbnd[2]-cbnd[0]:.2f}x{cbnd[3]-cbnd[1]:.2f} m "
                                       "(geometry outside CAD bumpers / >0.12 m beyond wheels, e.g. deployed intake)")
            hull = h2.convex_hull
    hull = hull.simplify(0.005)
    hxy = np.array(hull.exterior.coords)[:-1]
    cx, cy = (hxy.min(0) + hxy.max(0)) / 2
    for p in parts:
        p["v"][:, 0] -= cx; p["v"][:, 1] -= cy
        mn, mx = bbox(p["v"]); p["mn"], p["mx"], p["c"] = mn, mx, (mn + mx) / 2
    hxy = hxy - [cx, cy]
    if wc is not None:
        wc = wc - [cx, cy, 0]

    # ---- front direction
    front_rot, fconf, fwhy = 0, "low", "default: CAD +X (after up-axis rotation)"
    intake = [p for p in nb if INTAKE_NAME_RE.search(p["full"])]
    batt = [p for p in nb if BATTERY_NAME_RE.search(p["full"]) and p["diag"] > 0.1]
    def dir_of(vec):
        if abs(vec[0]) >= abs(vec[1]):
            return 0 if vec[0] > 0 else 2
        return 1 if vec[1] > 0 else 3   # number of +90deg turns that bring this direction to +X: computed below
    if intake:
        w_ = np.array([len(p["f"]) for p in intake], float)
        ic = np.average(np.array([p["c"][:2] for p in intake]), axis=0, weights=w_)
        hw = (hxy.max(0) - hxy.min(0)) / 2
        rel = ic / np.maximum(hw, 1e-6)
        if np.abs(rel).max() > 0.25:
            front_rot = dir_of(rel); fconf = "medium"; fwhy = f"intake-named parts ({len(intake)}) offset toward that side"
    if fconf == "low" and batt:
        bc = np.mean([p["c"][:2] for p in batt], axis=0)
        hw = (hxy.max(0) - hxy.min(0)) / 2
        rel = bc / np.maximum(hw, 1e-6)
        if np.abs(rel).max() > 0.3:
            front_rot = (dir_of(rel) + 2) % 4; fconf = "low"; fwhy = "opposite the battery"
    # rotate so chosen direction becomes +X: direction index k (0=+X,1=+Y,2=-X,3=-Y) needs rotation by -k*90deg
    if front_rot:
        Rf = rotz(-front_rot * math.pi / 2)
        for p in parts:
            p["v"] = p["v"] @ Rf.T
            mn, mx = bbox(p["v"]); p["mn"], p["mx"], p["c"] = mn, mx, (mn + mx) / 2
        hxy = hxy @ Rf[:2, :2].T
    res["front_confidence"] = fconf
    res["front_method"] = fwhy

    # ---- stats
    allmn = np.min([p["mn"] for p in parts], axis=0); allmx = np.max([p["mx"] for p in parts], axis=0)
    size = allmx - allmn
    res["bbox_m"] = [[round(float(x), 3) for x in allmn], [round(float(x), 3) for x in allmx]]   # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    res["bbox_size_m"] = [round(float(x), 3) for x in size]
    res["height_m"] = round(float(allmx[2]), 3)
    # extended-mechanism check: is the top of the robot a narrow column (raised elevator / climber hook)?
    res["height_note"] = None
    try:
        hh = float(allmx[2])
        topv = np.concatenate([p["v"][p["v"][:, 2] > 0.8 * hh][:, :2] for p in parts if (p["v"][:, 2] > 0.8 * hh).any()])
        top_area = MultiPoint(topv[:: max(1, len(topv) // 3000)]).convex_hull.area if len(topv) > 3 else 0.0
        fp_area = max(Polygon(hxy).area, 1e-6)
        top_names = sorted({(p["path"][-1] if p["path"] else p["name"]) for p in parts if p["mx"][2] > 0.9 * hh})[:4]
        mech = [p for p in parts if p["mx"][2] > 0.9 * hh and re.search(r"elevat|climb|lift|hook|arm|telescop|winch|extension", p["full"], re.I)]
        if hh > 1.35:
            res["height_note"] = f"height {hh:.2f} m is above typical stowed heights: mechanism probably extended/raised in CAD (top parts: {', '.join(top_names)})"
        elif hh > 0.9 and top_area / fp_area < 0.06:
            res["height_note"] = f"top 20% of the height is a narrow column ({top_area/fp_area:.0%} of footprint): possibly a raised elevator/climber (top parts: {', '.join(top_names)})"
        elif mech and hh > 0.9:
            res["height_note"] = f"tallest parts are mechanism parts ({mech[0]['full'][-60:]}): check whether it is extended"
        if res["height_note"]:
            res["warnings"].append("height_note: " + res["height_note"])
    except Exception as e:
        res["warnings"].append(f"height check failed: {e}")
    poly = Polygon(hxy)
    if not poly.exterior.is_ccw:
        hxy = hxy[::-1]
    res["footprint_m"] = [[round(float(x), 4), round(float(y), 4)] for x, y in hxy]
    res["frame_outline"] = res["footprint_m"]
    res["frame_outline_m"] = res["footprint_m"]
    fdim = hxy.max(0) - hxy.min(0)
    res["frame_length_m"] = round(float(fdim[0]), 3)
    res["frame_width_m"] = round(float(fdim[1]), 3)
    # CAD bumper depth (outer bumper extent beyond the frame outline), else Eddie's default 0.085
    bd = None
    if bump:
        bmn = np.min([p["mn"] for p in bump], axis=0); bmx = np.max([p["mx"] for p in bump], axis=0)
        fmn, fmx = hxy.min(0), hxy.max(0)
        ds = [fmn[0] - bmn[0], bmx[0] - fmx[0], fmn[1] - bmn[1], bmx[1] - fmx[1]]
        ds = [d for d in ds if 0.04 <= d <= 0.16]
        if ds:
            bd = round(float(np.median(ds)), 3)
    res["cad_bumper_depth_m"] = bd
    res["bumper_depth_m"] = 0.085
    res["footprint_size_m"] = [round(float(fdim[0]), 3), round(float(fdim[1]), 3)]
    res["footprint_area_m2"] = round(float(poly.area), 4)
    res["fits_under_trench"] = bool(res["height_m"] < TRENCH_H)

    # ---- completeness
    reasons = []
    if j.get("index_flags"):
        reasons += j["index_flags"]
    if max(fdim) < 0.4:
        reasons.append(f"footprint too small ({fdim[0]:.2f} x {fdim[1]:.2f} m): single mechanism/component")
    if max(size[0], size[1]) > 1.8:
        reasons.append(f"horizontal extent {size[0]:.2f} x {size[1]:.2f} m too large: not a single robot or bad units")
    if res["height_m"] < 0.15:
        reasons.append(f"height {res['height_m']:.2f} m: flat/partial model")
    elif res["height_m"] < 0.28:
        reasons.append(f"height {res['height_m']:.2f} m: drivetrain/frame only")
    if min(fdim) < 0.25 and max(fdim) >= 0.4:
        reasons.append(f"footprint {fdim[0]:.2f} x {fdim[1]:.2f} m is not robot-shaped")
    nm = (res.get("source_file") or "") + " " + (j.get("robot_name") or "")
    if re.search(r"drive ?train|drive ?base|chassis|swerve frame|frame assembly|configurable dt|\bdt\b", nm, re.I) and res["height_m"] < 0.45:
        reasons.append("name says drivetrain/frame and model is low")
    if j.get("skip"):
        reasons.append(j["skip"])
    res["complete_robot"] = not reasons
    res["full_robot"] = res["complete_robot"]
    res["partial"] = not res["full_robot"]
    res["exclude"] = bool(reasons)
    res["exclude_reason"] = "; ".join(reasons) if reasons else None
    if max(fdim) > 1.05:
        res["warnings"].append(f"outline {fdim[0]:.2f} x {fdim[1]:.2f} m larger than a legal frame: deployed mechanism or extra geometry at bumper height")

    # ---- group, decimate
    groups = {}
    depth_sets = {}
    for p in nb:
        for dpt in range(min(4, len(p["path"]))):
            depth_sets.setdefault(dpt, set()).add(p["path"][dpt])
    gdepth = next((d for d in sorted(depth_sets) if len(depth_sets[d]) >= 3), max(depth_sets) if depth_sets else None)
    def qcol(c):
        if not c:
            return (0.6, 0.6, 0.6, 1.0)
        return tuple(round(x * 6) / 6 for x in c[:3]) + (1.0,)
    for p in nb:
        top = p["path"][gdepth] if (gdepth is not None and len(p["path"]) > gdepth) else (p["path"][-1] if p["path"] else "robot")
        groups.setdefault(("part", top, qcol(p["color"])), []).append(p)
    if len(groups) > 150:
        g2 = {}
        for (k, top, c), ps in groups.items():
            g2.setdefault(("part", "robot", c), []).extend(ps)
        groups = g2
    for p in bump:
        bn = next((x for x in p["path"] + [p["name"]] if BUMPER_NAME_RE.search(x)), None) if p["bumper"] == "name" else f"{p['name']}"
        groups.setdefault(("bumper", bn or p["name"], qcol(p["color"])), []).append(p)
    tot = sum(len(p["f"]) for ps in groups.values() for p in ps)
    ratio = min(1.0, TARGET_TRIS / max(tot, 1))
    gmeshes = {}
    for gk, ps in groups.items():
        V = []; F = []; off = 0
        for p in ps:
            V.append(p["v"]); F.append(p["f"] + off); off += len(p["v"])
        gmeshes[gk] = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
    gbounds = {gk: (m.vertices.min(0), m.vertices.max(0)) for gk, m in gmeshes.items()}
    for _pass in range(4):
        cur = sum(len(m.faces) for m in gmeshes.values())
        if cur <= TARGET_TRIS * 1.15:
            break
        ratio = TARGET_TRIS / cur
        for gk, m in gmeshes.items():
            n0 = len(m.faces)
            tgt = max(int(n0 * ratio), min(n0, 60))
            if tgt < n0 * 0.97 and n0 > 100:
                try:
                    vv, ff = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32), target_reduction=1 - tgt / n0, agg=7)
                    if len(ff):
                        # quadric placement can overshoot the original surface: clamp to the group's original box
                        vv = np.clip(vv.astype(np.float64), gbounds[gk][0], gbounds[gk][1])
                        gmeshes[gk] = trimesh.Trimesh(vv, ff, process=False)
                except Exception as e:
                    res["warnings"].append(f"decimation failed for a group: {e}")
    scene = trimesh.Scene()
    used = set(); bumper_nodes = []; final_tris = 0
    TO_GLTF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # robot Z-up -> glTF Y-up (Blender importer maps back to Z-up)
    cur = sum(len(m.faces) for m in gmeshes.values())
    if cur > TARGET_TRIS * 1.15:
        # quadric decimation stalls on assemblies of many small closed parts: finish with vertex clustering
        lo_c, hi_c, best_c = 0.002, 0.05, None
        for _ in range(9):
            mid = math.sqrt(lo_c * hi_c)
            n = sum(len(cluster_faces(m.vertices, m.faces, mid)[1]) for m in gmeshes.values())
            if n > TARGET_TRIS * 1.1:
                lo_c = mid
            else:
                hi_c = mid; best_c = mid
        cell = best_c or hi_c
        for gk, m in list(gmeshes.items()):
            vv, ff = cluster_faces(m.vertices, m.faces, cell)
            if len(ff):
                gmeshes[gk] = trimesh.Trimesh(vv, ff, process=False)
        res["notes"].append(f"vertex clustering at {cell*1000:.1f} mm after quadric decimation")
    # decimation can lift the lowest vertices (wheel bottoms): re-seat on the floor and take bbox from the final mesh
    fmn = np.min([m.vertices.min(0) for m in gmeshes.values()], axis=0)
    fmx = np.max([m.vertices.max(0) for m in gmeshes.values()], axis=0)
    dz = float(fmn[2])
    if abs(dz) > 1e-4:
        for m in gmeshes.values():
            m.vertices = m.vertices - [0, 0, dz]
        fmx = fmx - [0, 0, dz]; fmn = fmn - [0, 0, dz]
        if abs(dz) > 0.005:
            res["notes"].append(f"decimated mesh re-seated on floor (shift {dz*1000:.0f} mm)")
    H = max(float(fmx[2]), float(res["height_m"]))   # conservative: never report less than the CAD height
    res["bbox_m"] = [[round(float(fmn[0]), 3), round(float(fmn[1]), 3), 0.0], [round(float(fmx[0]), 3), round(float(fmx[1]), 3), round(H, 3)]]
    res["bbox_size_m"] = [round(res["bbox_m"][1][i] - res["bbox_m"][0][i], 3) for i in range(3)]
    res["height_m"] = round(H, 3)
    res["fits_under_trench"] = bool(res["height_m"] < TRENCH_H)
    for (kind, top, col), ps in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        m = gmeshes[(kind, top, col)]
        base = ("BUMPER_" if kind == "bumper" else "") + re.sub(r"[^A-Za-z0-9_]+", "_", str(top)).strip("_")[:40] or "part"
        nm = base; k = 1
        while nm in used:
            k += 1; nm = f"{base}_{k}"
        used.add(nm)
        m.vertices = m.vertices @ TO_GLTF.T
        rgba = (np.array(col) * 255).astype(np.uint8)
        mat = trimesh.visual.material.PBRMaterial(name=nm + "_mat", baseColorFactor=rgba, metallicFactor=0.2, roughnessFactor=0.7)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(m, node_name=nm, geom_name=nm)
        final_tris += len(m.faces)
        if kind == "bumper":
            bumper_nodes.append(nm)
        # keep robot-frame copy for preview
        m.metadata["robot_v"] = m.vertices @ TO_GLTF  # inverse (orthogonal)
        m.metadata["bumper"] = kind == "bumper"
    out = ROOT / j["glb"]
    out.parent.mkdir(parents=True, exist_ok=True)
    data = scene.export(file_type="glb")
    tmpout = out.with_suffix(".glb.tmp"); tmpout.write_bytes(data); tmpout.replace(out)
    res["triangle_count"] = int(final_tris)
    res["node_count"] = len(used)
    res["bumper_meshes"] = bumper_nodes
    res["glb_bytes"] = len(data)
    res["model_file"] = j["glb"]
    res["glb_path"] = str(out)

    # ---- preview
    try:
        render_preview(scene, res, PREV / f"{key}.png")
    except Exception as e:
        res["warnings"].append(f"preview failed: {e}")

    # ---- confidence summary
    conf = "high"
    if res["up_confidence"] == "low" or res["warnings"]:
        conf = "medium"
    if res["up_confidence"] == "low" and res["warnings"]:
        conf = "low"
    res["confidence"] = conf
    res["status"] = "excluded" if res["exclude"] else "ok"
    res["seconds"] = round(time.time() - t0, 1)
    res["finished_at"] = now()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return res


def render_preview(scene, res, out_png):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tris, cols = [], []
    for name, g in scene.geometry.items():
        v = g.metadata.get("robot_v")
        if v is None:
            continue
        c = np.array(g.visual.material.baseColorFactor[:3], float) / 255.0
        if g.metadata.get("bumper"):
            c = np.array([1.0, 0.1, 0.6])
        tris.append(v[g.faces]); cols.append(np.repeat(c[None], len(g.faces), 0))
    T = np.concatenate(tris); C = np.concatenate(cols)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.4), dpi=80)
    views = [("top (+X right = front, +Y up)", (0, 1), 2, np.array([0.3, 0.3, 1.0])), ("side from -Y (+X right = front)", (0, 2), 1, np.array([0.3, -1.0, 0.4]))]
    for ax, (title, (a, b), depth_ax, light) in zip(axs, views):
        light = light / np.linalg.norm(light)
        sh = np.clip(np.abs(n @ light), 0.15, 1.0)[:, None]
        if depth_ax == 2:
            order = np.argsort(T[:, :, 2].mean(1))          # draw low first, high last
        else:
            order = np.argsort(-T[:, :, 1].mean(1))         # draw far (+Y) first
        pc = PolyCollection(T[order][:, :, [a, b]], facecolors=np.clip(C[order] * sh[order], 0, 1), edgecolors="none", linewidths=0)
        ax.add_collection(pc)
        if depth_ax == 2:
            fp = np.array(res["footprint_m"] + [res["footprint_m"][0]])
            ax.plot(fp[:, 0], fp[:, 1], "-", color="lime", lw=1.6)
            ax.annotate("", xy=(0.45, 0), xytext=(0.15, 0), arrowprops=dict(arrowstyle="->", color="red", lw=2))
        else:
            ax.axhline(0.10, color="lime", lw=0.8, ls="--"); ax.axhline(TRENCH_H, color="orange", lw=0.8, ls=":")
        ax.set_aspect("equal"); ax.autoscale_view()
        lim = 0.9
        if depth_ax == 2:
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        else:
            ax.set_xlim(-lim, lim); ax.set_ylim(-0.05, max(1.0, res["height_m"] + 0.1))
        ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
    st = "EXCL" if res.get("exclude") else "ok"
    fig.suptitle(f"{res['key']} {res.get('team_name','')[:22]} [{res['source']}] {st}  up={res['up_axis_detected']}({res['up_confidence']}) "
                 f"front={res['front_confidence']}  size={res['bbox_size_m']}  tris={res.get('triangle_count')}  bumpers={len(res.get('bumper_meshes') or [])}",
                 fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png); plt.close(fig)


# --------------------------------------------------------------------------- manifest

MANIFEST_FIELDS_ORDER = ["key", "id", "team", "team_number", "team_name", "year", "source", "source_file", "model_file", "glb", "full_robot",
                         "partial", "complete_robot", "exclude", "exclude_reason", "height_m", "height_note", "bbox_m", "bbox_size_m",
                         "frame_length_m", "frame_width_m", "footprint_m", "frame_outline_m", "frame_outline", "bumper_depth_m",
                         "cad_bumper_depth_m", "bumper_meshes",
                         "has_cad_bumpers", "fits_under_trench", "triangle_count", "up_axis_detected", "up_confidence", "units_detected",
                         "front_confidence", "front_method", "confidence", "warnings", "notes"]


def write_manifest():
    entries = []
    failed = []
    for p in sorted(RES.glob("*.json")):
        r = load(p)
        if not r:
            continue
        if r.get("status") in ("ok", "excluded"):
            e = dict(r)
            e["team_number"] = r["team"]
            e["glb"] = r.get("glb_path")
            for k in ("timing", "started_at", "glb_path"):
                e.pop(k, None)
            ordered = {k: e[k] for k in MANIFEST_FIELDS_ORDER if k in e}
            ordered.update({k: v for k, v in e.items() if k not in ordered})
            entries.append(ordered)
        else:
            failed.append({k: r.get(k) for k in ("key", "id", "team", "year", "source", "status", "error", "drive_folder")})
    entries.sort(key=lambda e: (e["year"], e["team"], e["key"]))
    man = {
        "generated_at": now(),
        "generator": "/workspace/frc/tools/convert_robots.py",
        "conventions": {
            "units": "meters",
            "robot_frame": "Z-up, +X = robot front, origin at the centre of the bumper-free frame outline, lowest point at z=0",
            "glb_axes": "GLB vertices are stored per the glTF spec (+Y up, +X front, i.e. robot (x,y,z) -> glTF (x,z,-y)); "
                        "Blender's glTF importer converts back, so in Blender the robot is Z-up, +X front, origin at floor centre.",
            "footprint_m": "convex hull of horizontal cross-sections at z = 0.02..0.12 m with CAD bumper parts excluded (robot frame, metres)",
            "bumper_meshes": "GLB node/mesh names that are CAD bumpers (hide them and use generated bumpers)",
            "model_file": "relative to /workspace/frc/",
            "bbox_m": "[[xmin,ymin,zmin],[xmax,ymax,zmax]] in robot frame (m), all geometry incl. bumpers; bbox_size_m = extents",
            "height_m": "max z of the model as posed in the CAD (driving config assumed); height_note set when a mechanism looks extended",
            "frame_length_m/frame_width_m": "X/Y extent of footprint_m (bare frame, no bumpers)",
            "bumper_depth_m": "0.085 default for generated bumpers; cad_bumper_depth_m = measured depth of CAD bumpers when present",
            "full_robot": "false = partial / mislabeled / subsystem-only model; see exclude_reason",
        },
        "counts": {"robots": len(entries), "full_robot": sum(1 for e in entries if e["full_robot"]),
                   "excluded": sum(1 for e in entries if not e["full_robot"]), "failed": len(failed)},
        "robots": entries,
        "failed": failed,
    }
    save(MANIFEST, man)
    return man


# --------------------------------------------------------------------------- runner

def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return 99


def proc_tree_rss_gb(pid):
    total = 0
    try:
        out = subprocess.run(["ps", "-o", "pid=,ppid=,rss=", "-e"], capture_output=True, text=True).stdout
        kids = {}
        rss = {}
        for line in out.splitlines():
            a, b, c = line.split()
            kids.setdefault(int(b), []).append(int(a)); rss[int(a)] = int(c)
        stack = [pid]
        while stack:
            x = stack.pop(); total += rss.get(x, 0); stack += kids.get(x, [])
    except Exception:
        pass
    return total / 1e6


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)


def excluded_without_conversion(j):
    r = {k: j.get(k) for k in ("key", "id", "source", "year", "team", "team_name", "robot_name", "drive_folder", "listed_team", "listed_year")}
    reasons = list(j.get("index_flags") or []) + ([j["skip"]] if j.get("skip") else [])
    r.update(status="excluded", full_robot=False, partial=True, complete_robot=False, exclude=True,
             exclude_reason="; ".join(reasons) + " (not converted)", model_file=None, glb_path=None,
             source_file=", ".join(f["name"] for f in j["files"])[:300], notes=[j["override_note"]] if j.get("override_note") else [],
             warnings=[], finished_at=now(), converted=False)
    return r


def run(args):
    workers = 2; timeout = 900; only = None; retry_failed = False; force = False; watch = False; order = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--workers": workers = int(args[i + 1]); i += 1
        elif a == "--timeout": timeout = int(args[i + 1]); i += 1
        elif a == "--only": only = set(args[i + 1].split(",")); i += 1
        elif a == "--retry-failed": retry_failed = True
        elif a == "--force": force = True
        elif a == "--watch": watch = True
        i += 1
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    (WORK / "run.pid").write_text(str(os.getpid()))
    running = {}  # key -> (Popen, start, job, logf)
    done_this_run = set()
    while True:
        jobs = build_plan()
        pending = []
        for j in jobs:
            k = j["key"]
            if only and k not in only and j["id"] not in only:
                continue
            if k in running or k in done_this_run:
                continue
            r = load(RES / f"{k}.json")
            if r and not force:
                if r.get("status") in ("ok", "excluded"):
                    continue
                if r.get("status") in ("failed", "timeout", "oom") and not retry_failed:
                    continue
            if (j.get("index_flags") or j.get("skip")) and not sources_present(j):
                save(RES / f"{k}.json", excluded_without_conversion(j)); done_this_run.add(k)
                log(f"EXCL  {k}: not converted ({'; '.join(j.get('index_flags') or [j.get('skip')])})")
                continue
            if not sources_present(j):
                continue
            pending.append(j)
        # small files first so results arrive early; very large ones run alone
        pending.sort(key=lambda j: sum(f["bytes"] for f in j["files"]))
        if not pending and not running:
            if watch:
                time.sleep(60); continue
            break
        while pending and len(running) < workers:
            j = pending[0]
            big = sum(f["bytes"] for f in j["files"]) > 200e6
            if big and running:
                break
            if mem_available_gb() < MIN_AVAIL_GB and running:
                break
            pending.pop(0)
            logf = open(LOGS / f"{j['key']}.log", "w")
            p = subprocess.Popen([sys.executable, str(SELF), "one", j["key"]], stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            running[j["key"]] = (p, time.time(), j, logf)
            log(f"START {j['key']} ({j['id']}, {sum(f['bytes'] for f in j['files'])/1e6:.0f} MB) pid={p.pid}")
            if big:
                break
        time.sleep(3)
        for k, (p, t0, j, logf) in list(running.items()):
            rc = p.poll()
            el = time.time() - t0
            why = None
            if rc is None:
                if el > timeout:
                    why = "timeout"
                elif proc_tree_rss_gb(p.pid) > RSS_CAP_GB:
                    why = "oom"
                elif mem_available_gb() < 0.8:
                    why = "oom"
                if why:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    p.wait()
            if rc is None and not why:
                continue
            logf.close()
            del running[k]; done_this_run.add(k)
            r = load(RES / f"{k}.json")
            if why or not r or r.get("status") not in ("ok", "excluded"):
                err = why or (r or {}).get("error") or f"exit code {rc}"
                fr = {k2: j.get(k2) for k2 in ("key", "id", "source", "year", "team", "team_name", "drive_folder", "glb")}
                fr.update(status=why or "failed", error=(f"killed: {why} after {el:.0f}s" if why else err), finished_at=now(), seconds=round(el, 1))
                if r and r.get("traceback"):
                    fr["traceback"] = r["traceback"]
                save(RES / f"{k}.json", fr)
                shutil.rmtree(TMP / k, ignore_errors=True)
                log(f"FAIL  {k}: {fr['error'][:200]}")
            else:
                log(f"DONE  {k}: {r['status']} tris={r.get('triangle_count')} bbox={r.get('bbox_m')} up={r.get('up_axis_detected')}/{r.get('up_confidence')} "
                    f"front={r.get('front_confidence')} bumpers={len(r.get('bumper_meshes') or [])} {r.get('seconds')}s"
                    + (f" EXCL: {r.get('exclude_reason')}" if r.get("exclude") else ""))
            write_manifest()
    write_manifest()
    log("run finished")


def one(key):
    jobs = {j["key"]: j for j in get_plan()}
    j = jobs[key]
    try:
        res = convert_one(j)
    except Exception as e:
        res = {k: j.get(k) for k in ("key", "id", "source", "year", "team", "team_name", "drive_folder", "glb")}
        res.update(status="failed", error=f"{type(e).__name__}: {e}"[:500], traceback=traceback.format_exc()[-3000:], finished_at=now())
        shutil.rmtree(TMP / key, ignore_errors=True)
    save(RES / f"{key}.json", res)
    print(json.dumps({k: res.get(k) for k in ("key", "status", "error", "triangle_count", "bbox_m", "exclude_reason")}))


def sheet(out, which):
    import matplotlib; matplotlib.use("Agg")
    from PIL import Image
    if which == "all":
        keys = sorted(p.stem for p in PREV.glob("*.png"))
    elif which.startswith("@"):
        keys = [l.strip() for l in open(which[1:]) if l.strip()]
    else:
        keys = which.split(",")
    ims = [Image.open(PREV / f"{k}.png") for k in keys if (PREV / f"{k}.png").exists()]
    if not ims:
        print("no previews"); return
    w, h = ims[0].size
    cols = 3
    rows = math.ceil(len(ims) / cols)
    S = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(ims):
        S.paste(im.convert("RGB").resize((w, h)), ((i % cols) * w, (i // cols) * h))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    S.save(out)
    print(out, len(ims))


def status():
    from collections import Counter
    jobs = build_plan()
    c = Counter(); have = 0
    for j in jobs:
        r = load(RES / f"{j['key']}.json")
        c[(r or {}).get("status", "pending" if sources_present(j) else "not_downloaded")] += 1
    print(dict(c), "jobs:", len(jobs))
    tot = sum(p.stat().st_size for p in MODELS.glob("*.glb"))
    print(f"GLB total: {tot/1e6:.1f} MB in {len(list(MODELS.glob('*.glb')))} files")


def todo(args):
    n = int(args[0]) if args and args[0].isdigit() else 50
    src = None
    if "--source" in args:
        src = args[args.index("--source") + 1]
    jobs = build_plan(); k = 0
    for j in jobs:
        if src and j["source"] != src:
            continue
        if sources_present(j) or load(RES / f"{j['key']}.json") or j.get("index_flags") or j.get("skip"):
            continue
        for f in j["files"]:
            if not f.get("needed", True):
                continue
            p = Path(f["local"])
            if not (p.exists() and p.stat().st_size == f["bytes"]):
                print(f"{j['key']}\t{f['id']}\t{f['bytes']/1e6:.1f}MB\t{f['local']}")
        k += 1
        if k >= n:
            break


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "plan":
        js = build_plan(); print(len(js), "jobs;", sum(1 for j in js if sources_present(j)), "with sources present")
    elif cmd == "run":
        run(sys.argv[2:])
    elif cmd == "one":
        one(sys.argv[2])
    elif cmd == "manifest":
        m = write_manifest(); print(m["counts"])
    elif cmd == "sheet":
        sheet(sys.argv[2], sys.argv[3])
    elif cmd == "status":
        status()
    elif cmd == "todo":
        todo(sys.argv[2:])
    else:
        print(__doc__)
