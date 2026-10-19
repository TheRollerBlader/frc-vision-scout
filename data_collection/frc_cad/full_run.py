#!/usr/bin/env python3
"""Full GrabCAD -> Drive pipeline, STAGER half (resumable, run with nohup).

For every GrabCAD robot in run3/robots.json (newest year first) it resolves the model,
downloads STEP files (or STEP members of zips when the model has no loose STEP) with
--grabcad-direct style URLs, validates them (ISO-10303-21), zips files > 95 MB
(split into .zip.001.. parts if the zip is still > 95 MB), bundles robots with many
small STEP files into <slug>_STEP_files[_partN].zip, and stages the upload units in
full/stage/<id>/ with a result file full/results/<id>.json (status "ready").
The agent uploads staged units to Drive and runs full_upload.py record ... which writes
full/uploaded/<id>.json and deletes the staged files. Disk use is bounded: the stager
waits while full/stage holds more than STAGE_LIMIT bytes.
"""
import json, os, re, shutil, sys, time, traceback, zipfile, logging
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/workspace/frc-cad")
import frc_cad_downloader as fcd

BASE = Path("/workspace/frc-cad/full")
RES, STAGE, UPL = BASE / "results", BASE / "stage", BASE / "uploaded"
INDEX = Path("/workspace/frc-cad/run3/robots.json")
COMPRESS_OVER = 95 * 1000 * 1000
STAGE_LIMIT = 1.0e9          # wait while staged bytes exceed this
MAX_READY = 40               # ...or while this many robots are waiting for upload
MAX_RAW = 1.5e9              # single downloaded file cap (disk safety)
BUNDLE_OVER_N = 5            # >N small STEP files -> bundle into zip(s)
BUNDLE_PART = 90 * 1000 * 1000
TEST_DONE = {"2024-2212-r238", "2025-250-r103", "2025-118-r98"}
MISLABEL = {
    "2026-2026-r36": "Spectrum page lists this as team 2026 'Senai', but the linked GrabCAD model is "
                     "'FRC 2026 - Team 3255 the SuperNURDs' by FRC Team 3255 - mismatch flagged, team not changed.",
    "2024-3522-r266": "Spectrum page lists this as team 3522 'CBoT´S4', but the linked GrabCAD model is "
                      "'FRC 2024 Robot - Team 3255 SuperNURDs' by FRC Team 3255 - mismatch flagged, team not changed.",
}
for d in (RES, STAGE, UPL):
    d.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("full")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/workspace/frc-cad/full_run.log"), logging.StreamHandler()])
logging.getLogger("frc_cad").handlers = []
logging.getLogger("frc_cad").propagate = True


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def pending_uploads():
    return [p.stem for p in RES.glob("*.json")
            if (load(p) or {}).get("status") == "ready" and not (UPL / p.name).exists()]


def wait_for_room():
    t0 = time.time()
    while True:
        used, ready = dir_bytes(STAGE), len(pending_uploads())
        if used < STAGE_LIMIT and ready < MAX_READY:
            return
        if time.time() - t0 > 5:
            log.info("waiting for uploads: staged %.2f GB, %d robots ready", used / 1e9, ready)
        time.sleep(30)


def team_mentions(text):
    return sorted({int(n) for n in re.findall(r"(?:team|frc)\s*#?\s*(\d{2,5})\b", text or "", re.I)})


def safe_drive_name(name, used):
    n = fcd.safe_filename(name)
    base, i = n, 2
    while n.lower() in used:
        stem, dot, ext = base.rpartition(".")
        n = f"{stem} ({i}).{ext}" if dot else f"{base} ({i})"
        i += 1
    used.add(n.lower())
    return n


def stage_robot(gc, http, r, slug_owner):
    rid = r["id"]
    src = r["cad_sources"][0]
    res = {k: r.get(k) for k in ("id", "team_number", "team_name", "year", "robot_name", "description", "folder")}
    res.update(grabcad_url=src["url"], slug=src.get("slug"), notes=[], units=[], started_at=fcd.now_iso())
    if rid in MISLABEL:
        res["notes"].append(MISLABEL[rid])
        res["mismatch_flag"] = True
    if src.get("unsupported") or not src.get("slug"):
        res.update(status="failed", error=f"unsupported GrabCAD link: {src.get('unsupported') or src['url']}")
        return res
    if src["slug"] in slug_owner and slug_owner[src["slug"]] != rid:
        res.update(status="duplicate", duplicate_of=slug_owner[src["slug"]])
        res["notes"].append(f"Same GrabCAD model as {slug_owner[src['slug']]}; uploaded once there.")
        return res
    gc.resolve(src)
    res.update(robot_name=r.get("robot_name") or src.get("model_name"), grabcad_model_name=src.get("model_name"),
               grabcad_author=src.get("author"), grabcad_model_description=src.get("model_description"),
               grabcad_created_at=src.get("model_created_at"))
    ment = team_mentions(" ".join(str(src.get(k) or "") for k in ("model_name", "author")))
    if ment and r.get("team_number") not in ment and rid not in MISLABEL:
        res["notes"].append(f"Possible team mismatch: GrabCAD model name/author mentions team(s) {ment}, "
                            f"page lists team {r.get('team_number')} (not changed).")
        res["mismatch_flag"] = True
    listing = src.get("files_listing") or []
    steps = [f for f in listing if fcd.ext_of(f["name"]) in fcd.STEP_EXTS]
    zips = [f for f in listing if fcd.ext_of(f["name"]) in fcd.ZIP_EXTS]
    res["other_formats_on_grabcad"] = sorted({fcd.ext_of(f["name"]) for f in listing}
                                             - fcd.STEP_EXTS - fcd.ZIP_EXTS - fcd.IMAGE_EXTS - {""})
    res["grabcad_step_files"], res["grabcad_zip_files"] = len(steps), len(zips)
    if not steps and not zips:
        res.update(status="no_step_available",
                   error=f"model has no STEP/STP (formats: {', '.join(res['other_formats_on_grabcad']) or 'none'})")
        return res
    slug_owner[src["slug"]] = rid
    work = STAGE / rid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    raw_small, big_units, problems, too_large = [], [], [], []
    seq = [0]

    def take_step(path: Path, original_name: str, folder: str):
        size = path.stat().st_size
        if not fcd.is_valid_step(path):
            problems.append(f"{folder}{original_name}: not a real STEP ({fcd.file_kind(path)})")
            path.unlink()
            return
        if size > COMPRESS_OVER:
            c = fcd.compress_for_upload(path, COMPRESS_OVER)
            big_units.append({"original_name": folder + original_name, "original_bytes": size,
                              "compressed": c["compressed"], "zip_bytes": c.get("zip_bytes"),
                              "files": c["files"], "rejoin": c.get("rejoin")})
        else:
            raw_small.append({"path": str(path), "original_name": folder + original_name, "bytes": size})

    def fetch(f):
        url = gc.file_url(f)
        seq[0] += 1
        dest = work / "dl" / f"{seq[0]:04d}_{fcd.safe_filename(f['name'])}"
        log.info("  downloading %s%s", f["folder"], f["name"])
        return http.download(url, dest, max_bytes=int(MAX_RAW))

    for f in (steps or zips):
        try:
            info = fetch(f)
        except fcd.DownloadError as e:
            if e.skipped:
                too_large.append(f"{f['folder']}{f['name']}: {e}")
            else:
                problems.append(f"{f['folder']}{f['name']}: {e}")
            continue
        except Exception as e:
            problems.append(f"{f['folder']}{f['name']}: {type(e).__name__}: {e}")
            continue
        p = Path(info["path"])
        if info["kind"] == "step":
            take_step(p, f["name"], f["folder"])
        elif info["kind"] == "zip":
            try:
                with zipfile.ZipFile(p) as z:
                    members = [m for m in z.infolist() if not m.is_dir() and fcd.ext_of(m.filename) in fcd.STEP_EXTS]
                    if not members:
                        problems.append(f"{f['name']}: zip contains no STEP/STP files")
                    for i, m in enumerate(members):
                        if m.file_size > MAX_RAW:
                            too_large.append(f"{f['name']}/{m.filename}: {m.file_size / 1e6:.0f} MB uncompressed")
                            continue
                        mdest = work / "unz" / f"{i:04d}_{fcd.safe_filename(m.filename.replace('/', '_').replace(chr(92), '_'))}"
                        mdest.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(m) as zs, open(mdest, "wb") as out:
                            shutil.copyfileobj(zs, out, 1 << 20)
                        mname = m.filename.replace("\\", "/")
                        take_step(mdest, mname.rsplit("/", 1)[-1], f"{f['name']}/" + (mname.rsplit("/", 1)[0] + "/" if "/" in mname else ""))
            except zipfile.BadZipFile as e:
                problems.append(f"{f['name']}: bad zip: {e}")
            p.unlink(missing_ok=True)
        else:
            p.unlink(missing_ok=True)
            problems.append(f"{f['name']}: unexpected content {info['kind']}")

    used = set()
    units = []
    if len(raw_small) > BUNDLE_OVER_N:
        slug_short = fcd.safe_name(src["slug"], 60)
        parts, cur, cur_est = [], [], 0
        for u in raw_small:
            est = u["bytes"] / 4
            if cur and cur_est + est > BUNDLE_PART:
                parts.append(cur); cur, cur_est = [], 0
            cur.append(u); cur_est += est
        if cur:
            parts.append(cur)
        for i, members in enumerate(parts, 1):
            zname = f"{slug_short}_STEP_files" + (f"_part{i}of{len(parts)}" if len(parts) > 1 else "") + ".zip"
            zpath = work / zname
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
                arcs = set()
                for u in members:
                    arc = u["original_name"].lstrip("/")
                    while arc.lower() in arcs:
                        arc = "dup_" + arc
                    arcs.add(arc.lower())
                    z.write(u["path"], arcname=arc)
            with zipfile.ZipFile(zpath) as z:
                bad = z.testzip()
            if bad:
                raise RuntimeError(f"bundle zip verification failed ({bad})")
            for u in members:
                Path(u["path"]).unlink(missing_ok=True)
            zb = zpath.stat().st_size
            if zb > fcd.UPLOAD_PART_BYTES:   # unlikely; split raw
                c = fcd.compress_for_upload(zpath, 0)  # no-op, then split manually below
                pieces, k = [], 1
                with open(zpath, "rb") as s:
                    while chunk := s.read(fcd.UPLOAD_PART_BYTES):
                        pp = zpath.with_name(f"{zpath.name}.{k:03d}"); pp.write_bytes(chunk); pieces.append(pp); k += 1
                zpath.unlink()
                for pp in pieces:
                    units.append({"path": str(pp), "drive_name": safe_drive_name(pp.name, used), "bytes": pp.stat().st_size,
                                  "kind": "bundle_zip_split_part", "members": [u["original_name"] for u in members],
                                  "original_bytes": sum(u["bytes"] for u in members)})
            else:
                units.append({"path": str(zpath), "drive_name": safe_drive_name(zname, used), "bytes": zb,
                              "kind": "bundle_zip", "members": [u["original_name"] for u in members],
                              "original_bytes": sum(u["bytes"] for u in members)})
        res["notes"].append(f"{len(raw_small)} small STEP files bundled into {len(parts)} zip(s) (folder paths kept inside).")
    else:
        for u in raw_small:
            units.append({"path": u["path"], "drive_name": safe_drive_name(u["original_name"].rsplit("/", 1)[-1], used),
                          "bytes": u["bytes"], "kind": "step", "original_name": u["original_name"],
                          "original_bytes": u["bytes"]})
    for b in big_units:
        for fp in b["files"]:
            fp = Path(fp)
            disp = fp.name.split("_", 1)[1] if re.match(r"^\d{4}_", fp.name) else fp.name
            units.append({"path": str(fp), "drive_name": safe_drive_name(disp, used), "bytes": fp.stat().st_size,
                          "kind": "zip" if b["compressed"] == "zip" else "zip_split_part",
                          "original_name": b["original_name"], "original_bytes": b["original_bytes"],
                          "zip_bytes": b["zip_bytes"], **({"rejoin": b["rejoin"]} if b.get("rejoin") else {})})
            if b["compressed"] == "zip_split":
                res.setdefault("notes", []).append(f"{b['original_name']} zip was {b['zip_bytes']/1e6:.0f} MB; split into "
                                                   f"{len(b['files'])} parts ({b['rejoin']}).")
    res["units"] = units
    if too_large:
        res["too_large"] = too_large
    if problems:
        res["problems"] = problems
    if units:
        res["status"] = "ready"
    elif too_large:
        res["status"] = "too_large"
    elif problems and all("no STEP" in p for p in problems):
        res["status"] = "no_step_available"; res["error"] = "; ".join(problems)
    else:
        res["status"] = "failed"; res["error"] = "; ".join(problems) or "nothing downloaded"
    if res["status"] != "ready":
        shutil.rmtree(work, ignore_errors=True)
    res["staged_at"] = fcd.now_iso()
    return res


def main():
    http = fcd.Http(delay=1.0, retries=4, timeout=300)
    gc = fcd.GrabCAD(http, {}, "", direct=True)
    robots = [r for r in load(INDEX)["robots"] if r["source_type"] == "grabcad"]
    yr = lambda r: r["year"] if isinstance(r["year"], int) else -1
    robots.sort(key=lambda r: (-yr(r), r["row_index"]))
    slug_owner = {}
    for r in robots:  # seed from tests/previous results
        p = RES / f"{r['id']}.json"
        prev = load(p)
        if r["id"] in TEST_DONE or (prev and prev.get("status") in ("ready", "too_large") and prev.get("slug")):
            slug_owner.setdefault(r["cad_sources"][0]["slug"], r["id"])
    for attempt in (1, 2):
        todo = []
        for r in robots:
            if r["id"] in TEST_DONE:
                continue
            prev = load(RES / f"{r['id']}.json")
            if prev is None or (prev.get("status") == "failed" and prev.get("attempts", 1) < attempt):
                todo.append(r)
        log.info("=== pass %d: %d robots to stage ===", attempt, len(todo))
        for n, r in enumerate(todo, 1):
            wait_for_room()
            t0 = time.time()
            log.info("[pass %d %d/%d] %s team %s %s", attempt, n, len(todo), r["year"], r["team_number"], r["team_name"])
            try:
                res = stage_robot(gc, http, r, slug_owner)
            except Exception as e:
                shutil.rmtree(STAGE / r["id"], ignore_errors=True)
                res = {k: r.get(k) for k in ("id", "team_number", "team_name", "year", "robot_name", "description", "folder")}
                res.update(grabcad_url=r["cad_sources"][0]["url"], slug=r["cad_sources"][0].get("slug"),
                           status="failed", error=f"{type(e).__name__}: {e}", notes=[])
                if r["id"] in MISLABEL:
                    res["notes"].append(MISLABEL[r["id"]]); res["mismatch_flag"] = True
                log.debug(traceback.format_exc())
            res["attempts"] = attempt
            res["stage_seconds"] = round(time.time() - t0, 1)
            fcd.write_json_atomic(RES / f"{r['id']}.json", res)
            log.info("  -> %s %s (%.0fs) %s", res["status"],
                     f"{len(res.get('units', []))} unit(s) {sum(u['bytes'] for u in res.get('units', []))/1e6:.0f} MB" if res.get("units") else "",
                     res["stage_seconds"], res.get("error") or "")
    log.info("STAGER DONE")


if __name__ == "__main__":
    main()
