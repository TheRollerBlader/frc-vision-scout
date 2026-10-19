#!/usr/bin/env python3
"""Uploader-side helper for full_run.py (the agent does the actual Drive calls).
  next [N]                 -> ready robots (not yet uploaded) with folder + units
  record ID FOLDER_ID JSON -> JSON = {"drive_name": "file_id", ...}; writes uploaded/ID.json, deletes staged files
  folders                  -> print drive_folders.json
  setfolder PATH ID        -> remember a Drive folder (e.g. "2023" or "2023/118_Robonauts")
  status                   -> counts
  index OUT                -> build master robots.json
"""
import json, sys, shutil, datetime
from pathlib import Path
BASE = Path("/workspace/frc-cad/full")
RES, STAGE, UPL = BASE / "results", BASE / "stage", BASE / "uploaded"
FOLD = BASE / "drive_folders.json"
ROOT = "1IvY43zP5HUR1r80-v82o71Qgxe2-FfZT"


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def save(p, d):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False)); tmp.replace(p)


def results():
    return {p.stem: load(p) for p in sorted(RES.glob("*.json"))}


cmd = sys.argv[1]
if cmd == "next":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    folders = load(FOLD, {})
    out = []
    for rid, r in sorted(results().items(), key=lambda kv: kv[1].get("staged_at", "")):
        if r.get("status") != "ready" or (UPL / f"{rid}.json").exists():
            continue
        y, t = r["folder"].split("/", 1)
        out.append({"id": rid, "folder": r["folder"], "year_id": folders.get(y), "team_id": folders.get(r["folder"]),
                    "units": [[u["path"], u["drive_name"], round(u["bytes"] / 1e6, 1)] for u in r["units"]]})
        if len(out) >= n:
            break
    print(json.dumps(out, ensure_ascii=False))
elif cmd == "ls":  # compact: id | folder | year_id | team_id ; then units "  path :: name :: MB"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    folders = load(FOLD, {})
    k = 0
    for rid, r in sorted(results().items(), key=lambda kv: kv[1].get("staged_at", "")):
        if r.get("status") != "ready" or (UPL / f"{rid}.json").exists():
            continue
        y = r["folder"].split("/", 1)[0]
        print(f"{rid} | {r['folder']} | year={folders.get(y)} | team={folders.get(r['folder'])}")
        for u in r["units"]:
            print(f"    {u['path']} :: {u['drive_name']} :: {u['bytes']/1e6:.1f}")
        k += 1
        if k >= n:
            break
    print("count", k)
elif cmd == "record":
    rid, fid, m = sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    r = load(RES / f"{rid}.json")
    missing = [u["drive_name"] for u in r["units"] if u["drive_name"] not in m]
    if missing:
        sys.exit(f"missing uploads for {rid}: {missing}")
    save(UPL / f"{rid}.json", {"id": rid, "drive_folder_id": fid, "files": m,
                               "uploaded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})
    shutil.rmtree(STAGE / rid, ignore_errors=True)
    f = load(FOLD, {}); f[r["folder"]] = fid; save(FOLD, f)
    print("recorded", rid)
elif cmd == "rec":  # rec ID FOLDER_ID FILEID1 [FILEID2 ...]  (file ids in unit order)
    rid, fid, ids = sys.argv[2], sys.argv[3], sys.argv[4:]
    r = load(RES / f"{rid}.json")
    if len(ids) != len(r["units"]):
        sys.exit(f"{rid}: {len(r['units'])} units but {len(ids)} ids")
    m = {u["drive_name"]: i for u, i in zip(r["units"], ids)}
    save(UPL / f"{rid}.json", {"id": rid, "drive_folder_id": fid, "files": m,
                               "uploaded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})
    shutil.rmtree(STAGE / rid, ignore_errors=True)
    f = load(FOLD, {}); f[r["folder"]] = fid; save(FOLD, f)
    print("recorded", rid)
elif cmd == "folders":
    print(json.dumps(load(FOLD, {}), indent=0))
elif cmd == "setfolder":
    f = load(FOLD, {}); f[sys.argv[2]] = sys.argv[3]; save(FOLD, f); print("ok")
elif cmd == "status":
    import collections
    rs = results()
    c = collections.Counter(r.get("status") for r in rs.values())
    up = len(list(UPL.glob("*.json")))
    print(dict(c), "| uploaded markers:", up, "| ready-not-uploaded:",
          sum(1 for k, r in rs.items() if r.get("status") == "ready" and not (UPL / f"{k}.json").exists()))
elif cmd == "index":
    out_path = Path(sys.argv[2])
    idx = load("/workspace/frc-cad/run3/robots.json")
    test = load("/workspace/frc-cad/robots.json")  # test-run master (3 uploaded robots)
    test_by = {r["id"]: r for r in test["robots"] if r["status"] == "uploaded"}
    rs = results()
    entries = {}
    for r in idx["robots"]:
        if r["source_type"] != "grabcad":
            continue
        rid = r["id"]
        if rid in test_by:
            e = dict(test_by[rid]); e["status"] = "uploaded"; e.setdefault("notes", [])
            e["zipped"] = any(f.get("compressed") for f in e["files"])
            entries[rid] = e; continue
        x = rs.get(rid)
        e = {"id": rid, "team_number": r["team_number"], "team_name": r["team_name"], "year": r["year"],
             "robot_name": r.get("robot_name"), "description": r.get("description"),
             "grabcad_url": r["cad_sources"][0]["url"], "status": "pending", "files": [], "notes": []}
        if x:
            for k in ("robot_name", "grabcad_model_name", "grabcad_author", "grabcad_model_description",
                      "other_formats_on_grabcad", "notes", "mismatch_flag", "too_large", "problems", "error",
                      "duplicate_of"):
                if x.get(k) not in (None, [], ""):
                    e[k] = x[k]
            e["robot_name"] = x.get("robot_name") or e["robot_name"]
            st = x["status"]
            u = load(UPL / f"{rid}.json")
            if st == "ready":
                if u:
                    e.update(status="uploaded", drive_folder=f"Complete Robot CAD Files/{x['folder']}",
                             drive_folder_id=u["drive_folder_id"],
                             drive_folder_link=f"https://drive.google.com/drive/folders/{u['drive_folder_id']}",
                             processed_at=u["uploaded_at"])
                    for un in x["units"]:
                        fid = u["files"][un["drive_name"]]
                        f = {"drive_name": un["drive_name"], "drive_bytes": un["bytes"], "kind": un["kind"],
                             "original_name": un.get("original_name"), "original_bytes": un.get("original_bytes"),
                             "compressed": {"step": None, "zip": "zip", "zip_split_part": "zip_split",
                                            "bundle_zip": "bundle_zip", "bundle_zip_split_part": "bundle_zip_split"}[un["kind"]],
                             "validated_step_header": "ISO-10303-21",
                             "drive_file_id": fid, "drive_link": f"https://drive.google.com/file/d/{fid}/view"}
                        if un.get("members"):
                            f["members"] = un["members"]
                        if un.get("rejoin"):
                            f["rejoin"] = un["rejoin"]
                        e["files"].append({k: v for k, v in f.items() if v is not None or k == "compressed"})
                    e["zipped"] = any(f["compressed"] for f in e["files"])
                    tot = sum((f.get("original_bytes") or f["drive_bytes"]) for f in e["files"])
                    if tot < 25e6 and x.get("other_formats_on_grabcad"):
                        e.setdefault("notes", []).append(
                            f"STEP content is small ({tot/1e6:.1f} MB) and the model's main CAD is in other formats "
                            f"({', '.join(x['other_formats_on_grabcad'])}); the STEP files may be components only, not the full robot.")
                        e["step_may_be_partial"] = True
                    if x.get("too_large"):
                        e["status"] = "uploaded_partial"
                else:
                    e["status"] = "staged_not_uploaded"
            else:
                e["status"] = st
        entries[rid] = e
    # duplicates: point at owner's files
    for e in entries.values():
        if e["status"] == "duplicate" and e.get("duplicate_of") in entries:
            o = entries[e["duplicate_of"]]
            e["duplicate_of_status"] = o["status"]
            e["files"] = o.get("files", [])
            if o.get("drive_folder_link"):
                e["drive_folder_link"] = o["drive_folder_link"]
    import collections
    lst = sorted(entries.values(), key=lambda e: (-(e["year"] if isinstance(e["year"], int) else -1), e["team_number"] or 0, e["id"]))
    cnt = collections.Counter(e["status"] for e in lst)
    total_bytes = sum(f["drive_bytes"] for e in lst if e["status"] in ("uploaded", "uploaded_partial") for f in e["files"])
    master = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "source_page": "https://spectrum3847.org/resources/frc-cad-collection/",
              "tool": "frc_cad_downloader.py 1.1.0 + full_run.py (--grabcad-direct)",
              "drive_root": {"name": "Complete Robot CAD Files", "id": ROOT},
              "notes": ["STEP files over 95 MB are stored zip-compressed as <name>.zip (one STEP per zip, testzip-verified); "
                        "if a zip was still over 95 MB it is split into <name>.zip.001, .002 ... (open .001 with 7-Zip, or cat the parts together).",
                        "Robots with more than 5 small STEP files have them bundled into <slug>_STEP_files[_partN].zip with the GrabCAD folder paths kept.",
                        "Models with no loose STEP but with .zip archives: the STEP members were extracted and uploaded.",
                        "status: uploaded | uploaded_partial | no_step_available | too_large | duplicate (same GrabCAD model as duplicate_of) | failed | pending"],
              "summary": {"grabcad_robots": len(lst), "status_counts": dict(cnt),
                          "zipped_robots": sum(1 for e in lst if e.get("zipped")),
                          "uploaded_bytes": total_bytes, "mismatch_flagged": sum(1 for e in lst if e.get("mismatch_flag"))},
              "robots": lst}
    out_path.write_text(json.dumps(master, indent=1, ensure_ascii=False))
    print(json.dumps(master["summary"]))
