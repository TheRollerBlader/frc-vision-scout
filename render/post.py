#!/usr/bin/env python3
"""Post-process rendered frames: broadcast degradation -> images/<id>.jpg, YOLO labels -> labels_yolo/<id>.txt,
optional previews with boxes -> preview/<id>.jpg.   python render/post.py OUTDIR [--preview] [--clean]
Classes: 0 bumper_red, 1 bumper_blue, 2 fuel. Boxes are the VISIBLE pixels of each object (occlusion-aware)."""
import argparse, glob, json, os, io, zlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
CLASSES = {"bumper_red": 0, "bumper_blue": 1, "fuel": 2}
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--preview", action="store_true")
ap.add_argument("--clean", action="store_true", help="no degradation (debug)")
ap.add_argument("--min-bumper-px", type=int, default=40); ap.add_argument("--min-fuel-px", type=int, default=12)
ap.add_argument("--out-res", default="", help="e.g. 1920x1080 to resize final images (default: render size)")
a = ap.parse_args()
for d in ("images", "labels_yolo", "preview"): os.makedirs(f"{a.out}/{d}", exist_ok=True)

def degrade(im, rng):
    w, h = im.size
    if rng.random() < 0.5:
        s = rng.uniform(0.5, 0.85); im = im.resize((int(w * s), int(h * s)), Image.BILINEAR).resize((w, h), Image.BICUBIC)
    if rng.random() < 0.3: im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    arr = np.asarray(im).astype(np.float32)
    arr = arr * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8)
    arr += rng.normal(0, rng.uniform(0.5, 4.0), arr.shape)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.UnsharpMask(radius=rng.uniform(1, 2.5), percent=int(rng.uniform(40, 120))))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=int(rng.uniform(45, 92)), subsampling=2); buf.seek(0)
    return Image.open(buf).convert("RGB")

n = 0; stats = {"bumper_red": 0, "bumper_blue": 0, "fuel": 0}
for lp in sorted(glob.glob(f"{a.out}/labels/*.json")):
    L = json.load(open(lp)); fid = L["frame_id"]; W, H = L["width"], L["height"]
    if os.path.exists(f"{a.out}/labels_yolo/{fid}.txt") and not a.preview: continue
    rng = np.random.default_rng(zlib.crc32(fid.encode()))
    im = Image.open(f"{a.out}/{L['image']}").convert("RGB")
    if not a.clean: im = degrade(im, rng)
    sx = sy = 1.0
    if a.out_res:
        ow, oh = [int(v) for v in a.out_res.split("x")]; sx, sy = ow / W, oh / H; im = im.resize((ow, oh), Image.LANCZOS); W, H = ow, oh
    im.save(f"{a.out}/images/{fid}.jpg", quality=95)
    k = (L["width"] / 1920.0) ** 2; mb, mf = max(4, a.min_bumper_px * k), max(2, a.min_fuel_px * k)   # thresholds scale with render size
    rows, draw_boxes = [], []
    for b in L["bumpers"]:
        if b["bbox_visible"] and b["visible_px"] >= mb:
            x0, y0, x1, y1 = b["bbox_visible"]; rows.append((CLASSES[b["class"]], x0 * sx, y0 * sy, x1 * sx, y1 * sy)); stats[b["class"]] += 1
            draw_boxes.append((rows[-1], str(b["team_number"])))
    for f in L["fuel"]:
        if f["bbox_visible"] and f["visible_px"] >= mf:
            x0, y0, x1, y1 = f["bbox_visible"]; rows.append((2, x0 * sx, y0 * sy, x1 * sx, y1 * sy)); stats["fuel"] += 1
            draw_boxes.append((rows[-1], None))
    with open(f"{a.out}/labels_yolo/{fid}.txt", "w") as fh:
        for c, x0, y0, x1, y1 in rows:
            fh.write(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
    if a.preview:
        p = im.copy(); d = ImageDraw.Draw(p)
        for (c, x0, y0, x1, y1), t in draw_boxes:
            col = {0: (255, 60, 60), 1: (60, 140, 255), 2: (255, 230, 0)}[c]
            d.rectangle([x0, y0, x1, y1], outline=col, width=2 if c < 2 else 1)
            if t: d.text((x0, max(0, y0 - 12)), t, fill=col)
        p.save(f"{a.out}/preview/{fid}.jpg", quality=88)
    n += 1
json.dump({"classes": {v: k for k, v in CLASSES.items()}, "frames": n, "boxes": stats}, open(f"{a.out}/post_stats.json", "w"), indent=1)
with open(f"{a.out}/dataset.yaml", "w") as fh:
    fh.write(f"path: {os.path.abspath(a.out)}\ntrain: images\nval: images\nnames:\n  0: bumper_red\n  1: bumper_blue\n  2: fuel\n")
print("post:", n, "frames", stats)
