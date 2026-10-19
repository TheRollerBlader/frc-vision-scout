#!/usr/bin/env python3
"""Top-down check plot of a frc-frame/v1 JSON over the field clearance map.
Usage: .venv/bin/python tools/plot_frame.py frames/frame_000001.json [more.json ...]  -> writes .png next to each"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Circle

def plot(path, F=None, out=None):
    fr = json.load(open(path))
    if F is None:
        M = np.load(fr["field"]["maps"]); clear, floor = M["clearance"], M["floor_z"]; L, W = [float(v) for v in M["size"]]
    else:
        clear, floor, L, W = F.clear, F.floor, F.L, F.W
    img = np.where(np.isfinite(clear), np.clip(clear, 0, 1.2), 1.5)
    fig, ax = plt.subplots(figsize=(16.5, 8.6))
    ax.imshow(img.T, origin="lower", extent=[0, L, 0, W], cmap="Greys_r", vmin=0, vmax=1.6, interpolation="nearest")
    ax.imshow(np.ma.masked_less(floor.T, 0.02), origin="lower", extent=[0, L, 0, W], cmap="Greens", vmin=0, vmax=0.3, alpha=0.5)
    ax.axvline(fr["config"]["robots"].get("_az", 4.028), color="b", lw=0.8, ls="--"); ax.axvline(L - 4.028, color="r", lw=0.8, ls="--")
    for r in fr["robots"]:
        col = "#d62728" if r["alliance"] == "red" else "#1f5bd6"
        ax.add_patch(MPoly(r["footprint"]["outline_xy"], closed=True, fc=col, ec="k", alpha=0.75, lw=1))
        p = r["pose"]; y = p["yaw"]
        hl = r["footprint"]["length"] / 2 + r["footprint"]["bumper_depth"]
        ax.plot([p["x"], p["x"] + hl * np.cos(y)], [p["y"], p["y"] + hl * np.sin(y)], color="w", lw=2)   # heading
        v = r["velocity"]
        if abs(v["vx"]) + abs(v["vy"]) > 0:
            ax.arrow(p["x"], p["y"], v["vx"] * 0.4, v["vy"] * 0.4, width=0.03, head_width=0.15, color="orange", length_includes_head=True, zorder=5)
        tag = f"{r['team_number']}\nh={r['height']:.2f}" + (" T" if r["under_trench"] else "") + (" B" if r["on_bump"] else "")
        ax.text(p["x"], p["y"], tag, ha="center", va="center", fontsize=7.5, color="w", weight="bold", zorder=6)
    g = [b for b in fr["fuel"] if b["state"] == "ground"]; a = [b for b in fr["fuel"] if b["state"] == "airborne"]
    if g:
        ax.scatter([b["x"] for b in g], [b["y"] for b in g], s=9, c="gold", edgecolors="k", linewidths=0.2, zorder=4)
    for b in a:
        ax.add_patch(Circle((b["x"], b["y"]), 0.075 + 0.04 * b["z"], fill=False, ec="#ffdd00", lw=1.5, zorder=7))
        ax.arrow(b["x"], b["y"], b["vx"] * 0.1, b["vy"] * 0.1, width=0.008, head_width=0.05, color="#ffdd00", zorder=7)
    RC = {"driver": "#ff7f0e", "operator": "#ff7f0e", "coach": "#ff7f0e", "technician": "#ff7f0e", "human_player": "#2ca02c",
          "ref": "k", "staff": "#9467bd", "emcee": "#e377c2", "crowd_foreground": "#7f7f7f"}
    ppl = fr.get("people", [])
    for q in ppl:
        p = q["pose"]; c = RC.get(q["role"], "c")
        ax.add_patch(Circle((p["x"], p["y"]), 0.22, fc=c, ec="w", lw=0.8, zorder=6))
        ax.plot([p["x"], p["x"] + 0.4 * np.cos(p["yaw"])], [p["y"], p["y"] + 0.4 * np.sin(p["yaw"])], color=c, lw=1.5, zorder=6)
        if q["role"] in ("ref", "emcee", "human_player") or q["role"] == "driver":
            ax.text(p["x"], p["y"] - 0.42, q["role"].replace("human_player", "HP") + (f" {q['outfit']['shirt'].get('number')}" if q["role"] == "driver" else ""),
                    ha="center", fontsize=6.5, zorder=6)
    if ppl:
        xs = [q["pose"]["x"] for q in ppl] + [0, L]; ys = [q["pose"]["y"] for q in ppl] + [0, W]
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6); ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color="k", lw=1)
        v = fr.get("venue", {})
        ax.text(0.01, 0.01, f"venue={v.get('stadium')} camera side={v.get('camera_side')}  people={len(ppl)} "
                "(orange drive team, green HP, black ref, purple staff, pink emcee, grey crowd)", transform=ax.transAxes, fontsize=8)
    else:
        ax.set_xlim(-0.1, L + 0.1); ax.set_ylim(-0.1, W + 0.1)
    ax.set_aspect("equal")
    ax.set_title(f"{fr['frame_id']} seed={fr['seed']} phase={fr['phase']}  ground fuel={len(g)} airborne={len(a)}  "
                 f"(white bar = heading, orange = velocity x0.4 s, T = under trench, B = on bump)", fontsize=10)
    ax.set_xlabel("x (m) from blue wall"); ax.set_ylabel("y (m)")
    out = out or path.rsplit(".", 1)[0] + ".png"
    plt.tight_layout(); plt.savefig(out, dpi=80); plt.close(fig)
    return out

if __name__ == "__main__":
    for p in sys.argv[1:]: print(plot(p))
