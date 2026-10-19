"""Per-stadium preparation for blender_render.py. Everything heavy here runs ONCE per stadium load (per Blender
process / chunk); the per-frame functions only write numpy arrays into attributes / light values.

  clear_union()      delete stadium faces over the UNION of all possible field placements (carpet + 0.5 m, grown by
                     the anchor jitter: +-along/across metres, +-jitter_deg, optional 180 deg flip) from 4 mm to 3.5 m
                     above the arena floor (floor decals/logos at +7 mm would otherwise poke through the 6 mm carpet), plus small loose props (hoops, benches, goals) near it.
  split_floor()      arena floor faces outside the carpet get their own ARENA_FLOOR material (randomized per frame,
                     never bright white ice -> no blow-out).
  detect_seats()     seat spots from the stadium mesh: upward faces 0.3..25 m above the floor, outside the cleared area,
                     area-sampled, thinned to ~0.55 m along rows / ~0.75 m between rows, kept only on stepped tiers
                     (raycast: higher step behind or a drop in front) with >= 1.2 m headroom. Cached on disk.
  build_crowd()      one point cloud + a geometry-nodes modifier that instances low-poly Rocketbox proxies
                     (render/assets/crowd/crowd_proxies.blend) on the occupied seats. Instances keep pass_index 0.
  update_crowd()     per frame: occupancy 30-90 % (+ per-section variation), proxy choice, yaw +-20 deg towards the
                     field, height 0.9-1.1, shirt / pants colours (per-instance attributes read by the shader), and
                     hides fans whose silhouette could cover the field volume from this camera (labels unchanged).
  stand_light_slots() / field_irradiance(): dim area lights over the stands + auto-exposure from the field irradiance.
"""
import bpy, bmesh, os, math, json, time, hashlib, colorsys
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import convex_hull_2d

ROOT = os.environ.get("FRC_ROOT", "/workspace/frc")
FC = (8.27, 4.035)               # field centre (field frame)
CARPET_T = 0.006
SEAT_ALGO_VERSION = 7
PROXY_BLEND = f"{ROOT}/render/assets/crowd/crowd_proxies.blend"
CACHE_DIR = os.environ.get("FRC_CACHE", f"{ROOT}/cache/stadium")

def log(*a): print("[prep]", *a, flush=True)

# ------------------------------------------------------------------ anchor-frame helpers
def to_anchor(P, A):
    """stadium world xy -> anchor frame (origin at the anchor centre, x along the anchor yaw)"""
    c, s = math.cos(A["yaw"]), math.sin(A["yaw"]); d = P[:, :2] - np.array(A["center_xy"])
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)

def from_anchor_dir(D, A):
    c, s = math.cos(A["yaw"]), math.sin(A["yaw"])
    return np.stack([c * D[:, 0] - s * D[:, 1], s * D[:, 0] + c * D[:, 1]], 1)

def union_polys(A, carpet, margin):
    """convex pieces whose union = every carpet(+margin) placement the frame generator can produce (anchor frame).
    p_anchor = R(delta + flip*pi) (p_field - FC) + (along, across); per angle the translation box makes the
    Minkowski sum convex, so the union over 0.5 deg steps (and both flips) is exact up to ~0.06 m."""
    ex, ey = carpet["x"], carpet["y"]
    x0, x1, y0, y1 = ex[0] - FC[0] - margin, ex[1] - FC[0] + margin, ey[0] - FC[1] - margin, ey[1] - FC[1] + margin
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    ja, jc = A.get("jitter_m", {}).get("along", 0.0), A.get("jitter_m", {}).get("across", 0.0)
    sq = np.array([[-ja, -jc], [ja, -jc], [ja, jc], [-ja, jc]])
    jd = math.radians(A.get("jitter_deg", 0.0)); flips = [0.0, math.pi] if A.get("allow_flip_180") else [0.0]
    n = max(2, int(round(math.degrees(2 * jd) / 0.5)) + 1)
    polys = []
    for f in flips:
        for d in np.linspace(-jd, jd, n):
            c, s = math.cos(d + f), math.sin(d + f)
            r = rect @ np.array([[c, s], [-s, c]])
            pts = (r[:, None, :] + sq[None]).reshape(-1, 2)
            idx = convex_hull_2d([tuple(p) for p in pts])
            polys.append(pts[list(idx)])
    return polys

def in_union(P, polys):
    inside = np.zeros(len(P), bool)
    if len(P) == 0: return inside
    allp = np.concatenate(polys); lo, hi = allp.min(0), allp.max(0)
    cand = np.flatnonzero((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1]))
    Q = P[cand]; ins = np.zeros(len(cand), bool)
    for a in polys:
        b = np.roll(a, -1, 0)
        cr = (b[:, 0] - a[:, 0])[None] * (Q[:, 1:2] - a[None, :, 1]) - (b[:, 1] - a[:, 1])[None] * (Q[:, 0:1] - a[None, :, 0])
        ins |= (cr >= -1e-9).all(1) | (cr <= 1e-9).all(1)
    inside[cand] = ins
    return inside

def world_verts(o):
    me = o.data; v = np.empty(len(me.vertices) * 3, np.float64); me.vertices.foreach_get("co", v); v = v.reshape(-1, 3)
    Mw = np.array(o.matrix_world); return v @ Mw[:3, :3].T + Mw[:3, 3]

def components(n, E):
    """connected components (union-find with pointer jumping, numpy) -> label per node"""
    lab = np.arange(n)
    if len(E) == 0: return lab
    while True:
        la, lb = lab[E[:, 0]], lab[E[:, 1]]; m = la != lb
        if not m.any(): break
        lo, hi = np.minimum(la, lb)[m], np.maximum(la, lb)[m]
        np.minimum.at(lab, hi, lo)
        while True:
            nl = lab[lab]
            if (nl == lab).all(): break
            lab = nl
    return lab

# ------------------------------------------------------------------ 0) coincident back-face twins
def duplicate_faces(objs, A):
    """the stadium GLBs carry a back-facing copy of most surfaces (same plane, opposite normal, often triangulated
    differently). Coplanar opposite twins make Cycles shadow rays hit the twin (shadow acne) -> arena floor, stand
    fronts and walls render BLACK. Per plane (normal + offset, orientation-free key) that has faces of both
    orientations, delete the faces that point away from a point 3 m above the field centre and lie inside the bbox of
    the kept side. Cycles shades both sides of a face, so nothing disappears. Returns {object name: delete mask}."""
    cx, cy = A["center_xy"]; T = np.array([cx, cy, A["floor_z"] + 3.0])
    C_, N_, own = [], [], []
    for o in objs:
        me = o.data; nf = len(me.polygons)
        if nf == 0: continue
        Mw = np.array(o.matrix_world)
        cen = np.empty(nf * 3); me.polygons.foreach_get("center", cen); cen = cen.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        nr = np.empty(nf * 3); me.polygons.foreach_get("normal", nr); nr = nr.reshape(-1, 3) @ np.linalg.inv(Mw[:3, :3])
        nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-12
        C_.append(cen); N_.append(nr); own.append((o.name, nf))
    if not C_: return {}
    C = np.concatenate(C_); N = np.concatenate(N_); n = len(C)
    # orientation-free plane key: flip so the largest-|component| of the normal is positive
    ax = np.abs(N).argmax(1); sgn = np.sign(N[np.arange(n), ax]); sgn[sgn == 0] = 1
    Nc = N * sgn[:, None]; d = (Nc * C).sum(1)
    key = np.c_[np.round(Nc / 0.02).astype(np.int64), np.round(d / 0.006).astype(np.int64)]
    _, g = np.unique(key, axis=0, return_inverse=True); g = g.ravel(); ng = g.max() + 1
    front = ((T - C) * N).sum(1) >= 0                       # faces facing the field-centre point are kept
    has_f = np.bincount(g, weights=front, minlength=ng) > 0; has_b = np.bincount(g, weights=~front, minlength=ng) > 0
    both = has_f & has_b
    lo = np.full((ng, 3), np.inf); hi = np.full((ng, 3), -np.inf); fi = np.flatnonzero(front)
    for k in range(3): np.minimum.at(lo[:, k], g[fi], C[fi, k]); np.maximum.at(hi[:, k], g[fi], C[fi, k])
    pad = 0.6
    inside = ((C >= lo[g] - pad) & (C <= hi[g] + pad)).all(1)
    kill = ~front & both[g] & inside
    out, off = {}, 0
    for name, nf in own: out[name] = kill[off:off + nf]; off += nf
    return out

def reset_custom_normals(objs):
    """the GLB custom split normals of many faces point opposite to the face (exporter wrote back-face copies with
    front normals): Cycles then shades them from behind -> pitch-black walls / stand fronts / floors. Drop the custom
    normals (flat architecture, faces are not smooth-shaded anyway) so shading follows the geometry."""
    n = 0
    for o in objs:
        me = o.data
        if not me.has_custom_normals: continue
        try:
            me.attributes.remove(me.attributes["custom_normal"])
        except Exception:
            me.normals_split_custom_set([(0.0, 0.0, 0.0)] * len(me.loops))
        me.update(); n += 1
    return n

# ------------------------------------------------------------------ 1) clear the union area (once per stadium)
def clear_union(objs, A, carpet):
    t0 = time.time(); zc = A["floor_z"]          # heights measured from the arena floor (4 mm .. 3.5 m)
    P_clear = union_polys(A, carpet, 0.5); P_prop = union_polys(A, carpet, 2.5)
    nkill = nprop = 0
    dup = duplicate_faces(objs, A)
    for o in objs:
        me = o.data; nf = len(me.polygons)
        if nf == 0: continue
        V = world_verts(o)
        ls = np.empty(nf, np.int64); me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(nf, np.int64); me.polygons.foreach_get("loop_total", lt)
        lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
        cen = np.add.reduceat(V[lv], ls, axis=0) / lt[:, None]
        zr = cen[:, 2] - zc
        kill = in_union(to_anchor(cen, A), P_clear) & (zr >= 0.004) & (zr <= 3.5)
        # loose props: geometric islands (glTF splits vertices at seams -> merge by position first)
        q = np.unique(np.round(V / 0.001).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
        E = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", E); E = q[E.reshape(-1, 2)]
        lab = components(q.max() + 1, E)
        comp_v = np.unique(lab[q], return_inverse=True)[1].ravel(); nc = comp_v.max() + 1
        Va = to_anchor(V, A); zv = V[:, 2] - zc
        mn = np.full((nc, 3), np.inf); mx = np.full((nc, 3), -np.inf)
        W = np.c_[Va, zv]
        for k in range(3): np.minimum.at(mn[:, k], comp_v, W[:, k]); np.maximum.at(mx[:, k], comp_v, W[:, k])
        near = np.bincount(comp_v, weights=in_union(Va, P_prop).astype(float), minlength=nc) > 0
        comp_f = comp_v[lv[ls]]; fcount = np.bincount(comp_f, minlength=nc)
        prop = (np.maximum(mx[:, 0] - mn[:, 0], mx[:, 1] - mn[:, 1]) < 15) & (mn[:, 2] < 1.0) & (mx[:, 2] > 0.004) & near & (fcount <= 50000)
        pk = prop[comp_f] & ~kill; kill |= prop[comp_f]
        kill |= dup.get(o.name, np.zeros(nf, bool))
        if kill.any():
            bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.flatnonzero(kill)], context="FACES")
            bm.to_mesh(me); bm.free(); me.update()
        nkill += int(kill.sum()); nprop += int(pk.sum())
    nfix = reset_custom_normals(objs)
    log(f"clear_union: reset imported custom normals on {nfix} meshes (many were flipped vs. the face -> black walls)")
    log(f"clear_union: deleted {nkill} faces ({nprop} as loose props, {sum(int(d.sum()) for d in dup.values())} coplanar back-face twins) over {len(P_clear)} placement pieces in {time.time() - t0:.1f}s")
    return {"faces_deleted": nkill, "prop_faces": nprop}

# ------------------------------------------------------------------ 2) arena floor material
def split_floor(objs, A, mat):
    """faces at floor level (outside the carpet; everything inside was the field area) -> ARENA_FLOOR material"""
    n = 0
    for o in objs:
        me = o.data; nf = len(me.polygons)
        if nf == 0: continue
        cen = np.empty(nf * 3); me.polygons.foreach_get("center", cen); cen = cen.reshape(-1, 3)
        nr = np.empty(nf * 3); me.polygons.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
        Mw = np.array(o.matrix_world); cen = cen @ Mw[:3, :3].T + Mw[:3, 3]
        nr = nr @ np.linalg.inv(Mw[:3, :3]); nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-12
        sel = (np.abs(nr[:, 2]) > 0.95) & (np.abs(cen[:, 2] - A["floor_z"]) < 0.25)   # GLBs carry back-face copies (nz<0)
        if not sel.any(): continue
        if mat.name not in me.materials: me.materials.append(mat)
        idx = list(me.materials).index(mat)
        mi = np.empty(nf, np.int32); me.polygons.foreach_get("material_index", mi); mi[sel] = idx
        me.polygons.foreach_set("material_index", mi); me.update(); n += int(sel.sum())
    log(f"split_floor: {n} floor faces -> {mat.name}")
    return n

def floor_patch(A, carpet, mat, collection, margin=2.0):
    """one quad 3 mm below the arena floor over the whole cleared union (+margin): fills holes left by deleted
    boards / stages so the area around the field is never a black void; the real floor (if present) stays on top"""
    P = np.concatenate(union_polys(A, carpet, 0.5 + margin)); lo, hi = P.min(0), P.max(0)
    c, s_ = math.cos(A["yaw"]), math.sin(A["yaw"]); cx, cy = A["center_xy"]; z = A["floor_z"] - 0.003
    pts = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    V = [(cx + c * x - s_ * y, cy + s_ * x + c * y, z) for x, y in pts]
    me = bpy.data.meshes.new("ARENA_FLOOR_PATCH"); me.from_pydata(V, [], [(0, 1, 2, 3)]); me.materials.append(mat)
    ob = bpy.data.objects.new("ARENA_FLOOR_PATCH", me); collection.objects.link(ob)
    return ob

def floor_slab(A, carpet, seats, mat, collection, drop=0.40, pad=8.0):
    """one big quad 0.40 m below the arena floor under the whole bowl (seat footprint + pad, at least the union + 30 m):
    wherever the model has no floor (under / between stands, around rinks with a lower concourse, holes after the
    clear-out) the camera sees a dark concrete floor instead of a black void. Below every real floor -> never on top."""
    P = np.concatenate(union_polys(A, carpet, 30.0)); lo, hi = P.min(0), P.max(0)
    if seats is not None and len(seats["P"]):
        Sa = to_anchor(seats["P"].astype(np.float64), A); lo = np.minimum(lo, Sa.min(0) - pad); hi = np.maximum(hi, Sa.max(0) + pad)
    lo = np.maximum(lo, -150); hi = np.minimum(hi, 150)
    c, s_ = math.cos(A["yaw"]), math.sin(A["yaw"]); cx, cy = A["center_xy"]; z = A["floor_z"] - drop
    pts = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    V = [(cx + c * x - s_ * y, cy + s_ * x + c * y, z) for x, y in pts]
    me = bpy.data.meshes.new("ARENA_FLOOR_SLAB"); me.from_pydata(V, [], [(0, 1, 2, 3)]); me.materials.append(mat)
    ob = bpy.data.objects.new("ARENA_FLOOR_SLAB", me); collection.objects.link(ob)
    log(f"floor slab: {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} m at floor-{drop} m")
    return ob

def floor_fill_slots(A, carpet, bvh, spacing=7.0, size=5.0):
    """ring(s) of soft down-facing fill lights over the floor around the field (between carpet and stands).
    Positions are found once per stadium: below any overhang / upper deck (up-ray), never inside geometry, and only
    where there is floor (or our slab) below. 'k' = mean overlap so per-frame energy hits the target irradiance."""
    fz = A["floor_z"]; c, s_ = math.cos(A["yaw"]), math.sin(A["yaw"]); cx, cy = A["center_xy"]; slots = []
    up, dn = Vector((0, 0, 1)), Vector((0, 0, -1))
    for margin in (3.0, 9.0, 16.0):
        Q = np.concatenate(union_polys(A, carpet, margin)); hull = Q[list(convex_hull_2d([tuple(q) for q in Q]))]
        seg = np.roll(hull, -1, 0) - hull; L = np.linalg.norm(seg, axis=1); per = L.sum(); n = max(6, int(per / spacing))
        cum = np.r_[0, np.cumsum(L)]
        for t in np.linspace(0, per, n, endpoint=False):
            i = min(np.searchsorted(cum, t, side="right") - 1, len(L) - 1); q = hull[i] + seg[i] * ((t - cum[i]) / L[i])
            x, y = cx + c * q[0] - s_ * q[1], cy + s_ * q[0] + c * q[1]
            g = bvh.ray_cast(Vector((x, y, fz + 0.3)), dn, 1.0)                 # floor (or nothing -> slab) below?
            if g[0] is not None and g[0].z < fz - 0.6: continue
            h = bvh.ray_cast(Vector((x, y, fz + 0.3)), up, 40.0)
            ceil = 40.0 if h[0] is None else h[0].z - fz
            hh = min(6.5, ceil - 0.8)
            if hh < 2.4: continue                                                  # under seating / inside structure
            slots.append({"pos": [float(x), float(y), float(fz + hh)], "h": float(hh), "size": size, "ring": margin})
    # overlap factor: irradiance at each slot's floor point from all slots (unit target each), no occlusion
    if slots:
        Pp = np.array([sl["pos"] for sl in slots]); H = np.array([sl["h"] for sl in slots]); Ar = size * size
        F = Pp.copy(); F[:, 2] = fz
        d = F[:, None, :] - Pp[None]; r2 = (d ** 2).sum(2); cosr = (Pp[None, :, 2] - F[:, None, 2]) / np.sqrt(r2 + 1e-9)
        w = (np.pi * H ** 2 + Ar)[None] / np.pi * cosr ** 2 / (r2 + Ar / np.pi)    # contribution if each light alone gives 1 below itself
        k = float(w.sum(1).mean())
        for sl in slots: sl["k"] = k
    log(f"floor fill: {len(slots)} lights (overlap {slots[0]['k'] if slots else 0:.2f})")
    return slots

# ------------------------------------------------------------------ 3) seats
def stadium_tris(objs):
    Vs, Ts, Ms, off = [], [], [], 0
    for o in objs:
        me = o.data; me.calc_loop_triangles(); nt = len(me.loop_triangles)
        if nt == 0: continue
        V = world_verts(o); T = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", T)
        mi = np.empty(nt, np.int64); me.loop_triangles.foreach_get("material_index", mi)
        sat = np.zeros(max(1, len(me.materials)))
        for i, m in enumerate(me.materials):
            b = m.node_tree.nodes.get("Principled BSDF") if (m and m.use_nodes) else None
            if b is not None:
                c = m.get("orig_col") or list(b.inputs["Base Color"].default_value); sat[i] = max(c[:3]) - min(c[:3])
        Vs.append(V); Ts.append(T.reshape(-1, 3) + off); Ms.append(sat[np.clip(mi, 0, len(sat) - 1)]); off += len(V)
    return np.concatenate(Vs), np.concatenate(Ts), np.concatenate(Ms)

def cache_key(A, carpet):
    st = os.stat(f"{ROOT}/{A['glb']}")
    h = hashlib.md5(json.dumps([A, carpet, st.st_size, int(st.st_mtime), SEAT_ALGO_VERSION], sort_keys=True).encode()).hexdigest()[:10]
    return f"{CACHE_DIR}/{A['name']}_{h}.npz"

def detect_seats(objs, A, carpet, bvh=None, tris=None):
    path = cache_key(A, carpet)
    if os.path.exists(path):
        d = np.load(path); log(f"seats: {len(d['P'])} from cache {os.path.basename(path)}")
        return {k: d[k] for k in d.files}
    t0 = time.time()
    V, T, SAT = tris if tris is not None else stadium_tris(objs)
    if bvh is None: bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(cr, axis=1); nz = cr[:, 2] / (2 * area + 1e-12)
    cen = (a + b + c) / 3; zr = cen[:, 2] - A["floor_z"]
    out = ~in_union(to_anchor(cen, A), union_polys(A, carpet, 1.5))
    up = (np.abs(nz) > 0.6) & (zr > 0.3)        # 0.6: some models draw seating as a sloped ramp with stripes & (zr < 25) & out & (area > 1e-3)
    # above 12 m something (roof / upper deck) must be above the triangle -> drops roof tops; lower bowls may be open
    # (some models have no roof over the lower bowl, e.g. arena)
    cand = np.flatnonzero(up & (zr >= 12.0)); upv0 = Vector((0, 0, 1))
    for i in cand:
        p = cen[i]
        if bvh.ray_cast(Vector((p[0], p[1], p[2] + 0.05)), upv0, 120.0)[0] is None: up[i] = False
    ids = np.flatnonzero(up); tot = area[ids].sum()
    dens = max(0.06, tot / 900000.0)                    # m^2 per candidate point (cap ~900k candidates)
    rng = np.random.default_rng(int(hashlib.md5(A["name"].encode()).hexdigest()[:8], 16))
    cnt = rng.poisson(area[ids] / dens); tid = np.repeat(ids, cnt)
    r1, r2 = rng.random(len(tid)), rng.random(len(tid)); s1 = np.sqrt(r1)
    P = a[tid] * (1 - s1)[:, None] + b[tid] * (s1 * (1 - r2))[:, None] + c[tid] * (s1 * r2)[:, None]
    sat = SAT[tid]
    # outward direction (anchor frame: away from the nearest point of the field core rectangle)
    hx, hy = (carpet["x"][1] - carpet["x"][0]) / 2, (carpet["y"][1] - carpet["y"][0]) / 2
    Pa = to_anchor(P, A); q = np.clip(Pa, [-hx, -hy], [hx, hy]); nA = Pa - q
    nA /= np.linalg.norm(nA, axis=1, keepdims=True) + 1e-9
    N = np.c_[from_anchor_dir(nA, A), np.zeros(len(P))]
    log(f"seats: {len(ids)} upward tris, {tot:.0f} m^2, {len(P)} candidates ({time.time() - t0:.1f}s)")
    # anisotropic greedy thinning: one person per ~0.55 m along the row (tangent) x ~0.75 m across (outward)
    order = rng.permutation(len(P)); cell = 0.8; grid = {}; keep = []
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]; NX, NY = N[:, 0], N[:, 1]
    for i in order:
        kx, ky = int(X[i] // cell), int(Y[i] // cell); ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((kx + dx, ky + dy), ()):
                    if abs(Z[j] - Z[i]) > 0.22: continue
                    ddx, ddy = X[j] - X[i], Y[j] - Y[i]; dn = ddx * NX[i] + ddy * NY[i]; dt = -ddx * NY[i] + ddy * NX[i]
                    if abs(dt) < 0.55 and abs(dn) < 0.75: ok = False; break
                if not ok: break
            if not ok: break
        if ok: grid.setdefault((kx, ky), []).append(i); keep.append(i)
    keep = np.array(keep, np.int64); P, N, sat = P[keep], N[keep], sat[keep]
    log(f"seats: {len(P)} after row thinning ({time.time() - t0:.1f}s)")
    # stepped-tier + headroom tests (raycasts)
    down = Vector((0, 0, -1)); upv = Vector((0, 0, 1)); good = np.zeros(len(P), bool); head = np.zeros(len(P)); rej = [0, 0, 0]
    for i in range(len(P)):
        p = P[i]; n = N[i]
        h = bvh.ray_cast(Vector((p[0], p[1], p[2] + 0.05)), upv, 3.0)
        hr = 3.0 if h[0] is None else (h[0].z - p[2])
        if hr < 1.0:
            # something low above: a big horizontal slab -> we sampled the underside of a tread/deck: reject;
            # small parts (chair pans / backs, rails) are fine: people stand between the chairs -> check above them
            if hr < 0.3 and area[h[2]] > 1.5 and abs(nz[h[2]]) > 0.7: rej[0] += 1; continue
            h = bvh.ray_cast(Vector((p[0], p[1], p[2] + 1.0)), upv, 2.0)
            hr = 3.0 if h[0] is None else (h[0].z - p[2])
        if hr < 1.2: rej[0] += 1; continue
        hb = bvh.ray_cast(Vector((p[0] + 0.9 * n[0], p[1] + 0.9 * n[1], p[2] + 1.6)), down, 3.6)
        hf = bvh.ray_cast(Vector((p[0] - 0.9 * n[0], p[1] - 0.9 * n[1], p[2] + 1.0)), down, 3.6)
        db = -9.0 if hb[0] is None else hb[0].z - p[2]          # rise of the surface behind (away from the field)
        df = 9.0 if hf[0] is None else p[2] - hf[0].z           # drop of the surface in front
        step_b = 0.12 < db < 1.45; step_f = df > 0.12
        if step_b or (step_f and db > -0.12):
            good[i] = True; head[i] = hr
        else: rej[1 if abs(db) <= 0.12 and abs(df) <= 0.12 else 2] += 1
    P, N, sat, head = P[good], N[good], sat[good], head[good]
    res = {"P": P.astype(np.float32), "N": N.astype(np.float32), "head": head.astype(np.float32), "sat": sat.astype(np.float32)}
    log(f"seats: rejected {rej[0]} (headroom < 1.2 m), {rej[1]} (flat), {rej[2]} (other step pattern)")
    log(f"seats: {len(P)} on stepped tiers with headroom (sat>0.15: {(sat > 0.15).mean() * 100 if len(sat) else 0:.0f}%) in {time.time() - t0:.1f}s")
    os.makedirs(CACHE_DIR, exist_ok=True); tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **res); os.replace(tmp, path)
    return res

# ------------------------------------------------------------------ 4) crowd instancing
_PROXIES = {}
def load_proxies():
    if "coll" in _PROXIES: return _PROXIES
    with bpy.data.libraries.load(PROXY_BLEND, link=False) as (src, dst):
        dst.collections = ["CROWD_PROXIES"]
    C = dst.collections[0]
    objs = sorted(C.objects, key=lambda o: o.name)        # Collection Info 'Separate Children' order = name order
    for o in objs: o.pass_index = 0
    w = np.array([0.35 if "child" in o.get("avatar", o.name) else 1.0 for o in objs]); w /= w.sum()
    _PROXIES.update(coll=C, names=[o.name for o in objs], heights=np.array([o.get("height_m", 1.75) for o in objs]), weights=w)
    log(f"crowd proxies: {len(objs)} from {os.path.basename(PROXY_BLEND)}")
    return _PROXIES

def crowd_node_group(C):
    ng = bpy.data.node_groups.get("CROWD_GN")
    if ng: return ng
    ng = bpy.data.node_groups.new("CROWD_GN", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    ci = N.new("GeometryNodeCollectionInfo"); ci.transform_space = "ORIGINAL"
    ci.inputs["Collection"].default_value = C; ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
    iop = N.new("GeometryNodeInstanceOnPoints"); iop.inputs["Pick Instance"].default_value = True
    def attr(name, dt):
        n = N.new("GeometryNodeInputNamedAttribute"); n.data_type = dt; n.inputs["Name"].default_value = name; return n
    occ, pidx, yaw, scl = attr("occ", "BOOLEAN"), attr("pidx", "INT"), attr("yaw", "FLOAT"), attr("scl", "FLOAT")
    cx = N.new("ShaderNodeCombineXYZ"); e2r = N.new("FunctionNodeEulerToRotation")
    L.new(gi.outputs[0], iop.inputs["Points"]); L.new(occ.outputs["Attribute"], iop.inputs["Selection"])
    L.new(ci.outputs[0], iop.inputs["Instance"]); L.new(pidx.outputs["Attribute"], iop.inputs["Instance Index"])
    L.new(yaw.outputs["Attribute"], cx.inputs["Z"]); L.new(cx.outputs[0], e2r.inputs[0]); L.new(e2r.outputs[0], iop.inputs["Rotation"])
    L.new(scl.outputs["Attribute"], iop.inputs["Scale"]); L.new(iop.outputs["Instances"], go.inputs[0])
    return ng

def build_crowd(seats, collection):
    """point cloud at the seats + GN instancer. show_viewport=False keeps it out of scene.ray_cast (camera clip test);
    it still renders."""
    PX = load_proxies(); P = seats["P"]; n = len(P)
    me = bpy.data.meshes.new("CROWD_PTS"); me.vertices.add(n); me.vertices.foreach_set("co", P.reshape(-1).astype(np.float32))
    for nm, t in (("occ", "BOOLEAN"), ("pidx", "INT"), ("yaw", "FLOAT"), ("scl", "FLOAT"), ("shirt", "FLOAT_COLOR"), ("pants", "FLOAT_COLOR")):
        me.attributes.new(nm, t, "POINT")
    ob = bpy.data.objects.new("CROWD", me); collection.objects.link(ob); ob.pass_index = 0
    mod = ob.modifiers.new("crowd", "NODES"); mod.node_group = crowd_node_group(PX["coll"]); mod.show_viewport = False
    ob.hide_viewport = False
    log(f"crowd: {n} seats instanced")
    return {"obj": ob, "n": n, "P": P.astype(np.float64), "N": seats["N"].astype(np.float64), "head": seats["head"].astype(np.float64)}

SHIRT_PALETTE = [[0.80, 0.10, 0.10], [0.05, 0.20, 0.60], [0.95, 0.55, 0.05], [0.10, 0.55, 0.20], [0.45, 0.10, 0.55], [0.95, 0.85, 0.10],
                 [0.05, 0.05, 0.05], [0.92, 0.92, 0.92], [0.00, 0.60, 0.65], [0.55, 0.05, 0.15], [0.20, 0.20, 0.22], [0.10, 0.12, 0.30]]
PANTS_PALETTE = [[0.10, 0.14, 0.25], [0.05, 0.05, 0.06], [0.20, 0.22, 0.30], [0.45, 0.40, 0.30], [0.30, 0.30, 0.32], [0.15, 0.25, 0.45]]
def _lin(c): c = np.asarray(c, float); return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

def occluders(P, yaw, scl, M, cam_pos, carpet, zmax=6.0):
    """True where a fan's silhouette lies on a camera ray that continues into the field volume (carpet x 0..zmax),
    i.e. the fan could cover a robot / fuel / field person. Vectorized slab test on 8 points per fan."""
    Mi = np.array(M.inverted()); C = np.array(Mi[:3, :3] @ np.array(cam_pos) + Mi[:3, 3])
    lat = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], 1)
    hs = []
    for h in (0.05, 1.0, 1.95):
        for s in (-0.35, 0.35): hs.append(P + lat * s + np.array([0, 0, 1.0]) * (h * scl)[:, None])
    for s in (-0.6, 0.6): hs.append(P + lat * s + np.array([0, 0, 1.6]) * scl[:, None])
    lo = np.array([carpet["x"][0], carpet["y"][0], -0.1]); hi = np.array([carpet["x"][1], carpet["y"][1], zmax])
    occ = np.zeros(len(P), bool)
    for Q in hs:
        Qf = Q @ Mi[:3, :3].T + Mi[:3, 3]; D = Qf - C
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - C) / D; t2 = (hi - C) / D
        tmin = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        tmax = np.nan_to_num(np.maximum(t1, t2), nan=np.inf, posinf=np.inf, neginf=-np.inf)
        # axis-parallel ray outside the slab -> no hit
        par = np.abs(D) < 1e-12; outside = par & ((C < lo) | (C > hi))
        tmin = np.where(par, -np.inf, tmin); tmax = np.where(par, np.inf, tmax)
        te = tmin.max(1); tx = tmax.min(1); miss = outside.any(1)
        occ |= (~miss) & (tx > 1.0) & (te < tx) & (tx > 0)
    return occ

def update_crowd(CR, rng, M, cam, carpet, A):
    """cheap per-frame variation (numpy -> attributes). Returns stats for the label JSON."""
    PX = _PROXIES; P, n = CR["P"], CR["n"]
    if n == 0: return {"seats": 0, "fans": 0}
    occ_p = rng.uniform(0.3, 0.9)
    ang = np.arctan2(*(to_anchor(P, A)[:, ::-1].T)); sec = ((ang + math.pi) / (2 * math.pi) * 16).astype(int) % 16
    prob = np.clip(occ_p + rng.uniform(-0.15, 0.15, 16)[sec], 0.03, 0.98)
    occ = rng.random(n) < prob
    fc = M @ Vector((FC[0], FC[1], 0)); yaw = np.arctan2(fc.y - P[:, 1], fc.x - P[:, 0]) + np.radians(rng.uniform(-20, 20, n))
    pidx = rng.choice(len(PX["names"]), n, p=PX["weights"])
    hmax = np.clip((CR["head"] - 0.05) / PX["heights"][pidx], 0.5, 1.2)
    scl = np.minimum(rng.uniform(0.9, 1.1, n), hmax)
    hide = occluders(P, yaw, scl, M, cam.matrix_world.translation, carpet)
    # fans right in front of the lens (big blurry heads at the frame edge): cull within clip+3 m, and within 9 m when in view
    Mi = np.array(cam.matrix_world.inverted()); Hd = P + np.array([0, 0, 1.2]); Qc = Hd @ Mi[:3, :3].T + Mi[:3, 3]
    dist = np.linalg.norm(Qc, axis=1); tx = 18.0 / cam.data.lens * 1.15; ty = tx * 0.5625 * 1.3 + 0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        inview = (Qc[:, 2] < 0) & (np.abs(Qc[:, 0] / -Qc[:, 2]) < tx) & (np.abs(Qc[:, 1] / -Qc[:, 2]) < ty)
    near = (dist < cam.data.clip_start + 3.0) | (inview & (dist < 9.0))
    hide |= near
    occ &= ~hide
    # shirts: team palette / random hue / neutrals, mildly jittered; pants: dark palette
    u = rng.random(n); pal = np.array(SHIRT_PALETTE)[rng.integers(len(SHIRT_PALETTE), size=n)]
    hsv = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(rng.random(n), rng.uniform(0.3, 0.9, n), rng.uniform(0.15, 0.9, n))]) if n else np.zeros((0, 3))
    neu = np.repeat(rng.uniform(0.04, 0.85, (n, 1)), 3, 1)
    shirt = np.where((u < 0.5)[:, None], pal, np.where((u < 0.8)[:, None], hsv, neu)) * rng.uniform(0.85, 1.1, (n, 1))
    shirt = _lin(np.clip(shirt, 0, 1)); pants = _lin(np.array(PANTS_PALETTE)[rng.integers(len(PANTS_PALETTE), size=n)] * rng.uniform(0.8, 1.15, (n, 1)))
    me = CR["obj"].data
    me.attributes["occ"].data.foreach_set("value", occ)
    me.attributes["pidx"].data.foreach_set("value", pidx.astype(np.int32))
    me.attributes["yaw"].data.foreach_set("value", yaw.astype(np.float32))
    me.attributes["scl"].data.foreach_set("value", scl.astype(np.float32))
    me.attributes["shirt"].data.foreach_set("color", np.c_[shirt, np.ones(n)].astype(np.float32).ravel())
    me.attributes["pants"].data.foreach_set("color", np.c_[pants, np.ones(n)].astype(np.float32).ravel())
    me.update()
    return {"seats": int(n), "fans": int(occ.sum()), "occupancy": round(float(occ_p), 3), "hidden_for_camera": int(hide.sum()), "hidden_near_lens": int(near.sum())}

# ------------------------------------------------------------------ 5) stand lights (placement once, energy per frame)
def stand_light_slots(seats, A, bvh):
    """cluster seats by angle sector (8) x height band; one big soft area light over each cluster, below the roof"""
    P = seats["P"]; slots = []
    if len(P) < 50: return slots
    Pa = to_anchor(P, A); ang = np.arctan2(Pa[:, 1], Pa[:, 0]); sec = ((ang + math.pi) / (2 * math.pi) * 8).astype(int) % 8
    band = (P[:, 2] - A["floor_z"] > 9.0).astype(int)
    for s in range(8):
        for b in (0, 1):
            m = (sec == s) & (band == b)
            if m.sum() < 40: continue
            Q = P[m]; c = Q.mean(0); spread = 2.0 * math.sqrt(Q[:, 0].var() + Q[:, 1].var())
            h = bvh.ray_cast(Vector((c[0], c[1], c[2] + 2.0)), Vector((0, 0, 1)), 40.0)
            d = 12.0 if h[0] is None else min(12.0, max(3.0, (h[0].z - c[2]) - 1.0))
            slots.append({"pos": [float(c[0]), float(c[1]), float(c[2] + d)], "size": float(np.clip(spread, 8, 30)), "d": float(d), "n": int(m.sum())})
    return slots

def field_irradiance(lights, p):
    """approximate irradiance (W/m^2) on a horizontal patch at p from area / spot / point lights"""
    E = 0.0
    for ob in lights:
        L = ob.data; pos = ob.matrix_world.translation; d = Vector(p) - pos; r2 = max(d.length_squared, 1e-4); dn = d.normalized()
        cos_r = max(0.0, -dn.z)
        if L.type == "AREA":
            nrm = (ob.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized(); cos_e = max(0.0, nrm.dot(dn))
            Ar = L.size * (L.size_y if L.shape in ("RECTANGLE", "ELLIPSE") else L.size)
            E += L.energy / math.pi * cos_e * cos_r / (r2 + Ar / math.pi)
        elif L.type in ("SPOT", "POINT"):
            if L.type == "SPOT":
                ax = (ob.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
                if ax.dot(dn) < math.cos(L.spot_size / 2): continue
            E += L.energy / (4 * math.pi) * cos_r / r2
    return E
