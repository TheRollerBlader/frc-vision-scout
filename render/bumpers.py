"""Procedural FRC bumpers (Blender). Ring around the robot frame outline (convex hull, simplified), padded/rounded,
saturated red/blue fabric, white team numbers in a random font on some/all straight sides."""
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector, Matrix
ROOT = os.environ.get("FRC_ROOT", "/workspace/frc")

def _hull(P):
    P = sorted(set(map(tuple, np.round(np.asarray(P, float), 4))))
    if len(P) < 3: return P
    def cr(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]       # CCW

def _simplify(H, min_edge=0.04, ang=math.radians(8)):
    H = [np.array(p) for p in H]; changed = True
    while changed and len(H) > 4:
        changed = False
        for i in range(len(H)):
            a, b, c = H[i - 1], H[i], H[(i + 1) % len(H)]
            u, v = b - a, c - b
            turn = abs(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))
            if turn < ang or np.linalg.norm(u) < min_edge:
                H.pop(i); changed = True; break
    return H

def _offset(H, d):
    """mitred outward offset of a CCW convex polygon"""
    n = len(H); out = []
    for i in range(n):
        a, b, c = H[i - 1], H[i], H[(i + 1) % n]
        e1, e2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        n1, n2 = np.array([e1[1], -e1[0]]), np.array([e2[1], -e2[0]])
        m = n1 + n2; m /= np.linalg.norm(m)
        out.append(b + m * d / max(m @ n1, 0.3))
    return out

def fabric(name, color, rng):
    if color == "red": base = [rng.uniform(0.55, 0.75), rng.uniform(0.005, 0.03), rng.uniform(0.008, 0.035)]
    else: base = [rng.uniform(0.005, 0.03), rng.uniform(0.04, 0.12), rng.uniform(0.45, 0.7)]
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1); b.inputs["Roughness"].default_value = rng.uniform(0.35, 0.6)
    b.inputs["Sheen Weight"].default_value = rng.uniform(0.2, 0.6); b.inputs["Specular IOR Level"].default_value = rng.uniform(0.4, 0.7)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    weave = nt.nodes.new("ShaderNodeTexWave"); weave.inputs["Scale"].default_value = rng.uniform(250, 500); weave.inputs["Distortion"].default_value = 2
    wr = nt.nodes.new("ShaderNodeTexNoise"); wr.inputs["Scale"].default_value = rng.uniform(4, 12); wr.inputs["Detail"].default_value = 3
    nt.links.new(tc.outputs["Object"], weave.inputs["Vector"]); nt.links.new(tc.outputs["Object"], wr.inputs["Vector"])
    add = nt.nodes.new("ShaderNodeMath"); add.operation = "ADD"; nt.links.new(weave.outputs["Fac"], add.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = rng.uniform(2, 6)
    nt.links.new(wr.outputs["Fac"], mul.inputs[0]); nt.links.new(mul.outputs[0], add.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = rng.uniform(0.08, 0.25); bump.inputs["Distance"].default_value = 0.002
    nt.links.new(add.outputs[0], bump.inputs["Height"]); nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m, base

def build_bumper(name, outline, depth, color, number, rng, FONTS, collection):
    H = _simplify(_hull(outline)); O = _offset(H, depth)
    zb = rng.uniform(0.02, 0.07); hgt = rng.uniform(0.115, 0.14); zt = zb + hgt
    n = len(H); me = bpy.data.meshes.new(name); bm = bmesh.new()
    ib = [bm.verts.new((p[0], p[1], zb)) for p in H]; it = [bm.verts.new((p[0], p[1], zt)) for p in H]
    ob_ = [bm.verts.new((p[0], p[1], zb)) for p in O]; ot = [bm.verts.new((p[0], p[1], zt)) for p in O]
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((ot[i], ot[j], it[j], it[i]))      # top
        bm.faces.new((ob_[i], ib[i], ib[j], ob_[j]))    # bottom
        bm.faces.new((ob_[i], ob_[j], ot[j], ot[i]))    # outer
        bm.faces.new((ib[i], it[i], it[j], ib[j]))      # inner
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    o = bpy.data.objects.new(name, me); collection.objects.link(o)
    bev = o.modifiers.new("pad", "BEVEL"); bev.width = min(0.03, hgt * rng.uniform(0.18, 0.32)); bev.segments = 4; bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(35); bev.profile = rng.uniform(0.5, 0.75)
    try: o.modifiers.new("ws", "WEIGHTED_NORMAL")
    except Exception: pass
    mat, base = fabric(name + "_fab", color, rng); me.materials.append(mat)
    objs = [o]; info = {"z_bottom": round(zb, 4), "height": round(hgt, 4), "depth": depth, "fabric_rgb": [round(x, 3) for x in base], "sides": []}
    # numbers
    font_e = FONTS["fonts"][rng.integers(len(FONTS["fonts"]))]
    fpath = f"{ROOT}/{font_e['file']}"
    font = bpy.data.fonts.get(os.path.basename(fpath)) or bpy.data.fonts.load(fpath, check_existing=True)
    white = rng.uniform(0.75, 0.95); nh = hgt * rng.uniform(0.62, 0.85)
    nm = bpy.data.materials.new(name + "_num"); nm.use_nodes = True; bb = nm.node_tree.nodes["Principled BSDF"]
    bb.inputs["Base Color"].default_value = (white, white, white, 1); bb.inputs["Roughness"].default_value = rng.uniform(0.3, 0.6)
    edges = sorted(range(n), key=lambda i: -np.linalg.norm(O[(i + 1) % n] - O[i]))
    p_side = rng.uniform(0.6, 1.0)
    for k, i in enumerate(edges[:4]):
        a, b = O[i], O[(i + 1) % n]; L = np.linalg.norm(b - a)
        if L < 0.28 or (k >= 2 and rng.random() > p_side): continue
        e = (b - a) / L; nrm = np.array([e[1], -e[0]])
        cu = bpy.data.curves.new(f"num_{name}_{k}", "FONT"); cu.body = str(number); cu.font = font
        cu.align_x = "CENTER"; cu.align_y = "CENTER"; cu.size = nh / 0.72; cu.extrude = 0.0006
        t = bpy.data.objects.new(f"num_{name}_{k}", cu); collection.objects.link(t); cu.materials.append(nm)
        bpy.context.view_layer.update()
        w = t.dimensions.x
        if w > 0.85 * L: cu.size *= 0.85 * L / w
        Z = Vector((nrm[0], nrm[1], 0)); Y = Vector((0, 0, 1)); X = Y.cross(Z)
        mid = (a + b) / 2; pos = Vector((mid[0], mid[1], (zb + zt) / 2)) + Z * 0.0015
        t.matrix_world = Matrix.Translation(pos) @ Matrix((X, Y, Z)).transposed().to_4x4()
        objs.append(t); info["sides"].append({"edge_len_m": round(float(L), 3), "font": font_e["name"]})
    return {"objects": objs, "info": info}
