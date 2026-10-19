#!/usr/bin/env python3
"""Convert SketchUp COLLADA stadiums -> GLB (one per stadium).

  python3 tools/convert_stadiums.py <name> [<name> ...]

Input : /workspace/frc/models/stadiums/raw/<name>/model.dae  (+ textures in model/)
Output: /workspace/frc/models/stadiums/glb/<name>.glb

Conventions (identical to tools/convert_robots.py robot GLBs):
  * metres: DAE coordinates are multiplied by the DAE <unit meter="..."> value exactly once
    (pycollada does NOT apply <unit>, so there is no double scaling).
  * scene frame is Z-up; GLB vertices are stored per the glTF spec (+Y up):
    scene (x, y, z) -> glTF (x, z, -y). trimesh/Blender glTF importers that map Y-up -> Z-up
    recover the original Z-up scene; a raw trimesh load gives Y-up exactly like the robot GLBs.
  * origin untouched, no simplification. Every DAE <material> is written with its exact name
    attribute; textures are embedded byte-for-byte (original JPEG/PNG).
  * geometry is grouped into one mesh/node per material (node & mesh named after the material).
  * SketchUp edge <lines> primitives are not surfaces and are not written (counted in the report).
"""
import sys, os, json, struct, time, re
import numpy as np
import collada
from collada import scene as cscene, triangleset, polylist, lineset, material as cmat

ROOT = "/workspace/frc/models/stadiums"
TO_GLTF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # Z-up scene -> glTF Y-up (same as robots)


def color4(c):
    if c is None or isinstance(c, cmat.Map):
        return None
    c = list(np.asarray(c, float).ravel())
    return (c + [1.0] * 4)[:4] if len(c) >= 3 else None


def mat_props(m, daedir, images, img_index):
    e = m.effect
    tex = None
    base = color4(e.diffuse)
    if isinstance(e.diffuse, cmat.Map):
        p = e.diffuse.sampler.surface.image.path
        full = os.path.normpath(os.path.join(daedir, p))
        if not os.path.isfile(full):
            print(f"  WARNING texture missing for {m.name}: {full}")
            return color4(None) or [0.8, 0.8, 0.8, 1.0], None, 1.0, e.shadingtype
        if full not in img_index:
            img_index[full] = len(images); images.append(full)
        tex = img_index[full]
        base = [1.0, 1.0, 1.0, 1.0]
    if base is None:  # SketchUp edge materials: <constant> with the colour in <transparent>
        base = color4(e.emission) or color4(e.transparent) or [0.8, 0.8, 0.8, 1.0]
        alpha = 1.0
    else:
        alpha = 1.0
        if e.shadingtype != "constant" and e.transparent is not None and not isinstance(e.transparent, cmat.Map):
            t = color4(e.transparent); tr = 1.0 if e.transparency is None else float(e.transparency)
            if e.opaque_mode == cmat.OPAQUE_MODE.RGB_ZERO:
                alpha = 1.0 - float(np.mean(t[:3])) * tr
            else:
                alpha = t[3] * tr
        elif e.transparency is not None and e.transparent is None:
            pass
    base = [float(x) for x in base[:3]] + [float(np.clip(alpha, 0, 1))]
    rough = 1.0
    try:
        if e.shininess is not None and not isinstance(e.shininess, cmat.Map):
            rough = float(np.sqrt(2.0 / (2.0 + float(e.shininess))))
    except Exception:
        pass
    return base, tex, rough, e.shadingtype


def convert(name):
    t0 = time.time()
    daedir = f"{ROOT}/raw/{name}"
    dae = f"{daedir}/model.dae"
    c = collada.Collada(dae, ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])
    unit = c.assetInfo.unitmeter; unitname = c.assetInfo.unitname; up = c.assetInfo.upaxis
    if unit is None:
        raise RuntimeError("no <unit> tag")
    if up not in ("Z_UP", collada.asset.UP_AXIS.Z_UP):
        raise RuntimeError(f"unexpected up axis {up}")
    scale = float(unit)  # applied once

    # ---- materials: all of them, in DAE order, exact names
    images, img_index = [], {}
    mats, mat_idx = [], {}
    for m in c.materials:
        base, tex, rough, shading = mat_props(m, daedir, images, img_index)
        nm = m.name if m.name is not None else m.id
        mat_idx[m.id] = len(mats)
        mats.append(dict(id=m.id, name=nm, base=base, tex=tex, rough=rough, shading=shading))
    src_names = [x["name"] for x in mats]

    # ---- geometry
    groups = {}  # mat index -> lists
    n_lines = 0; n_line_prims = 0; n_tri_src = 0; n_mirror = 0; n_nomat = 0; n_nonormal = 0
    for bg in c.scene.objects("geometry"):
        M = np.asarray(bg.matrix, float)
        M3 = M[:3, :3]
        det = np.linalg.det(M3)
        try:
            NM = np.linalg.inv(M3)  # row-vector normals: n' = n @ inv(M3)
        except np.linalg.LinAlgError:
            NM = None
        for p in bg.primitives():
            if isinstance(p, lineset.BoundLineSet):
                n_line_prims += 1; n_lines += len(p); continue
            if isinstance(p, polylist.BoundPolylist):
                p = p.triangleset()
            if not isinstance(p, triangleset.BoundTriangleSet):
                print("  skip primitive", type(p).__name__); continue
            if p.vertex is None or len(p.vertex_index) == 0:
                continue
            vi = p.vertex_index
            V = p.vertex[vi]  # (n,3,3) already transformed by pycollada
            n = len(vi); n_tri_src += n
            orig = p.original
            if orig.normal is not None and orig.normal_index is not None and NM is not None:
                N = orig.normal[orig.normal_index] @ NM
            else:
                n_nonormal += n
                fn = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
                N = np.repeat(fn[:, None, :], 3, axis=1)
            UV = None
            if len(p.texcoordset) > 0:
                UV = p.texcoordset[0][p.texcoord_indexset[0]]
            if det < 0:  # mirrored instance: keep front faces pointing outward
                n_mirror += n
                V = V[:, ::-1]; N = N[:, ::-1]
                UV = UV[:, ::-1] if UV is not None else None
            if p.material is not None and getattr(p.material, "id", None) in mat_idx:
                mi = mat_idx[p.material.id]
            else:
                n_nomat += n
                if "__default__" not in mat_idx:
                    mat_idx["__default__"] = len(mats)
                    mats.append(dict(id="__default__", name="__default__", base=[0.8, 0.8, 0.8, 1.0], tex=None, rough=1.0, shading="lambert"))
                mi = mat_idx["__default__"]
            g = groups.setdefault(mi, dict(V=[], N=[], UV=[]))
            g["V"].append(V.reshape(-1, 3)); g["N"].append(N.reshape(-1, 3))
            g["UV"].append(UV.reshape(-1, 2) if UV is not None else None)
    del c

    # ---- build GLB
    blob = bytearray(); bviews = []; accs = []

    def add_view(arr, target):
        while len(blob) % 4: blob.append(0)
        off = len(blob); b = arr.tobytes(); blob.extend(b)
        bviews.append(dict(buffer=0, byteOffset=off, byteLength=len(b), target=target))
        return len(bviews) - 1

    def add_acc(arr, ctype, typ, target, minmax=False):
        a = dict(bufferView=add_view(arr, target), componentType=ctype, count=int(arr.shape[0]), type=typ)
        if minmax:
            a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
        accs.append(a); return len(accs) - 1

    meshes, nodes = [], []
    tot_tris = 0; tot_verts = 0
    bmin = np.full(3, np.inf); bmax = np.full(3, -np.inf)
    for mi in sorted(groups):
        g = groups[mi]
        V = np.concatenate(g["V"]) * scale
        bmin = np.minimum(bmin, V.min(0)); bmax = np.maximum(bmax, V.max(0))
        N = np.concatenate(g["N"])
        ln = np.linalg.norm(N, axis=1, keepdims=True); bad = ln[:, 0] < 1e-12
        N = np.where(bad[:, None], [0, 0, 1], N / np.where(ln > 1e-12, ln, 1))
        has_uv = any(u is not None for u in g["UV"])
        if has_uv:
            UV = np.concatenate([u if u is not None else np.zeros((len(v), 2)) for u, v in zip(g["UV"], g["V"])])
            UV = np.column_stack([UV[:, 0], 1.0 - UV[:, 1]])  # COLLADA v-up -> glTF v-down
        Vg = (V @ TO_GLTF.T).astype(np.float32); Ng = (N @ TO_GLTF.T).astype(np.float32)
        cols = [Vg, Ng] + ([UV.astype(np.float32)] if has_uv else [])
        full = np.ascontiguousarray(np.hstack(cols))
        # lossless de-duplication of identical corners (same position+normal+uv)
        uniq, inv = np.unique(full.view(np.dtype((np.void, full.dtype.itemsize * full.shape[1]))).ravel(), return_inverse=True)
        uv_ = full[np.unique(inv, return_index=True)[1]]
        idx = inv.reshape(-1).astype(np.uint32)
        nt = len(idx) // 3; tot_tris += nt; tot_verts += len(uv_)
        attrs = dict(POSITION=add_acc(np.ascontiguousarray(uv_[:, 0:3]), 5126, "VEC3", 34962, True),
                     NORMAL=add_acc(np.ascontiguousarray(uv_[:, 3:6]), 5126, "VEC3", 34962))
        if has_uv:
            attrs["TEXCOORD_0"] = add_acc(np.ascontiguousarray(uv_[:, 6:8]), 5126, "VEC2", 34962)
        ind = add_acc(idx, 5125, "SCALAR", 34963)
        nm = mats[mi]["name"]
        meshes.append(dict(name=nm, primitives=[dict(attributes=attrs, indices=ind, material=mi, mode=4)]))
        nodes.append(dict(name=nm, mesh=len(meshes) - 1))
        del V, N, full, uv_, idx, Vg, Ng
    groups.clear()

    gimages, textures = [], []
    for i, path in enumerate(images):
        data = open(path, "rb").read()
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext)
        if mime is None:  # convert anything else to PNG
            from PIL import Image; import io
            b = io.BytesIO(); Image.open(path).save(b, format="PNG"); data = b.getvalue(); mime = "image/png"
        gimages.append(dict(name=os.path.basename(path), mimeType=mime, bufferView=add_view(np.frombuffer(data, np.uint8), None)))
        textures.append(dict(source=i, sampler=0))
    for bv in bviews:
        if bv.get("target") is None: bv.pop("target", None)

    gmats = []
    for m in mats:
        pbr = dict(baseColorFactor=m["base"], metallicFactor=0.0, roughnessFactor=m["rough"])
        if m["tex"] is not None:
            pbr["baseColorTexture"] = dict(index=m["tex"])
        gm = dict(name=m["name"], pbrMetallicRoughness=pbr, doubleSided=False)
        if m["base"][3] < 0.999:
            gm["alphaMode"] = "BLEND"
        elif m["tex"] is not None and images[m["tex"]].lower().endswith(".png"):
            from PIL import Image
            im = Image.open(images[m["tex"]])
            if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                gm["alphaMode"] = "BLEND"
        gm["extras"] = dict(collada_id=m["id"], collada_shading=str(m["shading"]))
        gmats.append(gm)

    bb_scene = [bmin.round(4).tolist(), bmax.round(4).tolist()]
    gltf = dict(
        asset=dict(version="2.0", generator="frc/tools/convert_stadiums.py (pycollada)",
                   extras=dict(source=dae, units="meters", source_unit=f"{unitname} ({unit} m)", source_up_axis="Z_UP",
                               axes="GLB stored per glTF spec (+Y up): scene (x,y,z) -> glTF (x,z,-y); same as robot GLBs",
                               scene_bbox_zup_m=bb_scene)),
        scene=0, scenes=[dict(name=name, nodes=list(range(len(nodes))))],
        nodes=nodes, meshes=meshes, materials=gmats, accessors=accs, bufferViews=bviews,
        samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)],
        buffers=[dict(byteLength=0)])
    if gimages:
        gltf["images"] = gimages; gltf["textures"] = textures
    while len(blob) % 4: blob.append(0)
    gltf["buffers"][0]["byteLength"] = len(blob)
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = f"{ROOT}/glb/{name}.glb"; os.makedirs(os.path.dirname(out), exist_ok=True)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(out + ".tmp", "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(blob)
    os.replace(out + ".tmp", out)
    rep = dict(name=name, glb=out, bytes=total, unit=unit, unitname=unitname, scale=scale,
               tris=tot_tris, verts=tot_verts, src_tris=n_tri_src, mirrored_tris=n_mirror, tris_no_material=n_nomat,
               tris_no_normals=n_nonormal, edge_line_prims_skipped=n_line_prims, edge_segments_skipped=n_lines,
               materials_src=len(src_names), materials_glb=len(gmats), materials_with_triangles=len(meshes),
               mesh_material_names=[m["name"] for m in meshes], src_material_names=src_names,
               textures=len(gimages), texture_files=[os.path.basename(p) for p in images],
               bbox_zup_m=bb_scene, seconds=round(time.time() - t0, 1))
    return rep


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        r = convert(nm)
        json.dump(r, open(f"{ROOT}/glb/{nm}.report.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k not in ("mesh_material_names", "src_material_names", "texture_files")}))
