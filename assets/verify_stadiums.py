#!/usr/bin/env python3
"""Verify stadium GLBs with trimesh and write glb/README.md."""
import sys, os, json, glob
import numpy as np, trimesh
ROOT = "/workspace/frc/models/stadiums"
FROM_GLTF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # glTF (x,y,z) -> scene Z-up (x,-z,y)

def verify(name):
    rep = json.load(open(f"{ROOT}/glb/{name}.report.json"))
    s = trimesh.load(f"{ROOT}/glb/{name}.glb", force="scene", process=False)
    names, textured = set(), set(); tris = 0
    for g in s.geometry.values():
        tris += len(g.faces)
        m = getattr(g.visual, "material", None)
        if m is not None:
            names.add(m.name)
            if getattr(m, "baseColorTexture", None) is not None:
                textured.add(m.name)
    b = s.bounds  # glTF frame (Y-up)
    corners = np.array(np.meshgrid(*zip(*b))).T.reshape(-1, 3) @ FROM_GLTF.T
    bz = np.array([corners.min(0), corners.max(0)]); ext = bz[1] - bz[0]
    exp_names = set(rep["mesh_material_names"])
    expect_tex = sum(1 for _ in [])  # placeholder
    chk = dict(
        loads=True,
        tris_match=tris == rep["tris"] == rep["src_tris"],
        names_exact=names == exp_names,
        names_subset_of_src=names <= set(rep["src_material_names"]),
        bbox_match=np.allclose(bz, rep["bbox_zup_m"], atol=2e-3),
        horiz_ok=bool(80 <= max(ext[:2]) <= 350),
        z_ok=bool(5 <= ext[2] <= 100),
        textures_present=len(textured) > 0 if rep["textures"] else True,
    )
    return dict(name=name, bytes=os.path.getsize(f"{ROOT}/glb/{name}.glb"), tris=tris, geoms=len(s.geometry),
                mats_loaded=len(names), textured_mats=len(textured), ext=ext.round(2).tolist(), bbox=bz.round(2).tolist(),
                missing=sorted(exp_names - names), extra=sorted(names - exp_names), checks=chk, rep=rep)

if __name__ == "__main__":
    out = [verify(n) for n in sys.argv[1:]]
    for v in out:
        r = v["rep"]
        print(v["name"], f"{v['bytes']/1e6:.1f}MB tris={v['tris']} mats(loaded/with-tris/src)={v['mats_loaded']}/{r['materials_with_triangles']}/{r['materials_src']} "
              f"textured={v['textured_mats']} tex_imgs={r['textures']} ext={v['ext']} min={v['bbox'][0]} checks={v['checks']} missing={v['missing']} extra={v['extra']}")
    json.dump([{k: v for k, v in x.items() if k != "rep"} for x in out], open(f"{ROOT}/glb/verify.json", "w"), indent=1)
