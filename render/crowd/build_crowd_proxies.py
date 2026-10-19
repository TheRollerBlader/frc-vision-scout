"""One-time build of low-poly crowd proxies from the 14 Rocketbox avatars (models/people/glb).
blender -b -P render/crowd/build_crowd_proxies.py            -> render/assets/crowd/crowd_proxies.blend
Each avatar is posed with a few clips (idle / cheer / clap), the armature deformation is applied (frozen static mesh),
the parts are joined and decimated to ~TARGET_TRIS triangles. Textures are dropped: every part gets a flat colour
(mean of its texture). Shirt + pants use shared materials that read per-instance colours ("shirt", "pants" instancer
attributes written by crowd.py), so one proxy covers every shirt colour. Proxy: feet at origin, faces +X, metres, Z-up."""
import bpy, sys, os, json, math
import numpy as np
ROOT = os.environ.get("FRC_ROOT", "/workspace/frc")
sys.path.insert(0, f"{ROOT}/tools/people")
import people_blender as pb
TARGET_TRIS = 1400
CLIPS = [("idle_neutral_01", 40), ("cheer_01", 45), ("clap", 25)]
OUT = f"{ROOT}/render/assets/crowd/crowd_proxies.blend"

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene; sc.frame_set(100)
P = json.load(open(f"{ROOT}/models/people/people.json"))["avatars"]

def mean_tex_color(m):
    if not m or not m.use_nodes: return [0.3, 0.3, 0.3]
    b = m.node_tree.nodes.get("Principled BSDF")
    tex = [n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image and n.image.colorspace_settings.name != "Non-Color"]
    if not tex:
        return list(b.inputs["Base Color"].default_value[:3]) if b else [0.3] * 3
    im = tex[0].image; w, h = im.size
    a = np.empty(w * h * 4, np.float32); im.pixels.foreach_get(a); a = a.reshape(-1, 4)[::7]
    a = a[a[:, 3] > 0.5] if (a[:, 3] > 0.5).sum() > 100 else a
    return a[:, :3].mean(0).tolist()          # pixels are linear floats already

def shared_attr_mat(name, attr, fallback):
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = attr
    # attribute missing (e.g. preview without crowd.py) -> black; mix with fallback via alpha-free trick: add fallback*0
    nt.links.new(at.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    m["fallback"] = fallback
    return m

def flat_mat(name, rgb, rough=0.6):
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough
    return m

C = bpy.data.collections.new("CROWD_PROXIES"); sc.collection.children.link(C)
summary = []
for av in P:
    name = av["name"]; path = f"{ROOT}/models/people/{av['file']}"
    for clip, fr in CLIPS:
        before = set(bpy.data.objects)
        arm = pb.import_person(path, clip=clip, frame=fr)
        sc.frame_set(100); dg = bpy.context.evaluated_depsgraph_get()
        new = [o for o in bpy.data.objects if o not in before]
        parts = []
        for o in new:
            # only the skinned body parts (the glTF importer also creates a bone-display icosphere without material)
            if o.type != "MESH" or not any(m.type == "ARMATURE" for m in o.modifiers) or not any(o.data.materials): continue
            ev = o.evaluated_get(dg); me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=False, depsgraph=dg)
            me.transform(o.matrix_world)
            # replace materials with flat / shared ones
            for i, m in enumerate(me.materials):
                base = m.name.split(".")[0] if m else "X"
                if base == av.get("shirt_material", "SHIRT"): nm = shared_attr_mat("CROWD_SHIRT", "shirt", av["shirt_mean_srgb"])
                elif base == av.get("pants_material", "PANTS"): nm = shared_attr_mat("CROWD_PANTS", "pants", av["pants_mean_srgb"])
                else: nm = flat_mat(f"crowd_{name}_{base}", mean_tex_color(m), 0.5 if "HAIR" in base else 0.6)
                me.materials[i] = nm
            while len(me.uv_layers): me.uv_layers.remove(me.uv_layers[0])
            for ca in list(me.color_attributes): me.color_attributes.remove(ca)
            po = bpy.data.objects.new(f"tmp_{o.name}", me); sc.collection.objects.link(po); parts.append(po)
        for o in new: bpy.data.objects.remove(o, do_unlink=True)
        bpy.ops.object.select_all(action="DESELECT")
        for p in parts: p.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]; bpy.ops.object.join()
        ob = bpy.context.view_layer.objects.active; ob.name = f"crowd_{name}_{clip}"; ob.data.name = ob.name
        tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        mod = ob.modifiers.new("dec", "DECIMATE"); mod.ratio = min(1.0, TARGET_TRIS / max(tris, 1)); mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier="dec")
        for p in ob.data.polygons: p.use_smooth = True
        ob.data.validate()
        for c in list(ob.users_collection): c.objects.unlink(ob)
        C.objects.link(ob); ob.pass_index = 0
        v = np.empty(len(ob.data.vertices) * 3, np.float32); ob.data.vertices.foreach_get("co", v); v = v.reshape(-1, 3)
        ob["avatar"] = name; ob["clip"] = clip; ob["height_m"] = float(v[:, 2].max())
        summary.append((ob.name, tris, len(ob.data.polygons), round(float(v[:, 2].max()), 2)))
        for blk in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.images, bpy.data.materials):
            for d in list(blk):
                if d.users == 0: blk.remove(d)
for s in summary: print("[proxy]", s)
sc.collection.children.unlink(C)
C.use_fake_user = True
os.makedirs(os.path.dirname(OUT), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print("[proxy] saved", OUT, len(summary), "proxies")
