"""Render frc-frame JSONs with Blender/Cycles and write per-frame label JSON (bumpers + fuel boxes).
blender -b -P render/blender_render.py -- --frames f1.json f2.json ... --out OUTDIR [--res 1920x1080] [--samples 64] [--cpu]
Writes OUTDIR/raw/<frame_id>.png (clean render) and OUTDIR/labels/<frame_id>.json; post.py makes the final jpg + YOLO txt.
Frames are rendered in the order given; the stadium is reloaded only when it changes (sort by stadium for speed)."""
import bpy, bmesh, sys, os, json, math, random, time, argparse
from mathutils import Matrix, Vector, Euler
import numpy as np
ROOT = os.environ.get("FRC_ROOT", "/workspace/frc")
sys.path.insert(0, f"{ROOT}/tools/people"); sys.path.insert(0, f"{ROOT}/render")
import people_blender as pb
import bumpers as BU
import stadium_prep as SP

ap = argparse.ArgumentParser()
ap.add_argument("--frames", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--res", default="1920x1080"); ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--cpu", action="store_true"); ap.add_argument("--no-people", action="store_true")
ap.add_argument("--no-crowd", action="store_true", help="no procedural stand crowd (A/B checks)")
ap.add_argument("--no-denoise", action="store_true")
ap.add_argument("--camera-check", default=None, help="no rendering: sample cameras for --frames, project robots, write stats JSON here")
ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--shutter", type=float, default=0.5)
A = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
RW, RH = [int(v) for v in A.res.split("x")]
os.makedirs(f"{A.out}/raw", exist_ok=True); os.makedirs(f"{A.out}/labels", exist_ok=True)
MANIFEST = {r["key"]: r for r in json.load(open(f"{ROOT}/manifest.json"))["robots"]}
FIELD_CARPET = json.load(open(f"{ROOT}/models/field/carpet_placeholder.json"))
FONTS = json.load(open(f"{ROOT}/render/assets/fonts/fonts.json"))
IDX_BUMPER, IDX_FUEL, IDX_ROBOT = 1000, 3000, 100     # object pass_index ranges

def log(*a): print("[render]", *a, flush=True)

# ---------------------------------------------------------------- scene setup
def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    dev = "CPU"
    if not A.cpu:
        for t in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = t; prefs.get_devices()
                if any(d.type == t for d in prefs.devices):
                    for d in prefs.devices: d.use = d.type == t
                    dev = t; break
            except Exception: pass
    sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
    log("device", dev)
    sc.cycles.samples = A.samples; sc.cycles.use_denoising = not A.no_denoise
    try: sc.cycles.denoiser = "OPTIX" if dev == "OPTIX" else "OPENIMAGEDENOISE"
    except Exception: pass
    sc.cycles.max_bounces = 6; sc.cycles.transparent_max_bounces = 12; sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = RW, RH, 100
    sc.render.fps = int(A.fps); sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = A.shutter
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGB"
    vl = sc.view_layers[0]; vl.use_pass_object_index = True
    sc.frame_start = sc.frame_end = sc.frame_current = 10
    return sc

def coll(name):
    c = bpy.data.collections.get(name)
    if c is None: c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
    return c

def clear_coll(name):
    c = bpy.data.collections.get(name)
    if not c: return
    for o in list(c.all_objects): bpy.data.objects.remove(o, do_unlink=True)
    for b in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.actions, bpy.data.curves, bpy.data.images, bpy.data.lights):
        for d in list(b):
            if d.users == 0: b.remove(d)

def import_glb(path, collection):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in list(o.users_collection): c.objects.unlink(o)
        collection.objects.link(o)
    return new

def roots(objs): return [o for o in objs if o.parent is None]

def principled(name, rgb, rough=0.5, metal=0.0, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m

def srgb2lin(c): return [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c]

# ---------------------------------------------------------------- stadium / field / carpet (per stadium)
STATE = {"stadium": None, "M": None}
def load_venue(venue, rng):
    clear_coll("venue")
    C = coll("venue")
    A_ = json.load(open(f"{ROOT}/{venue['anchor']}"))
    objs = import_glb(f"{ROOT}/{A_['glb']}", C)
    STATE["stadium"] = venue["stadium"]; STATE["stadium_objs"] = [o for o in objs if o.type == "MESH"]; STATE["stadium_names"] = {o.name for o in STATE["stadium_objs"]}
    # field + carpet are imported in field frame and parented to an empty that carries field_to_stadium
    fe = bpy.data.objects.new("FIELD", None); C.objects.link(fe); STATE["field_empty"] = fe
    fo = import_glb(f"{ROOT}/models/field/rebuilt_field.glb", C)
    for o in roots(fo): o.parent = fe
    co = import_glb(f"{ROOT}/models/field/carpet_placeholder.glb", C)
    for o in roots(co): o.parent = fe
    field_materials([o for o in fo if o.type == "MESH"])
    carpet_material([o for o in co if o.type == "MESH"])
    # ---- heavy per-stadium prep, ONCE per load (not per frame) ----
    t0 = time.time(); carpet = FIELD_CARPET["extent_field_m"]; A_["name"] = venue["stadium"]
    for m in {m for o in STATE["stadium_objs"] for m in o.data.materials if m}:      # remember original colours
        b = m.node_tree.nodes.get("Principled BSDF") if m.use_nodes else None
        if b is not None and "orig_col" not in m: m["orig_col"] = list(b.inputs["Base Color"].default_value)
    STATE["clear"] = SP.clear_union(STATE["stadium_objs"], A_, carpet)
    STATE["floor_mat"] = arena_floor_material(); SP.split_floor(STATE["stadium_objs"], A_, STATE["floor_mat"]); SP.floor_patch(A_, carpet, STATE["floor_mat"], C)
    # sorted -> deterministic rng consumption (a set's order changes between runs)
    STATE["stadium_mats"] = sorted({m for o in STATE["stadium_objs"] for m in o.data.materials if m and m != STATE["floor_mat"]}, key=lambda m: m.name)
    tris = SP.stadium_tris(STATE["stadium_objs"]); STATE["bvh"] = SP.BVHTree.FromPolygons(tris[0].tolist(), tris[1].tolist(), all_triangles=True)
    seats = SP.detect_seats(STATE["stadium_objs"], A_, carpet, STATE["bvh"], tris)
    STATE["crowd"] = SP.build_crowd(seats, C) if not A.no_crowd else None
    STATE["stand_slots"] = SP.stand_light_slots(seats, A_, STATE["bvh"]); STATE["anchor"] = A_
    SP.floor_slab(A_, carpet, seats, STATE["floor_mat"], C)                    # no black void under / between the stands
    STATE["fill_slots"] = SP.floor_fill_slots(A_, carpet, STATE["bvh"])          # lights over the floor ring, below overhangs
    STATE["prep_s"] = round(time.time() - t0, 1)
    log(f"stadium prep {venue['stadium']}: {STATE['prep_s']}s, seats={len(seats['P'])}, stand lights={len(STATE['stand_slots'])}")

TEX = f"{ROOT}/render/assets"
def arena_floor_material():
    """floor around the field (was e.g. white hockey ice -> blow-out). Concrete texture x random tint per frame."""
    m = bpy.data.materials.new("ARENA_FLOOR"); m.use_nodes = True; nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.6
    tint = nt.nodes.new("ShaderNodeRGB"); tint.name = "tint"; tint.outputs[0].default_value = (0.08, 0.08, 0.08, 1)
    path = f"{TEX}/concrete/smooth_concrete_floor_diff_1k.jpg"
    if os.path.exists(path):
        geo = nt.nodes.new("ShaderNodeNewGeometry"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1 / 3.0, 1 / 3.0, 1)
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path)
        nt.links.new(geo.outputs["Position"], mp.inputs[0]); nt.links.new(mp.outputs[0], t.inputs[0])
        bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(t.outputs[0], bw.inputs[0])
        k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 2.0; nt.links.new(bw.outputs[0], k.inputs[0])
        mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"; mix.inputs["Factor"].default_value = 1.0
        nt.links.new(tint.outputs[0], mix.inputs["A"]); nt.links.new(k.outputs[0], mix.inputs["B"]); nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    else:
        nt.links.new(tint.outputs[0], b.inputs["Base Color"])
    return m

def randomize_floor(rng):
    """dark/mid concrete, black flooring or blue-grey covers; linear albedo 0.06..0.28"""
    u = rng.random(); v = rng.uniform(0.06, 0.28)
    c = [v, v, v] if u < 0.6 else ([v * 0.8, v * 0.9, v * 1.2] if u < 0.85 else [v * 1.1, v, v * 0.85])
    STATE["floor_mat"].node_tree.nodes["tint"].outputs[0].default_value = (*c, 1)
    STATE["floor_mat"].node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = rng.uniform(0.35, 0.8)
    return round(v, 3)

_ALU = {}
def alu_material(name, rgb, rough):
    """brushed aluminium (CC0 ambientCG Metal011), box-projected in object space; falls back to flat metal"""
    m = principled(name, rgb, rough=rough, metal=1.0)
    cp, rp = f"{TEX}/metal/brushed_alu_color.jpg", f"{TEX}/metal/brushed_alu_roughness.jpg"
    if not (os.path.exists(cp) and os.path.exists(rp)): return m
    if "c" not in _ALU:
        _ALU["c"] = bpy.data.images.load(cp); _ALU["r"] = bpy.data.images.load(rp); _ALU["r"].colorspace_settings.name = "Non-Color"
        _ALU["c"].use_fake_user = True; _ALU["r"].use_fake_user = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (2.5, 2.5, 2.5)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    tcol = nt.nodes.new("ShaderNodeTexImage"); tcol.image = _ALU["c"]; tcol.projection = "BOX"; tcol.projection_blend = 0.3
    trou = nt.nodes.new("ShaderNodeTexImage"); trou.image = _ALU["r"]; trou.projection = "BOX"; trou.projection_blend = 0.3
    nt.links.new(mp.outputs[0], tcol.inputs[0]); nt.links.new(mp.outputs[0], trou.inputs[0])
    bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(tcol.outputs[0], bw.inputs[0])
    k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 1 / 0.36; nt.links.new(bw.outputs[0], k.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"; mix.inputs["Factor"].default_value = 1.0
    mix.inputs["A"].default_value = (*rgb, 1); nt.links.new(k.outputs[0], mix.inputs["B"]); nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = max(0.05, rough - 0.12); mr.inputs["To Max"].default_value = min(1, rough + 0.15)
    nt.links.new(trou.outputs[0], mr.inputs["Value"]); nt.links.new(mr.outputs[0], b.inputs["Roughness"])
    return m

def field_materials(objs):
    cache = {}
    for o in objs:
        cat = o.name.split("__")[0]
        col = None
        if o.data.color_attributes:
            a = o.data.color_attributes[0]; v = [0, 0, 0]
            try:
                arr = np.zeros(len(a.data) * 4, np.float32); a.data.foreach_get("color", arr); v = arr.reshape(-1, 4)[:, :3].mean(0).tolist()
            except Exception: pass
            col = v
        col = col or [0.6, 0.6, 0.6]
        key = (cat, tuple(round(c, 2) for c in col))
        if key not in cache:
            lum = sum(col) / 3; sat = max(col) - min(col)
            if cat in ("perimeter", "hub", "outpost", "tower") and lum > 0.7 and sat < 0.1:     # light grey/white panels -> polycarbonate
                m = principled(f"POLYCARB_{cat}", [0.9, 0.92, 0.95], rough=0.05, **{"Transmission Weight": 0.92, "IOR": 1.58})
            elif sat < 0.1 and 0.25 < lum <= 0.7:                                                # mid grey -> aluminium
                m = alu_material(f"ALU_{cat}", [0.75, 0.76, 0.78], rough=0.35)
            elif sat < 0.1:                                                                       # dark -> black plastic / powder coat
                m = principled(f"BLACK_{cat}", [max(col[0], 0.02)] * 3, rough=0.55)
            else:                                                                                 # alliance red/blue, orange
                m = principled(f"PAINT_{cat}", srgb2lin(col) if max(col) > 0.9 else col, rough=0.45)
            cache[key] = m
        o.data.materials.clear(); o.data.materials.append(cache[key])

def carpet_material(objs):
    """Placeholder for Shaw Neyland II 20: grey low-pile with fine noise. A real texture can be dropped in at
    render/assets/carpet/neyland_ii_20.(png|jpg) (tile size in carpet_tile_m.txt, default 0.6 m)."""
    m = bpy.data.materials.new("CARPET_NEYLAND_II_20"); m.use_nodes = True; nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.95
    tex_path = next((f"{ROOT}/render/assets/carpet/neyland_ii_20.{e}" for e in ("png", "jpg") if os.path.exists(f"{ROOT}/render/assets/carpet/neyland_ii_20.{e}")), None)
    uv = nt.nodes.new("ShaderNodeUVMap"); mp = nt.nodes.new("ShaderNodeMapping"); nt.links.new(uv.outputs[0], mp.inputs[0])
    if tex_path:
        tile = float(open(f"{ROOT}/render/assets/carpet/carpet_tile_m.txt").read()) if os.path.exists(f"{ROOT}/render/assets/carpet/carpet_tile_m.txt") else 0.6
        mp.inputs["Scale"].default_value = (1 / tile, 1 / tile, 1)
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(tex_path); nt.links.new(mp.outputs[0], t.inputs[0])
        nt.links.new(t.outputs[0], b.inputs["Base Color"])
        bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(t.outputs[0], bw.inputs[0])
        bm = nt.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = 0.2; bm.inputs["Distance"].default_value = 0.002
        nt.links.new(bw.outputs[0], bm.inputs["Height"]); nt.links.new(bm.outputs[0], b.inputs["Normal"])
    else:
        n = nt.nodes.new("ShaderNodeTexNoise"); n.inputs["Scale"].default_value = 180; n.inputs["Detail"].default_value = 6
        nt.links.new(mp.outputs[0], n.inputs["Vector"])
        r = nt.nodes.new("ShaderNodeValToRGB"); r.color_ramp.elements[0].color = (0.055, 0.058, 0.064, 1); r.color_ramp.elements[1].color = (0.13, 0.135, 0.145, 1)
        nt.links.new(n.outputs["Fac"], r.inputs[0]); nt.links.new(r.outputs[0], b.inputs["Base Color"])
        bm = nt.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = 0.25; nt.links.new(n.outputs["Fac"], bm.inputs["Height"]); nt.links.new(bm.outputs[0], b.inputs["Normal"])
    for o in objs: o.data.materials.clear(); o.data.materials.append(m)

def place_field(M):
    STATE["field_empty"].matrix_world = M

# ---------------------------------------------------------------- per-frame randomization
STADIUM_ALBEDO_CAP = 0.3
def randomize_stadium_colors(rng):
    """small hue/value jitter of every stadium material + random seat colour for saturated materials (seats)"""
    for m in STATE["stadium_mats"]:          # sorted list (deterministic)
        if not m.use_nodes: continue
        b = m.node_tree.nodes.get("Principled BSDF")
        if b is None: continue
        if "orig_col" not in m: m["orig_col"] = list(b.inputs["Base Color"].default_value)
        c = list(m["orig_col"])
        mx, mn = max(c[:3]), min(c[:3])
        if mx - mn > 0.15 and rng.random() < 0.7:         # saturated -> likely seats/branding: new hue
            h = rng.random(); import colorsys; r, g, bb = colorsys.hsv_to_rgb(h, rng.uniform(0.4, 0.9), mx)
            c = [r, g, bb, 1]
        else:
            k = rng.uniform(0.8, 1.2); c = [min(1, x * k) for x in c[:3]] + [1]
        mxc = max(c[:3])
        if mxc > STADIUM_ALBEDO_CAP:      # white stadium paint (boards, ice, seats) blows out next to the dark carpet
            c = [x * STADIUM_ALBEDO_CAP / mxc for x in c[:3]] + [1]
        b.inputs["Base Color"].default_value = c
        if b.inputs["Alpha"].default_value < 0.99 and not b.inputs["Alpha"].is_linked:   # rink / box glass: keep it subtle
            b.inputs["Alpha"].default_value = min(b.inputs["Alpha"].default_value, 0.08)

def setup_lights(rng, M, frame):
    clear_coll("lights"); C = coll("lights")
    w = bpy.data.worlds.get("W") or bpy.data.worlds.new("W"); bpy.context.scene.world = w; w.use_nodes = True
    bg = w.node_tree.nodes["Background"]; bg.inputs[1].default_value = rng.uniform(0.05, 0.4)
    bg.inputs[0].default_value = (0.5, 0.5, 0.55, 1)
    kelvin = rng.uniform(3800, 6800); col = kelvin_rgb(kelvin)
    h = rng.uniform(10, 22); nx, ny = int(rng.integers(2, 5)), int(rng.integers(2, 4))
    total = rng.uniform(1.2e4, 4.0e4)          # watts over the whole rig
    for i in range(nx):
        for j in range(ny):
            L = bpy.data.lights.new(f"rig{i}{j}", "AREA"); L.shape = "RECTANGLE"; L.size = rng.uniform(2, 6); L.size_y = rng.uniform(2, 6)
            L.energy = total / (nx * ny) * rng.uniform(0.7, 1.3); L.color = [min(1, c * rng.uniform(0.95, 1.05)) for c in col]
            ob = bpy.data.objects.new(L.name, L); C.objects.link(ob)
            ob.location = M @ Vector((-4 + 24.5 * (i + 0.5) / nx + rng.normal(0, 1), -4 + 16 * (j + 0.5) / ny + rng.normal(0, 1), h))
    if rng.random() < 0.5:   # a few broadcast/stage spots
        for k in range(int(rng.integers(1, 4))):
            L = bpy.data.lights.new(f"spot{k}", "SPOT"); L.energy = rng.uniform(1e3, 6e3); L.spot_size = math.radians(rng.uniform(25, 60)); L.color = kelvin_rgb(rng.uniform(3200, 6000))
            ob = bpy.data.objects.new(L.name, L); C.objects.link(ob)
            p = M @ Vector((rng.uniform(-5, 21), rng.choice([-12.0, 20.0]), rng.uniform(8, 16))); ob.location = p
            ob.rotation_euler = ((M @ Vector((8.27, 4.0, 0))) - p).to_track_quat("-Z", "Y").to_euler()
    # field irradiance (mean of 5 points on the field) -> stand lights relative to it + auto exposure
    bpy.context.view_layer.update()
    pts = [M @ Vector((x, y, 0)) for x, y in ((8.27, 4.035), (2, 1), (14.5, 1), (2, 7), (14.5, 7))]
    E = sum(SP.field_irradiance(list(C.objects), p) for p in pts) / len(pts)
    lr = random_stream(frame, 31)                       # own stream: does not shift robots / fuel / camera draws
    ratio = lr.uniform(0.05, 0.18); hk = lr.uniform(3000, 5200)
    slots = STATE.get("stand_slots") or []
    for k, sl in enumerate(slots):    # dim, soft house lights over each stand section (invisible to the camera)
        L = bpy.data.lights.new(f"stand{k}", "AREA"); L.shape = "SQUARE"; L.size = sl["size"]
        L.energy = ratio * E * (sl["size"] ** 2 + math.pi * sl["d"] ** 2) * lr.uniform(0.7, 1.3); L.color = kelvin_rgb(hk + lr.normal(0, 200))
        ob = bpy.data.objects.new(L.name, L); C.objects.link(ob); ob.location = sl["pos"]; ob.visible_camera = False
    fr_ = lr.uniform(0.18, 0.45)      # floor ring around the field (driver stations, scoring table side, aisles, stand fronts)
    for k, sl in enumerate(STATE.get("fill_slots") or []):       # positions fixed per stadium (below overhangs, never inside geometry)
        L = bpy.data.lights.new(f"floorfill{k}", "AREA"); L.shape = "SQUARE"; L.size = sl["size"]; hh = sl["h"]
        L.energy = fr_ * E * (sl["size"] ** 2 + math.pi * hh ** 2) / sl["k"] * lr.uniform(0.75, 1.25); L.color = kelvin_rgb(hk + lr.normal(0, 200))
        ob = bpy.data.objects.new(L.name, L); C.objects.link(ob); ob.location = sl["pos"]; ob.visible_camera = False; ob.visible_glossy = False
    if not slots:                     # no seats detected: old-style house lights aimed outwards
        for k in range(4):
            L = bpy.data.lights.new(f"house{k}", "AREA"); L.size = 20; L.energy = ratio * E * (400 + math.pi * 30 ** 2); L.color = kelvin_rgb(hk)
            ob = bpy.data.objects.new(L.name, L); C.objects.link(ob); ob.visible_camera = False
            d = [(0, -1), (0, 1), (-1, 0), (1, 0)][k]; p = M @ Vector((8.27 + d[0] * 8, 4.0 + d[1] * 4, h))
            ob.location = p; ob.rotation_euler = ((M @ Vector((8.27 + d[0] * 40, 4.0 + d[1] * 30, 8))) - p).to_track_quat("-Z", "Y").to_euler()
    ev = math.log2(E_REF / max(E, 1e-3)) + lr.uniform(-0.35, 0.35)
    bpy.context.scene.view_settings.exposure = ev
    return {"kelvin": round(kelvin), "rig_height_m": round(h, 2), "rig_watts": round(total), "grid": [nx, ny],
            "field_irradiance": round(E, 2), "exposure": round(ev, 3), "stand_ratio": round(ratio, 3), "floor_fill_ratio": round(fr_, 3), "stand_lights": len(slots), "fill_lights": len(STATE.get("fill_slots") or [])}

E_REF = float(os.environ.get("FRC_E_REF", "25.0"))     # field irradiance that maps to exposure 0 (calibrated on CPU tests)
def random_stream(fr, k): return np.random.default_rng([int(fr["seed"]), k])

def kelvin_rgb(k):
    t = k / 100.0
    r = 1.0 if t <= 66 else min(1, max(0, 1.2929 * (t - 60) ** -0.1332))
    g = (0.3901 * math.log(t) - 0.6318) if t <= 66 else 1.1298 * (t - 60) ** -0.0755
    b = 1.0 if t >= 66 else (0 if t <= 19 else 0.5432 * math.log(t - 10) - 1.1962)
    return [min(1, max(0, r)), min(1, max(0, g)), min(1, max(0, b))]

ROBOT_PAINT = [[0.02, 0.02, 0.02], [0.8, 0.8, 0.82], [0.05, 0.1, 0.35], [0.5, 0.02, 0.02], [0.9, 0.45, 0.02], [0.02, 0.25, 0.05],
               [0.25, 0.02, 0.35], [0.9, 0.75, 0.05], [0.0, 0.35, 0.4], [0.35, 0.35, 0.37]]
def robot_materials(objs, rng):
    """CAD colours are flat or missing: re-skin each material as aluminium / powder coat / black plastic / polycarb"""
    team_col = ROBOT_PAINT[rng.integers(len(ROBOT_PAINT))]; acc_col = ROBOT_PAINT[rng.integers(len(ROBOT_PAINT))]
    mats = {}
    for o in objs:
        if o.type != "MESH": continue
        for i, m in enumerate(o.data.materials):
            key = m.name if m else f"none{i}"
            if key not in mats:
                u = rng.random()
                if u < 0.42: nm = alu_material("r_alu", [rng.uniform(0.6, 0.85)] * 3, rough=rng.uniform(0.25, 0.5))
                elif u < 0.62: nm = principled("r_paint", team_col, rough=rng.uniform(0.3, 0.6))
                elif u < 0.75: nm = principled("r_acc", acc_col, rough=rng.uniform(0.3, 0.6))
                elif u < 0.93: nm = principled("r_black", [rng.uniform(0.01, 0.05)] * 3, rough=rng.uniform(0.4, 0.8))
                else: nm = principled("r_pc", [0.9, 0.9, 0.92], rough=0.05, **{"Transmission Weight": 0.85, "IOR": 1.58})
                mats[key] = nm
            o.data.materials[i] = mats[key]
        if len(o.data.materials) == 0:
            o.data.materials.append(mats.setdefault("__default", alu_material("r_alu", [0.75] * 3, rough=0.35)))

def keyframe_motion(ob, M, pose, vel, fps, dz=0.0):
    """place ob at field pose (x,y,z,yaw) via M and keyframe +-1 frame with velocity for motion blur"""
    sc = bpy.context.scene; f0 = sc.frame_current
    for df in (-1, 0, 1):
        t = df / fps
        x = pose["x"] + vel.get("vx", 0) * t; y = pose["y"] + vel.get("vy", 0) * t; z = pose.get("z", 0) + dz + vel.get("vz", 0) * t
        yaw = pose.get("yaw", 0) + vel.get("wz", 0) * t
        T = M @ Matrix.Translation((x, y, z)) @ Euler((pose.get("roll", 0), pose.get("pitch", 0), yaw)).to_matrix().to_4x4()
        ob.matrix_world = T
        ob.keyframe_insert("location", frame=f0 + df); ob.keyframe_insert("rotation_euler", frame=f0 + df)
    sc.frame_set(f0)

# ---------------------------------------------------------------- robots + bumpers
def add_robots(fr, M, rng):
    clear_coll("robots"); C = coll("robots"); out = []
    for k, r in enumerate(fr["robots"]):
        spec = MANIFEST.get(r.get("manifest_key"))
        root = bpy.data.objects.new(f"robot_{r['id']}", None); C.objects.link(root); root.rotation_mode = "XYZ"
        if spec and os.path.exists(f"{ROOT}/{spec['model_file']}"):
            objs = import_glb(f"{ROOT}/{spec['model_file']}", C)
            hide = set(spec.get("bumper_meshes") or [])
            for o in objs:
                if hide and (o.name in hide or o.name.split(".")[0] in hide): o.hide_render = True
                if o.type == "MESH": o.pass_index = IDX_ROBOT + k
            for o in roots(objs): o.parent = root
            robot_materials(objs, rng)
            outline = spec.get("frame_outline_m") or spec.get("footprint_m")
        else:
            L, W = r["footprint"]["length"], r["footprint"]["width"]
            outline = [[L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2]]
        b = BU.build_bumper(f"bumper_{r['id']}", outline, r["footprint"]["bumper_depth"], r["bumper_color"], r["team_number"], rng, FONTS, C)
        for o in b["objects"]: o.parent = root; o.pass_index = IDX_BUMPER + k
        keyframe_motion(root, M, r["pose"], r["velocity"], A.fps)
        out.append({"k": k, "robot": r, "bumper_objs": b["objects"], "bumper_info": b["info"]})
    return out

# ---------------------------------------------------------------- fuel
def add_fuel(fr, M, rng):
    clear_coll("fuel"); C = coll("fuel")
    me = bpy.data.meshes.new("fuel_mesh"); bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.075); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    base = [rng.uniform(0.85, 0.95), rng.uniform(0.62, 0.75), rng.uniform(0.0, 0.05)]
    mats = []
    for i in range(4):
        m = principled(f"fuel{i}", [min(1, c * rng.uniform(0.9, 1.08)) for c in base], rough=rng.uniform(0.55, 0.8), **{"Subsurface Weight": 0.15})
        mats.append(m)
    me.materials.append(mats[0])
    objs = []
    for i, f in enumerate(fr["fuel"]):
        o = bpy.data.objects.new(f"fuel_{f['id']}", me); C.objects.link(o)
        o.pass_index = IDX_FUEL + i; o.rotation_mode = "XYZ"
        slot = o.material_slots[0]; slot.link = "OBJECT"; slot.material = mats[int(rng.integers(4))]
        pose = {"x": f["x"], "y": f["y"], "z": f["z"], "yaw": rng.uniform(0, 6.28), "roll": rng.uniform(0, 6.28)}
        if f.get("state") == "ground" and abs(f.get("vx", 0)) + abs(f.get("vy", 0)) == 0:
            o.matrix_world = M @ Matrix.Translation((f["x"], f["y"], f["z"]))
        else:
            keyframe_motion(o, M, pose, {"vx": f.get("vx", 0), "vy": f.get("vy", 0), "vz": f.get("vz", 0), "wz": rng.uniform(-15, 15)}, A.fps)
        objs.append((i, f, o))
    return objs

# ---------------------------------------------------------------- people
PEOPLE_TPL = {}
def person_instance(path, clip, frame, C):
    """import each avatar GLB once per process (template, not linked to the scene) and hand out copies:
    objects + mesh copies (so shirt recolouring stays per person) sharing images/actions. ~ms instead of ~1.4 s."""
    if path not in PEOPLE_TPL:
        T = bpy.data.collections.get("people_tpl") or bpy.data.collections.new("people_tpl")
        arm = pb.import_person(path, collection=T); T.use_fake_user = True
        for an in dict(arm.get("person_actions", {})).values():      # keep clips alive (clear_coll purges 0-user data)
            if bpy.data.actions.get(an): bpy.data.actions[an].use_fake_user = True
        PEOPLE_TPL[path] = arm
    tpl = PEOPLE_TPL[path]
    arm = tpl.copy(); C.objects.link(arm)
    if arm.animation_data:
        for t in list(arm.animation_data.nla_tracks): arm.animation_data.nla_tracks.remove(t)
        arm.animation_data.action = None
    for ch in tpl.children:
        nc = ch.copy(); nc.parent = arm; nc.matrix_parent_inverse = ch.matrix_parent_inverse.copy()
        if ch.type == "MESH": nc.data = ch.data.copy()
        for md in nc.modifiers:
            if md.type == "ARMATURE": md.object = arm
        C.objects.link(nc)
    pb.set_clip(arm, clip, frame)
    return arm

def add_people(fr, M, rng):
    clear_coll("people"); C = coll("people"); n = 0
    for p in fr.get("people", []):
        try:
            clipinfo = p["clip"]; fr_in_clip = 1 + int(p["clip_time_s"] * 30)
            arm = person_instance(f"{ROOT}/{p['model_file']}", p["clip"], fr_in_clip, C)
            arm.scale = (p["scale"],) * 3
            keyframe_motion(arm, M, p["pose"], p["velocity"], A.fps)
            sh = p["outfit"]["shirt"]
            if sh["pattern"] == "stripes": pb.stripe_shirt(arm, tuple(srgb2lin(sh["colors_srgb"][0])), tuple(srgb2lin(sh["colors_srgb"][1])), stripe_m=sh.get("stripe_width_m", 0.03))
            elif sh["pattern"] == "solid": pb.recolor_shirt(arm, tuple(srgb2lin(sh["color_srgb"])))
            n += 1
        except Exception as e:
            log("person failed", p.get("id"), e)
    return n

# ---------------------------------------------------------------- camera
FIELD_X, FIELD_Y = (-0.6, 17.14), (0.0, 8.07)        # field incl. driver-station walls (field frame, m)
def _cam_basis(pos, fwd, roll=0.0):
    """rotation matrix (columns = cam x, y, z; camera looks along -z) for a forward vector, z-up, optional roll"""
    f = fwd.normalized(); R = f.to_track_quat("-Z", "Y").to_matrix()
    if roll: R = R @ Matrix.Rotation(roll, 3, "Z")
    return R

def _tan_coords(pos, R, pts):
    Ri = R.transposed(); out = []
    for p in pts:
        q = Ri @ (Vector(p) - pos)
        out.append((q.x / -q.z, q.y / -q.z) if q.z < -0.05 else None)
    return out

def _include(pos, R, tx, pts, mx=0.96, my=0.93):
    """keep every point inside the frame WITHOUT changing the yaw (head-on view): zoom out symmetrically and re-tilt"""
    asp = RH / RW
    for _ in range(4):
        T = _tan_coords(pos, R, pts)
        if any(t is None for t in T): return R, tx, False
        ty = tx * asp; xs = [t[0] for t in T]; ys = [t[1] for t in T]
        if max(abs(v) for v in xs) <= tx * mx and max(abs(v) for v in ys) <= ty * my: return R, tx, True
        tx = max(tx, max(abs(v) for v in xs) / mx); ty = tx * asp
        y0, y1 = min(-ty, min(ys) / my), max(ty, max(ys) / my)
        if y1 - y0 > 2 * ty: tx = (y1 - y0) / 2 / asp; ty = tx * asp
        cy = (y0 + y1) / 2
        R = R @ Matrix.Rotation(math.atan(cy), 3, "X")                       # tilt only (about the camera x axis)
    return R, tx, False

def camera_pose(fr, rng):
    """head-on broadcast camera (as the YouTube broadcasts, reference/broadcast/*.png): on the frame's camera_side,
    centred on midfield (x = 8.27 +-0.5 m), looking straight across the field (yaw +-2.5 deg), 8.5-20.5 m behind the
    near guardrail, elevation 11-25 deg to mid-field (~3-9 m up), far guardrail at 84-98 % of the width (nearly edge
    to edge), near guardrail at / just below the bottom edge, roll +-1 deg, small look-at jitter. Robots are kept in
    frame by zooming out / re-tilting only (never re-yawing). Returns pos, rotation (3x3), tan(hfov/2), info."""
    side = fr.get("venue", {}).get("camera_side", "+y"); s = 1 if side == "+y" else -1
    near_y, far_y = (FIELD_Y[1], FIELD_Y[0]) if s > 0 else (FIELD_Y[0], FIELD_Y[1])
    FC3 = Vector((8.27, 4.035, 0.0)); asp = RH / RW
    back = 8.5 + 12.0 * rng.beta(1.6, 2.4); el = math.radians(rng.uniform(11, 25))   # >= 3.5 m behind foreground people
    hgt = float(np.clip(math.tan(el) * (back + 4.035), 3.0, 13.0)); xoff = float(np.clip(rng.normal(0, 0.25), -0.5, 0.5))
    pos = Vector((8.27 + xoff, near_y + s * back, hgt))
    yaw_j = math.radians(float(np.clip(rng.normal(0, 1.0), -2.5, 2.5)))
    yaw_dir = Matrix.Rotation(yaw_j, 3, "Z") @ Vector((0, -s, 0))           # perpendicular to the long axis
    far_frac = rng.uniform(0.84, 0.98); vb = rng.uniform(-0.06, 0.12)       # near guardrail height in frame (0 = bottom edge)
    far = [Vector((x, far_y, z)) for x in FIELD_X for z in (0.0, 1.2)]
    tx = math.tan(math.radians(25)); pitch = math.atan2(hgt, back + 4.035)
    for _ in range(6):        # solve hfov (far guardrail width) and pitch (near guardrail at vb) jointly
        R = _cam_basis(pos, yaw_dir * math.cos(pitch) - Vector((0, 0, math.sin(pitch))))
        T = _tan_coords(pos, R, far); tx = max(abs(t[0]) for t in T) / far_frac
        nn = _tan_coords(pos, R, [Vector((pos.x, near_y, 0.0))])[0]
        want = -tx * asp * (1 - 2 * vb)
        pitch += math.atan(want) - math.atan(nn[1])
    R = _cam_basis(pos, yaw_dir * math.cos(pitch) - Vector((0, 0, math.sin(pitch))))
    R = R @ Matrix.Rotation(rng.normal(0, 0.006), 3, "X")                  # look-at (tilt) jitter
    roll = float(np.clip(rng.normal(0, 0.007), -0.017, 0.017))
    keep_all = rng.random() < 0.97; fit = True
    if keep_all:
        rp = [Vector((x, y, z)) for r in fr.get("robots", []) for x, y in (r["footprint"].get("outline_xy") or [[r["pose"]["x"], r["pose"]["y"]]]) for z in (0.0, 0.35)]
        if rp: R, tx, fit = _include(pos, R, tx, rp)
    R = R @ Matrix.Rotation(roll, 3, "Z")
    tx = min(max(tx, math.tan(math.radians(9))), math.tan(math.radians(52)))
    fwd = R @ Vector((0, 0, -1)); hit_t = -pos.z / fwd.z if fwd.z < -1e-3 else 30.0
    info = {"mode": "side", "side": side, "pos_field_m": [round(v, 3) for v in pos], "aim_ground_field_m": [round(v, 3) for v in (pos + fwd * hit_t)],
            "yaw_dev_deg": round(math.degrees(yaw_j), 2), "elev_deg": round(math.degrees(math.atan2(pos.z, (pos.xy - FC3.xy).length)), 2),
            "far_frac": round(far_frac, 3), "reframed_for_robots": bool(keep_all), "robots_fit": bool(fit), "roll_rad": round(roll, 4)}
    return pos, R, tx, info

def sample_camera(fr, M, rng):
    """place the scene camera (pose from camera_pose), clip stadium geometry between camera and field, focus/DOF,
    pan motion blur. rng = the camera's own stream (random_stream(fr, 34))."""
    sc = bpy.context.scene
    cam = bpy.data.objects.get("CAM") or bpy.data.objects.new("CAM", bpy.data.cameras.new("CAM"))
    if cam.name not in sc.collection.objects: sc.collection.objects.link(cam)
    sc.camera = cam; cam.animation_data_clear()
    pos, R, tx, info = camera_pose(fr, rng)
    cam.matrix_world = M @ (Matrix.Translation(pos) @ R.to_4x4())
    cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = 36; cam.data.lens = 18.0 / tx; cam.data.clip_end = 600
    # the camera may sit in the stands: clip away seats/rails between it and the field (rays to the field perimeter)
    dg = bpy.context.evaluated_depsgraph_get(); cw = M @ pos; clip = 0.3
    dmin = min((Vector((float(np.clip(pos.x, *FIELD_X)), float(np.clip(pos.y, *FIELD_Y)), 0.0)) - pos).length, 60.0)
    per = [Vector((x, y, z)) for x in np.linspace(*FIELD_X, 5) for y in FIELD_Y for z in (0.1, 1.0)] + \
          [Vector((x, y, z)) for x in FIELD_X for y in np.linspace(*FIELD_Y, 3) for z in (0.1, 1.0)]
    for q in per:
        qw = M @ q; d = (qw - cw); dist = d.length; d.normalize(); o = cw; travelled = 0.0
        for _ in range(20):
            hit, loc, nrm, fi, ob, mx = sc.ray_cast(dg, o, d, distance=max(0.0, dist - travelled - 1.5))
            if not hit or ob is None or ob.name not in STATE.get("stadium_names", set()): break
            travelled = (loc - cw).length; clip = max(clip, travelled + 0.3); o = loc + d * 0.01
    cam.data.clip_start = min(clip, max(0.3, dmin - 2.0))
    # depth of field: focus on the action (field centre +-), f/2.8..11 -> slight softness only on near/far extremes
    dfc = (Vector((8.27, 4.035, 0.3)) - pos).length
    cam.data.dof.use_dof = bool(rng.random() < 0.6)
    cam.data.dof.focus_distance = dfc * rng.uniform(0.85, 1.15); cam.data.dof.aperture_fstop = rng.uniform(2.8, 11.0)
    # operator pan / tilt during the exposure -> motion blur (35 % of frames)
    f0 = sc.frame_current; moving = rng.random() < 0.35
    pan = rng.normal(0, 0.15) if moving else 0.0; tilt = rng.normal(0, 0.04) if moving else 0.0
    base = cam.matrix_world.copy()
    for df in (-1, 0, 1):
        t = df / A.fps
        cam.matrix_world = base @ (Matrix.Rotation(pan * t, 4, "Y") @ Matrix.Rotation(tilt * t, 4, "X"))
        cam.keyframe_insert("location", frame=f0 + df); cam.keyframe_insert("rotation_euler", frame=f0 + df)
    sc.frame_set(f0)
    hfov = 2 * math.degrees(math.atan(tx))
    info["cam_to_field"] = [[round(v, 5) for v in row] for row in (M.inverted() @ base)]      # camera pose (field frame) at the frame time
    info.update({"lens_mm": round(cam.data.lens, 2), "clip_start_m": round(cam.data.clip_start, 2), "sensor_mm": 36, "hfov_deg": round(hfov, 2),
                 "pan_rad_s": round(pan, 4), "tilt_rad_s": round(tilt, 4), "dof": cam.data.dof.use_dof,
                 "fstop": round(cam.data.dof.aperture_fstop, 2), "focus_m": round(cam.data.dof.focus_distance, 2)})
    return cam, info

# ---------------------------------------------------------------- labels
def project(cam, pts_world):
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene; out = []
    for p in pts_world:
        v = world_to_camera_view(sc, cam, p)
        out.append((v.x * RW, (1 - v.y) * RH, v.z))
    return out

def amodal_box(cam, pts):
    P = [p for p in project(cam, pts) if p[2] > 0]
    if not P: return None
    xs = [p[0] for p in P]; ys = [p[1] for p in P]
    return [min(xs), min(ys), max(xs), max(ys)]

def read_index_pass(path):
    img = bpy.data.images.load(path); img.colorspace_settings.name = "Non-Color"
    w, h = img.size; arr = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(arr)
    bpy.data.images.remove(img)
    a = arr.reshape(h, w, 4)[::-1, :, 0]
    return np.rint(a).astype(np.int32)

def boxes_from_index(idx, lo, hi):
    m = (idx >= lo) & (idx < hi); ys, xs = np.nonzero(m); out = {}
    if len(xs) == 0: return out
    ids = idx[ys, xs]
    order = np.argsort(ids, kind="stable"); ids, xs, ys = ids[order], xs[order], ys[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(ids)]):
        out[int(ids[s])] = [int(xs[s:e].min()), int(ys[s:e].min()), int(xs[s:e].max()) + 1, int(ys[s:e].max()) + 1, int(e - s)]
    return out

def setup_index_output(out_exr):
    """compositor: write the IndexOB pass to a 32-bit EXR next to the beauty render"""
    sc = bpy.context.scene; sc.use_nodes = True; nt = sc.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers"); comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = nt.nodes.new("CompositorNodeOutputFile"); fo.base_path = os.path.dirname(out_exr)
    fo.format.file_format = "OPEN_EXR"; fo.format.color_depth = "32"; fo.format.exr_codec = "ZIP"
    fo.file_slots[0].path = os.path.basename(out_exr).replace(".exr", "_")
    nt.links.new(rl.outputs["IndexOB"], fo.inputs[0])
    return fo

# ---------------------------------------------------------------- main loop
def render_frame(path):
    fr = json.load(open(path)); fid = fr["frame_id"]
    if os.path.exists(f"{A.out}/labels/{fid}.json"): log("skip (done)", fid); return
    t0 = time.time(); rng = np.random.default_rng([int(fr["seed"]), 99])
    venue = fr["venue"]; M = Matrix(venue["field_to_stadium"]); T = {}
    if STATE["stadium"] != venue["stadium"]: load_venue(venue, rng); T["load"] = round(time.time() - t0, 2)
    t1 = time.time(); place_field(M); randomize_stadium_colors(rng); floor_v = randomize_floor(random_stream(fr, 32))
    sc = bpy.context.scene; sc.frame_set(10)
    light_info = setup_lights(rng, M, fr); light_info["floor_albedo"] = floor_v; T["setup"] = round(time.time() - t1, 2); t1 = time.time()
    robots = add_robots(fr, M, rng); T["robots"] = round(time.time() - t1, 2); t1 = time.time()
    fuel = add_fuel(fr, M, rng)
    npeople = 0 if A.no_people else add_people(fr, M, rng); T["people"] = round(time.time() - t1, 2); t1 = time.time()
    cam, cam_info = sample_camera(fr, M, random_stream(fr, 34))     # own stream (camera independent of robots / fuel draws)
    crowd_info = {"seats": 0, "fans": 0}
    if STATE.get("crowd"):
        crowd_info = SP.update_crowd(STATE["crowd"], random_stream(fr, 33), M, cam, FIELD_CARPET["extent_field_m"], STATE["anchor"])
    T["camera_crowd"] = round(time.time() - t1, 2); t1 = time.time()
    if os.environ.get("FRC_DEBUG_OVERVIEW"):      # debug only: wide view of the stands (labels are meaningless)
        cam.animation_data_clear(); cam.data.clip_start = 0.5; cam.data.lens = 14
        pos = Vector((8.27, -38.0, 28.0)); cam.matrix_world = M @ (Matrix.Translation(pos) @ (Vector((8.27, 12.0, 0)) - pos).to_track_quat("-Z", "Y").to_matrix().to_4x4())
    if os.environ.get("FRC_PROBE"): probe(cam, M)
    tmpdir = f"{A.out}/tmp"; os.makedirs(tmpdir, exist_ok=True)
    fo = setup_index_output(f"{tmpdir}/{fid}.exr")
    sc.render.filepath = f"{A.out}/raw/{fid}.png"
    bpy.ops.render.render(write_still=True); T["render"] = round(time.time() - t1, 2)
    exr = f"{tmpdir}/{fid}_{sc.frame_current:04d}.exr"
    idx = read_index_pass(exr)
    vb = boxes_from_index(idx, IDX_BUMPER, IDX_BUMPER + 100); vf = boxes_from_index(idx, IDX_FUEL, IDX_FUEL + 5000)
    vr = boxes_from_index(idx, IDX_ROBOT, IDX_ROBOT + 100)
    dg = bpy.context.evaluated_depsgraph_get()
    bumpers = []
    for R in robots:
        k = R["k"]; r = R["robot"]
        pts = [o.matrix_world @ Vector(c) for o in R["bumper_objs"] if o.type == "MESH" and not o.name.startswith("num") for c in o.bound_box]
        am = amodal_box(cam, pts); v = vb.get(IDX_BUMPER + k)
        bumpers.append({"robot_id": r["id"], "team_number": r["team_number"], "alliance": r["alliance"], "color": r["bumper_color"],
                        "class": "bumper_" + r["bumper_color"], "bbox_visible": v[:4] if v else None, "visible_px": v[4] if v else 0,
                        "bbox_amodal": [round(x, 1) for x in am] if am else None,
                        "robot_visible_px": vr.get(IDX_ROBOT + k, [0] * 5)[4], "numbers": R["bumper_info"]})
    fuels = []
    for i, f, o in fuel:
        v = vf.get(IDX_FUEL + i)
        c = o.matrix_world.translation; rr = 0.075
        am = amodal_box(cam, [c + Vector((dx, dy, dz)) for dx in (-rr, rr) for dy in (-rr, rr) for dz in (-rr, rr)])
        if v is None and (am is None or am[2] < 0 or am[0] > RW or am[3] < 0 or am[1] > RH): continue
        fuels.append({"fuel_id": f["id"], "state": f.get("state"), "class": "fuel", "bbox_visible": v[:4] if v else None,
                      "visible_px": v[4] if v else 0, "bbox_amodal": [round(x, 1) for x in am] if am else None})
    lab = {"schema": "frc-labels/v1", "frame_id": fid, "image": f"raw/{fid}.png", "width": RW, "height": RH,
           "source_frame": os.path.abspath(path), "stadium": venue["stadium"], "camera": cam_info, "lights": light_info,
           "render": {"samples": A.samples, "fps": A.fps, "shutter": A.shutter, "engine": "CYCLES", "people": npeople,
                      "seconds": round(time.time() - t0, 1), "timing_s": T, "crowd": crowd_info},
           "bumpers": bumpers, "fuel": fuels}
    json.dump(lab, open(f"{A.out}/labels/{fid}.json", "w"), indent=1)
    os.remove(exr)
    log(f"{fid} {venue['stadium']} bumpers={sum(1 for b in bumpers if b['bbox_visible'])}/{len(bumpers)} fuel={sum(1 for f in fuels if f['bbox_visible'])} people={npeople} fans={crowd_info['fans']} {time.time() - t0:.1f}s {T}")

def probe(cam, M):
    """debug: what does each pixel of a 16x9 grid see (object / material / height above the arena floor)"""
    sc = bpy.context.scene; dg = bpy.context.evaluated_depsgraph_get(); fz = STATE["anchor"]["floor_z"]
    from bpy_extras.view3d_utils import region_2d_to_vector_3d
    fr_ = cam.data.view_frame(scene=sc); Mw = cam.matrix_world
    for j in range(9):
        row = []
        for i in range(16):
            u, v = (i + 0.5) / 16, 1 - (j + 0.5) / 9
            p = fr_[2].lerp(fr_[1], u); q = fr_[3].lerp(fr_[0], u); loc_c = p.lerp(q, v)
            o = Mw.translation; d = (Mw.to_3x3() @ loc_c).normalized()
            hit, loc, n, fi, ob, mx = sc.ray_cast(dg, o + d * cam.data.clip_start, d)
            if not hit: row.append("--"); continue
            mat = ob.material_slots[ob.data.polygons[fi].material_index].material.name[:10] if ob.type == "MESH" and ob.material_slots and fi < len(ob.data.polygons) else "?"
            row.append(f"{ob.name[:12]}|{mat}|{loc.z - fz:.2f}")
        log("PROBE", j, " ; ".join(row))

def camera_check(out):
    """fast (projection-only) check of the camera distribution: are all robots inside the frame?"""
    rows = []
    for p in A.frames:
        fr = json.load(open(p)); pos, R, tx, info = camera_pose(fr, random_stream(fr, 34)); ty = tx * RH / RW; full = part = 0
        for r in fr.get("robots", []):
            pts = [Vector((x, y, z)) for x, y in r["footprint"]["outline_xy"] for z in (0.0, 0.2)]
            T = _tan_coords(pos, R, pts); ins = [t is not None and abs(t[0]) <= tx and abs(t[1]) <= ty for t in T]
            full += all(ins); part += any(ins)
        info.update({"frame": fr["frame_id"], "robots": len(fr.get("robots", [])), "full_in_frame": full, "any_in_frame": part,
                     "lens_mm": round(18 / tx, 1), "hfov_deg": round(2 * math.degrees(math.atan(tx)), 1)})
        rows.append(info)
    n = len(rows); allin = sum(r["full_in_frame"] == r["robots"] for r in rows)
    summ = {"frames": n, "all_robots_fully_in_frame_pct": round(100 * allin / max(n, 1), 1),
            "yaw_dev_deg_max": max(abs(r["yaw_dev_deg"]) for r in rows), "x_offset_m_max": round(max(abs(r["pos_field_m"][0] - 8.27) for r in rows), 3),
            "lens_mm": [min(r["lens_mm"] for r in rows), float(np.median([r["lens_mm"] for r in rows])), max(r["lens_mm"] for r in rows)],
            "height_m": [round(min(r["pos_field_m"][2] for r in rows), 2), round(float(np.median([r["pos_field_m"][2] for r in rows])), 2), round(max(r["pos_field_m"][2] for r in rows), 2)],
            "elev_deg": [min(r["elev_deg"] for r in rows), float(np.median([r["elev_deg"] for r in rows])), max(r["elev_deg"] for r in rows)]}
    json.dump({"summary": summ, "frames": rows}, open(out, "w"), indent=1); log("camera-check", json.dumps(summ))

if A.camera_check:
    camera_check(A.camera_check); sys.exit(0)
setup_scene()
frames = A.frames
for p in frames:
    try: render_frame(p)
    except Exception as e:
        import traceback; traceback.print_exc(); log("FAILED", p, e)
log("done")
