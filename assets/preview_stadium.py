"""Blender preview: field placed in a stadium using its anchor (no jitter).
blender -b -P tools/preview_stadium.py -- <stadium> <out_prefix> [samples]
Renders <out_prefix>_broadcast.png (camera in the stands on the -y side) and <out_prefix>_overview.png."""
import bpy, bmesh, sys, json, math
from mathutils import Matrix, Vector
ROOT = "/workspace/frc"
argv = sys.argv[sys.argv.index("--") + 1:]
name, out = argv[0], argv[1]; samples = int(argv[2]) if len(argv) > 2 else 24
A = json.load(open(f"{ROOT}/models/stadiums/anchors/{name}.json"))
c, s = math.cos(A["yaw"]), math.sin(A["yaw"]); cx, cy = A["center_xy"]; fz = A["floor_z"]
M = Matrix(((c, -s, 0, cx - (c * 8.27 - s * 4.035)), (s, c, 0, cy - (s * 8.27 + c * 4.035)), (0, 0, 1, fz + 0.006), (0, 0, 0, 1)))   # carpet thickness
Mi = M.inverted()
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
def imp(path):
    before = set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]
# stadium + clear box (delete faces whose centroid is inside the box, in field frame)
hx, hy = A["clear_box"]["size_m"][0] / 2, A["clear_box"]["size_m"][1] / 2; z0, z1 = A["clear_box"]["z_from_floor"]
CE = json.load(open(f"{ROOT}/models/field/carpet_placeholder.json"))["extent_field_m"]
CX0, CX1 = CE["x"][0] - 0.5, CE["x"][1] + 0.5; CY0, CY1 = CE["y"][0] - 0.5, CE["y"][1] + 0.5   # clear the whole carpet + 0.5 m
deleted = 0
removed_objs = []
for o in imp(f"{ROOT}/{A['glb']}"):
    if o.type != "MESH": continue
    bb = [Mi @ (o.matrix_world @ Vector(c)) for c in o.bound_box]
    bx0, bx1 = min(q.x for q in bb), max(q.x for q in bb); by0, by1 = min(q.y for q in bb), max(q.y for q in bb)
    if max(bx1 - bx0, by1 - by0) < 15 and bx1 > CX0 - 3 and bx0 < CX1 + 3 and by1 > CY0 - 3 and by0 < CY1 + 3 \
            and min(q.z for q in bb) < 1.0:            # small standalone props on/near the carpet (hoops, goals, benches)
        removed_objs.append(o.name); bpy.data.objects.remove(o); continue
    mw = o.matrix_world; bm = bmesh.new(); bm.from_mesh(o.data)
    kill = []
    for f in bm.faces:
        p = Mi @ (mw @ f.calc_center_median())
        if CX0 <= p.x <= CX1 and CY0 <= p.y <= CY1 and max(z0, 0.004) <= p.z <= z1: kill.append(f)   # keep the arena floor itself; drop decals/boards above carpet top
    # small loose parts (hoops, goals, benches) standing on/near the carpet go too, even if no face centre is inside
    vis, extra, kset = set(), [], set(f.index for f in kill)
    for f0 in bm.faces:
        if f0.index in vis: continue
        isl, stack = [], [f0]; vis.add(f0.index)
        while stack:
            f = stack.pop(); isl.append(f)
            for e in f.edges:
                for g in e.link_faces:
                    if g.index not in vis: vis.add(g.index); stack.append(g)
        if len(isl) > 50000: continue
        pts = [Mi @ (mw @ v.co) for f in isl for v in f.verts]
        xs = [q.x for q in pts]; ys = [q.y for q in pts]; zs = [q.z for q in pts]
        if max(max(xs) - min(xs), max(ys) - min(ys)) < 15 and min(zs) < 1.0 and max(zs) > 0.004 \
                and max(xs) > CX0 - 2 and min(xs) < CX1 + 2 and max(ys) > CY0 - 2 and min(ys) < CY1 + 2:
            extra += [f for f in isl if f.index not in kset]; kset.update(f.index for f in isl)
    kill = kill + extra
    if kill:
        deleted += len(kill); bmesh.ops.delete(bm, geom=kill, context="FACES"); bm.to_mesh(o.data)
    bm.free()
print("clear box deleted faces:", deleted, "removed objects:", removed_objs)
# field (GLB stored in field frame)
for g in ("rebuilt_field.glb", "carpet_placeholder.glb"):
    for o in imp(f"{ROOT}/models/field/{g}"):
        if o.parent is None: o.matrix_world = M @ o.matrix_world
# lights: grid of area lights 12 m over the field, plus dim world
for i in range(3):
    for j in range(2):
        L = bpy.data.lights.new(f"L{i}{j}", "AREA"); L.size = 6; L.energy = 9000
        ob = bpy.data.objects.new(f"L{i}{j}", L); sc.collection.objects.link(ob)
        ob.location = M @ Vector((2.5 + 5.75 * i, 1.5 + 5.0 * j, 12.0))
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
ENGINE = argv[3] if len(argv) > 3 else "CYCLES"
sc.render.engine = ENGINE
if ENGINE == "CYCLES": sc.cycles.device = "CPU"; sc.cycles.samples = samples; sc.cycles.use_denoising = True
else:
    sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "MATERIAL"; sh.show_cavity = True; sh.show_shadows = False
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.view_settings.view_transform = "AgX"
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.clip_end = 1000
def shoot(pos_f, tgt_f, lens, path):
    p, t = M @ Vector(pos_f), M @ Vector(tgt_f)
    cam.location = p; cam.rotation_euler = (t - p).to_track_quat("-Z", "Y").to_euler(); cam.data.lens = lens
    sc.render.filepath = path; bpy.ops.render.render(write_still=True); print("wrote", path)
shoot((8.27, -13.0, 8.0), (8.27, 4.0, 0.0), 24, f"{out}_broadcast.png")
shoot((-14.0, -16.0, 24.0), (8.27, 4.0, 0.0), 18, f"{out}_overview.png")
