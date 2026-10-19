"""blender -b --python render_preview.py -- OUT.png MODE(tex|mask) CLIP FRAME glb1 glb2 ...
Lines avatars up along -Y..+Y, all facing +X; camera on +X looking back at them (front view)."""
import bpy, sys, os, math
from mathutils import Vector
a = sys.argv[sys.argv.index('--') + 1:]
out, mode, clip, frame, glbs = a[0], a[1], a[2], int(a[3]), a[4:]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sp = 0.85
n = len(glbs)
labels = []
seen_acts = set()
for i, g in enumerate(glbs):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=g)
    new = [o for o in bpy.data.objects if o not in before]
    arm = [o for o in new if o.type == 'ARMATURE'][0]
    y = (i - (n - 1) / 2) * sp
    for o in new:
        if o.parent is None: o.location.y += y
    cl = clip.split(',')[i] if ',' in clip else clip
    act = None
    if cl == 'rest':
        if arm.animation_data: arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.scale = (1, 1, 1)
    else:
        newacts = [ac for ac in bpy.data.actions if ac not in seen_acts]
        cands = [ac for ac in newacts if ac.name == cl or ac.name.startswith(cl + '.') or ac.name.startswith(cl + '_')]
        act = (cands or [None])[0]
        if act:
            arm.animation_data_create(); arm.animation_data.action = act
            if hasattr(arm.animation_data, 'action_slot') and act.slots:
                arm.animation_data.action_slot = act.slots[0]
    labels.append(act.name if act else '')
    seen_acts |= set(bpy.data.actions)
    if mode == 'mask':
        for o in new:
            if o.type != 'MESH': continue
            for m in o.data.materials:
                base = m.name.split('.')[0]
                col = {'SHIRT': (1, 0, 1, 1), 'PANTS': (0, 0.9, 1, 1)}.get(base)
                if col:
                    nt = m.node_tree; b = nt.nodes.get('Principled BSDF')
                    for l in list(b.inputs['Base Color'].links): nt.links.remove(l)
                    b.inputs['Base Color'].default_value = col
sc.frame_set(frame)
# floor, light, camera
bpy.ops.mesh.primitive_plane_add(size=40)
fl = bpy.context.object; fm = bpy.data.materials.new('floor'); fm.use_nodes = True
fm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.35, 0.35, 0.37, 1); fl.data.materials.append(fm)
w = bpy.data.worlds.new('w'); w.use_nodes = True; w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.6; sc.world = w
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(50), 0, math.radians(70))); bpy.context.object.data.energy = 3.5
width = n * sp + 0.4
cam_d = 30.0
bpy.ops.object.camera_add(location=(cam_d, 0, 1.0), rotation=(math.radians(90), 0, math.radians(90)))
cam = bpy.context.object; sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = width
sc.render.resolution_x = int(width * 200); sc.render.resolution_y = int(2.15 * 200)
cam.location.z = 2.0 / 2 + 0.03
sc.render.engine = 'CYCLES'; sc.cycles.samples = 24; sc.cycles.device = 'CPU'; sc.render.threads_mode = 'FIXED'; sc.render.threads = 4
sc.cycles.use_denoising = True
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print('ACTIONS', labels)
