"""blender -b --python render_sheet.py -- OUT.png
Row 1: every avatar, original textures, clip idle_neutral_01.  Row 2: same avatars recoloured by role with
people_blender.py (referee stripes, red/blue drive team, field staff, emcee) in assorted clips.
All face +X; camera on +X (front view)."""
import bpy, sys, math, json
sys.path.insert(0, '/workspace/frc/tools/people')
import people_blender as pb
args = sys.argv[sys.argv.index('--') + 1:]
out = args[0]; ROWS = [int(args[1])] if len(args) > 1 else [0, 1]
doc = json.load(open(pb.PEOPLE_JSON))
avs = doc['avatars']
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene; sc.render.fps = 30
sc.frame_set(60)
n = len(avs); sp = 0.85
roles = [('ref', None, 'point_right'), ('red', (0.75, 0.03, 0.03), 'cheer_01'), ('blue', (0.02, 0.12, 0.75), 'clap'),
         ('staff', (0.95, 0.75, 0.05), 'talk_neutral'), ('emcee', (0.1, 0.1, 0.1), 'emcee_moderate'),
         ('red', (0.75, 0.03, 0.03), 'walk'), ('blue', (0.02, 0.12, 0.75), 'idle_look_around'), ('staff', (0.1, 0.6, 0.15), 'wave')]
for row in ROWS:
    for i, a in enumerate(avs):
        y = (i - (n - 1) / 2) * sp
        z_off = 0
        x = 0
        if row == 0:
            arm = pb.import_person('/workspace/frc/models/people/' + a['file'], loc=(x, y, 0), clip='idle_neutral_01', frame=40)
        else:
            role, col, clip = roles[i % len(roles)]
            arm = pb.import_person('/workspace/frc/models/people/' + a['file'], loc=(x, y, 0), clip=clip, frame=25 if clip == 'walk' else 70)
            if role == 'ref':
                pb.stripe_shirt(arm, (0.02, 0.02, 0.02), (0.9, 0.9, 0.9), stripe_m=0.045)
            else:
                pb.recolor_shirt(arm, col)
bpy.ops.mesh.primitive_plane_add(size=60)
fl = bpy.context.object; fm = bpy.data.materials.new('floor'); fm.use_nodes = True
fm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.3, 0.3, 0.32, 1); fl.data.materials.append(fm)
w = bpy.data.worlds.new('w'); w.use_nodes = True; w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.6; sc.world = w
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(45), 0, math.radians(60))); bpy.context.object.data.energy = 3.5
width = n * sp + 0.4
# slightly elevated camera looking down so both rows are visible
bpy.ops.object.camera_add(location=(30, 0, 1.03), rotation=(math.radians(90), 0, math.radians(90)))
cam = bpy.context.object; sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = width
sc.render.resolution_x = int(width * 170); sc.render.resolution_y = int(2.15 * 170)
sc.render.engine = 'CYCLES'; sc.cycles.samples = 24; sc.cycles.device = 'CPU'; sc.render.threads_mode = 'FIXED'; sc.render.threads = 4
sc.cycles.use_denoising = True
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
