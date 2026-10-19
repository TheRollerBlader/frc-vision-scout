"""blender -b --python shirt_stats.py -- OUT.json glb...  -> per GLB: SHIRT mean colour/luminance, rest bbox"""
import bpy, sys, json, os
import numpy as np
a = sys.argv[sys.argv.index('--') + 1:]
out, glbs = a[0], a[1:]
res = {}
for g in glbs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=g)
    arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
    if arm.animation_data: arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.scale = (1, 1, 1)
    me_o = max((o for o in bpy.data.objects if o.type == 'MESH' and o.parent == arm), key=lambda o: len(o.data.vertices)); me = me_o.data
    co = np.array([v.co[:] for v in me.vertices]) @ np.array(me_o.matrix_world.to_3x3()).T + np.array(me_o.matrix_world.translation)
    d = {'rest_bbox_min': np.round(co.min(0), 3).tolist(), 'rest_bbox_max': np.round(co.max(0), 3).tolist()}
    for mi, m in enumerate(me.materials):
        base = m.name.split('.')[0]
        if base not in ('SHIRT', 'PANTS'): continue
        img = [n.image for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image and n.image.colorspace_settings.name != 'Non-Color'][0]
        W, H = img.size
        px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, img.channels)[..., :3]
        uv = me.uv_layers.active.data
        cols = []
        for p in me.polygons:
            if p.material_index != mi: continue
            u = np.mean([uv[l].uv[:] for l in p.loop_indices], 0)
            cols.append(px[int((u[1] % 1) * H) % H, int((u[0] % 1) * W) % W])
        cols = np.array(cols)
        lin = np.where(cols <= 0.04045, cols / 12.92, ((cols + 0.055) / 1.055) ** 2.4)
        lum = (lin @ np.array([0.2126, 0.7152, 0.0722]))
        d[base.lower() + '_mean_srgb'] = [round(float(x), 3) for x in cols.mean(0)]
        d[base.lower() + '_mean_linear_luminance'] = round(float(lum.mean()), 4)
    # clip sanity after the GLB round trip: lowest foot joint, pelvis XY excursion, toe direction
    chk = {}
    acts = [a for a in bpy.data.actions]
    for act in acts:
        arm.animation_data_create(); arm.animation_data.action = act
        if hasattr(arm.animation_data, 'action_slot') and len(act.slots): arm.animation_data.action_slot = act.slots[0]
        f0, f1 = [int(x) for x in act.frame_range]
        lows, exc, fx = [], [], []
        for f in range(f0, f1 + 1, 5):
            bpy.context.scene.frame_set(f)
            P = lambda n: arm.matrix_world @ arm.pose.bones[n].head
            lows.append(min(P('Bip01 L Toe0').z, P('Bip01 R Toe0').z, P('Bip01 L Foot').z, P('Bip01 R Foot').z))
            exc.append(P('Bip01 Pelvis').xy.length)
            fx.append(((P('Bip01 L Toe0') + P('Bip01 R Toe0')) - (P('Bip01 L Foot') + P('Bip01 R Foot'))).normalized().x)
        chk[act.name.split('_Bip01')[0]] = {'min_foot_joint_z': round(min(lows), 3), 'max_pelvis_xy': round(max(exc), 3), 'mean_toe_dir_x': round(sum(fx) / len(fx), 2)}
    d['clip_check'] = chk
    res[os.path.basename(g)[:-4]] = d
json.dump(res, open(out, 'w'), indent=1)
