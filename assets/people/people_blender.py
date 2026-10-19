"""Helpers for placing / recolouring Rocketbox people GLBs inside Blender (bpy). Import with:
    import sys; sys.path.append('/workspace/frc/tools/people'); import people_blender as pb
    arm = pb.import_person('/workspace/frc/models/people/glb/male_adult_09.glb', loc=(1, 2, 0), yaw_deg=180, clip='clap', frame=40)
    pb.recolor_shirt(arm, (0.9, 0.05, 0.05))                    # solid role colour, keeps fabric shading
    pb.stripe_shirt(arm, (0.02, 0.02, 0.02), (0.95, 0.95, 0.95), stripe_m=0.05)   # referee stripes
Conventions: person faces +X at yaw 0 (same as robots), origin = floor between the feet, metres, Z-up after import.
"""
import bpy, json, os, math

PEOPLE_JSON = '/workspace/frc/models/people/people.json'
_meta = None

def meta(name):
    global _meta
    if _meta is None:
        _meta = {p['name']: p for p in json.load(open(PEOPLE_JSON))['avatars']}
    return _meta.get(name, {})

def import_person(path, loc=(0, 0, 0), yaw_deg=0.0, clip=None, frame=None, collection=None):
    """Import one person GLB; returns the armature object. clip = clip name from people.json (or None = rest).
    frame = frame inside the clip (1-based, 30 fps); the action is offset with an NLA-free approach:
    the action is assigned and the scene frame is used, so give each person its own copy if you need different frames."""
    before = set(bpy.data.objects); acts_before = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    new_acts = [a for a in bpy.data.actions if a not in acts_before]
    arm = [o for o in new if o.type == 'ARMATURE'][0]
    arm['person_name'] = os.path.splitext(os.path.basename(path))[0]
    arm['person_actions'] = {a.name.split('.')[0].split('_' + arm.name)[0]: a.name for a in new_acts}
    for o in new:
        if collection is not None:
            for c in o.users_collection: c.objects.unlink(o)
            collection.objects.link(o)
    arm.location = loc
    arm.rotation_mode = 'XYZ'; arm.rotation_euler = (0, 0, math.radians(yaw_deg))
    set_clip(arm, clip, frame)
    return arm

def set_clip(arm, clip, frame=None):
    """Pose the armature with a clip. If frame is given the clip is baked to that frame via an NLA strip offset
    so different people can show different frames of the same clip at one scene frame."""
    ad = arm.animation_data or arm.animation_data_create()
    for t in list(ad.nla_tracks): ad.nla_tracks.remove(t)
    ad.action = None
    if not clip:
        for p in arm.pose.bones:
            p.location = (0, 0, 0); p.rotation_quaternion = (1, 0, 0, 0); p.scale = (1, 1, 1)
        return
    acts = dict(arm.get('person_actions', {}))
    act = bpy.data.actions.get(acts.get(clip, clip))
    if act is None:
        raise KeyError(f'clip {clip!r} not found; have {sorted(acts)}')
    if frame is None:
        ad.action = act
        if hasattr(ad, 'action_slot') and len(act.slots): ad.action_slot = act.slots[0]
        return
    tr = ad.nla_tracks.new(); scf = bpy.context.scene.frame_current
    st = tr.strips.new(clip, int(scf - (frame - act.frame_range[0])), act)
    st.extrapolation = 'HOLD'

def _materials(arm, base):
    out = []
    for o in arm.children_recursive:
        if o.type == 'MESH':
            for m in o.data.materials:
                if m and m.name.split('.')[0] == base and m not in out: out.append(m)
    return out

def _own(o_arm, base):
    """make the material single-user for this person so recolouring doesn't affect other people"""
    mats = []
    for o in o_arm.children_recursive:
        if o.type != 'MESH': continue
        for i, m in enumerate(o.data.materials):
            if m and m.name.split('.')[0] == base:
                if m.users > 1:
                    m = m.copy(); o.data.materials[i] = m
                mats.append(m)
    return mats

def recolor_shirt(arm, rgb, gain=None, material='SHIRT'):
    """Base colour = rgb * fabric shading, where shading = luminance(original texture) * gain.
    gain defaults to 0.75 / mean luminance of the original shirt (people.json shirt_mean_linear_luminance)."""
    if gain is None:
        lum = meta(arm.get('person_name', '')).get(material.lower() + '_mean_linear_luminance') or 0.3
        gain = min(0.75 / max(lum, 0.005), 60.0)
    for m in _own(arm, material):
        nt = m.node_tree; b = nt.nodes['Principled BSDF']
        tex = [n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image and n.image.colorspace_settings.name != 'Non-Color'][0]
        bw = nt.nodes.new('ShaderNodeRGBToBW')
        g = nt.nodes.new('ShaderNodeMath'); g.operation = 'MULTIPLY'; g.inputs[1].default_value = gain; g.use_clamp = False
        mn = nt.nodes.new('ShaderNodeMath'); mn.operation = 'MINIMUM'; mn.inputs[1].default_value = 1.3
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
        mix.inputs['Factor'].default_value = 1.0; mix.inputs['A'].default_value = (*rgb, 1.0)
        nt.links.new(tex.outputs['Color'], bw.inputs['Color']); nt.links.new(bw.outputs[0], g.inputs[0])
        nt.links.new(g.outputs[0], mn.inputs[0]); nt.links.new(mn.outputs[0], mix.inputs['B'])
        nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
        m['role_color'] = list(rgb)
    return arm

def stripe_shirt(arm, rgb1, rgb2, stripe_m=0.05, vertical=True, gain=None, material='SHIRT'):
    """Two-colour stripes in the person's rest (undeformed, 'Generated') space: vertical = stripes run up/down
    (bands across the body's left-right axis), like FRC referee shirts. Fabric shading is kept as in recolor_shirt."""
    recolor_shirt(arm, (1, 1, 1), gain, material)
    p = meta(arm.get('person_name', ''))
    lo, hi = p.get('rest_bbox_min', [-0.3, -0.8, 0]), p.get('rest_bbox_max', [0.3, 0.8, 1.9])
    axis = 1 if vertical else 2
    span = hi[axis] - lo[axis]
    for m in _materials(arm, material):
        nt = m.node_tree
        mix = [n for n in nt.nodes if n.type == 'MIX' and n.blend_type == 'MULTIPLY'][-1]
        tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
        mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = span / (2 * stripe_m)
        fr = nt.nodes.new('ShaderNodeMath'); fr.operation = 'FRACT'
        gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.5
        cm = nt.nodes.new('ShaderNodeMix'); cm.data_type = 'RGBA'
        cm.inputs['A'].default_value = (*rgb1, 1); cm.inputs['B'].default_value = (*rgb2, 1)
        nt.links.new(tc.outputs['Generated'], sep.inputs[0]); nt.links.new(sep.outputs['XYZ'[axis]], mul.inputs[0])
        nt.links.new(mul.outputs[0], fr.inputs[0]); nt.links.new(fr.outputs[0], gt.inputs[0])
        nt.links.new(gt.outputs[0], cm.inputs['Factor']); nt.links.new(cm.outputs['Result'], mix.inputs['A'])
    return arm
