"""Rocketbox avatar FBX + Rocketbox animation FBXs -> one GLB (skinned, named clips, SHIRT/PANTS split).

Run:  blender -b --python convert_avatar.py -- JOB.json
JOB.json: {name, fbx, texdir, tex_cache, out_glb, out_info, clips:[{name, fbx, in_place}]}
Conventions (Blender scene, Z-up):  person faces +X, origin on the floor centred between the feet, metres.
glTF storage is the usual Y-up (exporter converts), same as the robot / stadium GLBs.
"""
import bpy, bmesh, sys, os, json, math, collections
import numpy as np
from mathutils import Matrix, Vector

job = json.load(open(sys.argv[sys.argv.index('--') + 1]))
NAME = job['name']
FPS = 30

def log(*a):
    print('[avatar]', NAME, *a, flush=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = FPS; sc.render.fps_base = 1.0

def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, use_anim=True, ignore_leaf_bones=False, automatic_bone_orientation=False)
    return [o for o in bpy.data.objects if o not in before]

def strip_prefix(n):
    # 'Bip01 L Hand' / 'Bip02 L Hand' -> 'L Hand'; 'Bip01' -> ''
    parts = n.split(' ', 1)
    return parts[1] if len(parts) > 1 and parts[0].startswith('Bip') else ('' if n.startswith('Bip') else n)

# ------------------------------------------------------------------ avatar
new = import_fbx(job['fbx'])
A = [o for o in new if o.type == 'ARMATURE'][0]
M = [o for o in new if o.type == 'MESH'][0]
for o in new:
    if o.type == 'EMPTY':
        bpy.data.objects.remove(o, do_unlink=True)
if A.animation_data: A.animation_data_clear()
sc.frame_set(1)
src_prefix = A.name.split(' ')[0]
log('armature', A.name, 'mesh', M.name, 'bones', len(A.data.bones))

Amw = A.matrix_world.copy()
def bone_world_head(b):  # rest, world
    return Amw @ b.head_local

# facing: avatar looks along (toe - ankle); rotate about Z so it looks along +X
bn = {strip_prefix(b.name): b for b in A.data.bones}
fwd = Vector((0, 0, 0))
for s in 'LR':
    fwd += bone_world_head(bn[f'{s} Toe0']) - bone_world_head(bn[f'{s} Foot'])
ang = math.atan2(fwd.y, fwd.x)
rot_z = -ang  # rotate so fwd -> +X
ROT = Matrix.Rotation(rot_z, 4, 'Z')
feet = [ROT @ bone_world_head(bn[f'{s} {p}']) for s in 'LR' for p in ('Foot', 'Toe0')]
cx = sum(p.x for p in feet) / 4; cy = sum(p.y for p in feet) / 4
G = Matrix.Translation((-cx, -cy, 0)) @ ROT
log('source facing angle deg', round(math.degrees(ang), 1), 'rotate', round(math.degrees(rot_z), 1), 'feet centre', round(cx, 3), round(cy, 3))

# --- new clean armature at identity, metres, root bone 'Bip01' replacing the FBX armature object
arm_data = bpy.data.armatures.new(NAME + '_skeleton')
N = bpy.data.objects.new(NAME, arm_data)
sc.collection.objects.link(N)
bpy.context.view_layer.objects.active = N
N.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
def mk(name, mat, length):
    eb = arm_data.edit_bones.new(name)
    eb.head = (0, 0, 0); eb.tail = (0, length, 0)
    loc, rot, _ = mat.decompose()
    eb.matrix = Matrix.LocRotScale(loc, rot, None)
    eb.length = length
    return eb
root = mk('Bip01', G @ Amw, 0.15)
ebs = {}
ascale = Amw.to_scale()[0]
for b in A.data.bones:
    nm = 'Bip01 ' + strip_prefix(b.name)
    ebs[b.name] = mk(nm, G @ Amw @ b.matrix_local, max(b.length * ascale, 0.005))
for b in A.data.bones:
    eb = ebs[b.name]
    eb.parent = ebs[b.parent.name] if b.parent else root
    eb.use_connect = False
    eb.use_deform = True
root.use_deform = False
bpy.ops.object.mode_set(mode='OBJECT')
N.select_set(False)

# --- mesh: bake world transform into vertices, re-target to N
mw = G @ M.matrix_world
M.parent = None
M.matrix_world = Matrix()
M.data.transform(mw)
if mw.determinant() < 0:
    M.data.flip_normals()
M.name = NAME + '_mesh'; M.data.name = NAME + '_mesh'
for vg in M.vertex_groups:
    vg.name = 'Bip01 ' + strip_prefix(vg.name)
for md in list(M.modifiers):
    M.modifiers.remove(md)
md = M.modifiers.new('Armature', 'ARMATURE'); md.object = N
M.parent = N
bpy.data.objects.remove(A, do_unlink=True)
for a in list(bpy.data.actions):
    bpy.data.actions.remove(a)
bpy.context.view_layer.update()

# ------------------------------------------------------------------ materials / shirt split
def srgb_to_lab(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Mx = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ Mx.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

def kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
    return C, np.bincount(lab, minlength=k) / len(X)

texdir = job['texdir']
def tex(kind):
    for f in os.listdir(texdir):
        if f.endswith('_' + kind + '.tga') and 'wrinkle' not in f:
            return os.path.join(texdir, f)
    return None

me = M.data
matnames = [m.name for m in me.materials]
def mat_idx(suffix):
    for i, n in enumerate(matnames):
        if n.lower().endswith(suffix):
            return i
    return None
i_body, i_head, i_opac = mat_idx('_body'), mat_idx('_head'), mat_idx('_opacity')
log('source materials', matnames)

img = bpy.data.images.load(tex('body_color'))
W, H = img.size
px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, img.channels)[..., :3]
bpy.data.images.remove(img)

bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
uvl = bm.loops.layers.uv.active
dl = bm.verts.layers.deform.verify()
gname = {g.index: strip_prefix(g.name) for g in M.vertex_groups}

def region(bone):
    if bone in ('Spine', 'Spine1', 'Spine2', 'Neck', 'Head') or 'Clavicle' in bone or 'UpperArm' in bone or 'Forearm' in bone:
        return 'UPPER'
    if bone == 'Pelvis' or 'Thigh' in bone:
        return 'HIP'
    if 'Calf' in bone:
        return 'CALF'
    if 'Hand' in bone or 'Finger' in bone:
        return 'HAND'
    if 'Foot' in bone or 'Toe' in bone:
        return 'FOOT'
    return 'UPPER'

body_faces = [f for f in bm.faces if f.material_index == i_body]
fcol = {}; freg = {}; fbone = {}
for f in body_faces:
    wsum = collections.Counter()
    for v in f.verts:
        for g, w in v[dl].items():
            wsum[gname[g]] += w
    bone = wsum.most_common(1)[0][0] if wsum else 'Spine'
    fbone[f.index] = bone; freg[f.index] = region(bone)
    uvs = [Vector(l[uvl].uv) for l in f.loops]
    c = sum(uvs, Vector((0, 0))) / len(uvs)
    pts = [c] + [c.lerp(u, 0.6) for u in uvs]
    cols = [px[int((p.y % 1) * H) % H, int((p.x % 1) * W) % W] for p in pts]
    fcol[f.index] = srgb_to_lab(np.median(np.array(cols), 0))

def labs(pred):
    return np.array([fcol[f.index] for f in body_faces if pred(f)])
skin = np.median(labs(lambda f: freg[f.index] == 'HAND'), 0)
SKW = np.array([0.3, 1.0, 1.0])  # skin distance: lightness-tolerant (baked lighting / dark skin), chroma-strict
def dskin(c):
    return float(np.linalg.norm((np.asarray(c) - skin) * SKW))
def refs(X, k, min_frac=0.06, skin_de=12):
    if len(X) == 0: return np.zeros((0, 3))
    C, frac = kmeans(X, k)
    keep = [c for c, fr in zip(C, frac) if fr >= min_frac and dskin(c) > skin_de]
    return np.array(keep) if keep else np.zeros((0, 3))
shirt_refs = refs(labs(lambda f: fbone[f.index] in ('Spine1', 'Spine2')), 4)
if len(shirt_refs) == 0:
    shirt_refs = refs(labs(lambda f: fbone[f.index] in ('Spine1', 'Spine2')), 4, skin_de=0)
pants_refs = refs(labs(lambda f: 'Thigh' in fbone[f.index]), 3)   # thighs are always covered (trousers or shorts)
pants_refs = np.array([c for c in pants_refs if min(np.linalg.norm(c - r) for r in shirt_refs) > 12]) if len(pants_refs) and len(shirt_refs) else pants_refs
log('skin Lab', np.round(skin, 1).tolist(), 'shirt refs', np.round(shirt_refs, 1).tolist(), 'pants refs', np.round(pants_refs, 1).tolist())

def dmin(c, R):
    return min(np.linalg.norm(c - r) for r in R) if len(R) else 1e9

# UV islands (garment panels in the atlas)
parent = list(range(len(bm.faces)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for e in bm.edges:
    if len(e.link_faces) != 2: continue
    f1, f2 = e.link_faces
    if f1.material_index != i_body or f2.material_index != i_body: continue
    def uvs_at(f):
        return {l.vert.index: tuple(round(x, 5) for x in l[uvl].uv) for l in f.loops if l.vert in e.verts}
    if uvs_at(f1) == uvs_at(f2):
        parent[find(f1.index)] = find(f2.index)
islands = collections.defaultdict(list)
for f in body_faces:
    islands[find(f.index)].append(f.index)
isl_of = {}
isl_upper = {}
isl_thigh = {}
for root_i, isl in islands.items():
    up = sum(1 for i in isl if freg[i] == 'UPPER' and fbone[i] != 'Neck')
    lo = sum(1 for i in isl if freg[i] == 'CALF' or 'Thigh' in fbone[i])
    for i in isl:
        isl_of[i] = root_i
    isl_upper[root_i] = up > lo
    isl_thigh[root_i] = any('Thigh' in fbone[i] or fbone[i] == 'Pelvis' for i in isl)
# skin threshold adapted to the skin texture variance (hand faces)
hand_de = np.array([dskin(fcol[f.index]) for f in body_faces if freg[f.index] == 'HAND'])
SKIN_DE = float(np.clip(np.percentile(hand_de, 95) * 1.4, 10, 20))
log('skin dE threshold', round(SKIN_DE, 1))
label = {}
for f in body_faces:
    c = fcol[f.index]; r = freg[f.index]
    ds, dsh, dp = dskin(c), dmin(c, shirt_refs), dmin(c, pants_refs)
    if r == 'UPPER':
        # any non-skin surface on the torso/arms is upper-body clothing (incl. layered under-sleeves, collars)
        label[f.index] = 'SHIRT' if (dsh < ds or ds > SKIN_DE) else 'BODY'
    elif r == 'HIP':
        # skin (bare legs / midriff) by colour; otherwise the garment is decided by the UV panel it belongs to
        if ds < min(dsh, dp) and ds < SKIN_DE:
            label[f.index] = 'BODY'
        else:
            label[f.index] = 'SHIRT' if isl_upper[isl_of[f.index]] else 'PANTS'
    elif r == 'CALF':
        # lower legs: trousers vs bare skin by colour (if the thigh colour equals the shirt colour, e.g. denim-on-denim,
        # the pants refs were dropped and the shirt refs stand in)
        dpe = dp if len(pants_refs) else dsh
        label[f.index] = 'PANTS' if dpe < 0.8 * ds else 'BODY'
    elif r == 'FOOT':
        # trouser hems weighted to the ankle: same UV panel as the thigh and pants-coloured
        label[f.index] = 'PANTS' if (isl_thigh[isl_of[f.index]] and dp < ds and dp < 12) else 'BODY'
    else:
        label[f.index] = 'BODY'

for isl in islands.values():
    cnt = collections.Counter(label[i] for i in isl)
    regs = collections.Counter(freg[i] for i in isl)
    n = len(isl)
    hands_feet = (regs['HAND'] + regs['FOOT']) / n
    if hands_feet > 0.05: continue
    for lab in ('SHIRT', 'PANTS'):
        if cnt[lab] / n >= 0.7:
            for i in isl:
                if freg[i] in ('UPPER', 'HIP') and (lab == 'SHIRT' or dskin(fcol[i]) > SKIN_DE):
                    label[i] = lab
# majority smoothing over edge neighbours (not hands/feet)
for it in range(2):
    newl = dict(label)
    for f in body_faces:
        if freg[f.index] in ('HAND', 'FOOT'): continue
        nb = collections.Counter(label[g.index] for e in f.edges for g in e.link_faces if g.index != f.index and g.material_index == i_body)
        if nb:
            top, k = nb.most_common(1)[0]
            if top != label[f.index] and k >= 2 and k > nb[label[f.index]]:
                newl[f.index] = top
    label = newl
# drop tiny SHIRT / PANTS islands (connected components < 6 faces)
for lab in ('SHIRT', 'PANTS'):
    seen = set()
    for f in body_faces:
        if label[f.index] != lab or f.index in seen: continue
        comp = [f]; seen.add(f.index); st = [f]
        while st:
            g = st.pop()
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen and h.material_index == i_body and label.get(h.index) == lab:
                        seen.add(h.index); comp.append(h); st.append(h)
        anchor = 'HIP' if lab == 'PANTS' else 'UPPER'   # garments must connect to the thighs / torso
        if len(comp) < 6 or not any(freg[g.index] == anchor for g in comp):
            for g in comp: label[g.index] = 'BODY'

cnt = collections.Counter(label.values())
log('face labels', dict(cnt))

# --- build output materials
def load_img(kind, size, fmt):
    src = tex(kind)
    if not src: return None
    os.makedirs(job['tex_cache'], exist_ok=True)
    base = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(job['tex_cache'], f'{base}_{size}.{fmt}')
    if not os.path.exists(dst):
        im = bpy.data.images.load(src)
        im.scale(size, size)
        im.filepath_raw = dst
        im.file_format = 'JPEG' if fmt == 'jpg' else 'PNG'
        if fmt == 'jpg':
            sc.render.image_settings.quality = 90
        im.save()
        bpy.data.images.remove(im)
    im = bpy.data.images.load(dst)
    im.name = os.path.basename(dst)
    if 'normal' in kind or 'specular' in kind:
        im.colorspace_settings.name = 'Non-Color'
    return im

TEX = job.get('tex_size', 1024)
I = {k: load_img(k, TEX, 'png' if k == 'opacity_color' else 'jpg') for k in ('body_color', 'body_normal', 'head_color', 'head_normal', 'opacity_color')}

def make_mat(name, color_img, normal_img, rough, alpha=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = 0.0
    if color_img:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = color_img; t.location = (-500, 300)
        nt.links.new(t.outputs['Color'], bsdf.inputs['Base Color'])
        if alpha:
            nt.links.new(t.outputs['Alpha'], bsdf.inputs['Alpha'])
            m.blend_method = 'HASHED' if hasattr(m, 'blend_method') else None
            try: m.surface_render_method = 'DITHERED'
            except Exception: pass
    if normal_img:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = normal_img; t.location = (-500, -100)
        nmap = nt.nodes.new('ShaderNodeNormalMap'); nmap.location = (-200, -100)
        nt.links.new(t.outputs['Color'], nmap.inputs['Color'])
        nt.links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])
    return m

out_mats = {
    'BODY': make_mat('BODY', I['body_color'], I['body_normal'], 0.75),
    'SHIRT': make_mat('SHIRT', I['body_color'], I['body_normal'], 0.8),
    'PANTS': make_mat('PANTS', I['body_color'], I['body_normal'], 0.8),
    'HEAD': make_mat('HEAD', I['head_color'], I['head_normal'], 0.6),
}
if i_opac is not None:
    out_mats['HAIR_ALPHA'] = make_mat('HAIR_ALPHA', I['opacity_color'] or I['head_color'], None, 0.6, alpha=True)
order = [k for k in ('BODY', 'SHIRT', 'PANTS', 'HEAD', 'HAIR_ALPHA') if k in out_mats and (k not in ('SHIRT', 'PANTS') or cnt.get(k))]
for f in bm.faces:
    if f.material_index == i_body:
        key = label[f.index]
    elif f.material_index == i_head:
        key = 'HEAD'
    elif f.material_index == i_opac:
        key = 'HAIR_ALPHA'
    else:
        key = 'BODY'
    f.material_index = order.index(key)
bm.to_mesh(me); bm.free()
old_mats = list(me.materials)
for i, k in enumerate(order):
    if i < len(me.materials): me.materials[i] = out_mats[k]
    else: me.materials.append(out_mats[k])
while len(me.materials) > len(order):
    me.materials.pop(index=len(me.materials) - 1)
for m in old_mats:
    if m and m.users == 0: bpy.data.materials.remove(m)
# shirt UV bbox (for team-number / stripe textures)
uvs = np.array([me.uv_layers.active.data[li].uv[:] for p in me.polygons if me.materials[p.material_index].name == 'SHIRT' for li in p.loop_indices]) if 'SHIRT' in order else np.zeros((0, 2))
shirt_uv_bbox = [round(float(x), 4) for x in (uvs.min(0).tolist() + uvs.max(0).tolist())] if len(uvs) else None
me.uv_layers.active.name = 'UVMap'

# ------------------------------------------------------------------ rest stats
co = np.array([v.co[:] for v in me.vertices])
height = float(co[:, 2].max())
log('rest height', round(height, 3), 'bbox', np.round(co.min(0), 3).tolist(), np.round(co.max(0), 3).tolist())
NB = {strip_prefix(b.name): b for b in N.data.bones}
def chain_len(bones):
    return sum(b.length for b in bones)
def leg_len_rest_N():
    # hip joint -> ankle, using head positions
    return (NB['L Thigh'].head_local - NB['L Calf'].head_local).length + (NB['L Calf'].head_local - NB['L Foot'].head_local).length
def foot_points_N():
    return [N.matrix_world @ N.pose.bones['Bip01 ' + n].head for n in ('L Foot', 'R Foot', 'L Toe0', 'R Toe0')]
rest_foot_min = min(p.z for p in foot_points_N())

# ------------------------------------------------------------------ animations
def add_constraints(T, tmap):
    for pb in N.pose.bones:
        for c in list(pb.constraints): pb.constraints.remove(c)
        if pb.name == 'Bip01':
            c = pb.constraints.new('COPY_LOCATION'); c.target = T
            c = pb.constraints.new('COPY_ROTATION'); c.target = T
        else:
            key = strip_prefix(pb.name)
            if key in tmap:
                c = pb.constraints.new('COPY_ROTATION'); c.target = T; c.subtarget = tmap[key]

clips_info = []
baked_names = set()
FACIAL = {strip_prefix(b.name) for b in NB['Head'].children_recursive}  # eyes, jaw, lips, brows ...
for clip in job['clips']:
    objs = import_fbx(clip['fbx'])
    T = [o for o in objs if o.type == 'ARMATURE'][0]
    act = T.animation_data.action
    f0, f1 = [int(round(x)) for x in act.frame_range]
    tmap = {strip_prefix(b.name): b.name for b in T.data.bones}
    tsc = T.matrix_world.to_scale()[0]
    tb = {strip_prefix(b.name): b for b in T.data.bones}
    leg_T = ((tb['L Thigh'].head_local - tb['L Calf'].head_local).length + (tb['L Calf'].head_local - tb['L Foot'].head_local).length) * tsc
    r = leg_len_rest_N() / leg_T
    speed = None
    if clip.get('in_place'):
        fcs = [fc for fc in act.fcurves if fc.data_path == 'location']
        disp = Vector((0, 0, 0))
        for fc in fcs:
            if fc.array_index > 1: continue
            k0, k1 = fc.keyframe_points[0].co, fc.keyframe_points[-1].co
            slope = (k1.y - k0.y) / (k1.x - k0.x)
            disp[fc.array_index] = k1.y - k0.y
            for kp in fc.keyframe_points:
                d = slope * (kp.co.x - k0.x)
                kp.co.y -= d; kp.handle_left.y -= d; kp.handle_right.y -= d
        speed = disp.length * r / ((f1 - f0) / FPS)
    for o in objs:
        if o is not T: bpy.data.objects.remove(o, do_unlink=True)
    E = bpy.data.objects.new('E', None); sc.collection.objects.link(E)
    E.matrix_world = G @ Matrix.Scale(r, 4)
    T.parent = E; T.matrix_parent_inverse = Matrix()
    add_constraints(T, tmap)
    sc.frame_start, sc.frame_end = f0, f1
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    N.select_set(True); bpy.context.view_layer.objects.active = N
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.nla.bake(frame_start=f0, frame_end=f1, step=1, only_selected=False, visual_keying=True,
                     clear_constraints=False, use_current_action=False, bake_types={'POSE'})
    bpy.ops.object.mode_set(mode='OBJECT')
    baked = N.animation_data.action
    baked.name = clip['name']; baked.use_fake_user = True
    # keep: root (Bip01) location+rotation, body/finger rotations. drop: child-bone location/scale (constant
    # by construction) and facial bones (face stays at rest pose) -> much smaller GLBs
    for fc in list(baked.fcurves):
        bname = fc.data_path.split('"')[1] if '"' in fc.data_path else ''
        prop = fc.data_path.rsplit('.', 1)[-1]
        keep = (bname == 'Bip01' and prop in ('location', 'rotation_quaternion')) or \
               (prop == 'rotation_quaternion' and strip_prefix(bname) not in FACIAL)
        if not keep:
            baked.fcurves.remove(fc)
    for pb in N.pose.bones:
        if pb.name != 'Bip01':
            pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
            if strip_prefix(pb.name) in FACIAL: pb.rotation_quaternion = (1, 0, 0, 0)
    for pb in N.pose.bones:
        for c in list(pb.constraints): pb.constraints.remove(c)
    tact = T.animation_data.action if T.animation_data else None
    bpy.data.objects.remove(T, do_unlink=True); bpy.data.objects.remove(E, do_unlink=True)
    baked_names.add(baked.name)
    for a in list(bpy.data.actions):
        if a.name not in baked_names: bpy.data.actions.remove(a)
    # ground contact: lowest foot/toe joint over the clip should match the rest pose
    lows = []
    for f in range(f0, f1 + 1):
        sc.frame_set(f); lows.append(min(p.z for p in foot_points_N()))
    dz = rest_foot_min - min(lows)
    if abs(dz) > 0.003:
        rb = N.data.bones['Bip01']
        dloc = rb.matrix_local.to_3x3().inverted() @ Vector((0, 0, dz))
        for fc in baked.fcurves:
            if fc.data_path == 'pose.bones["Bip01"].location':
                for kp in fc.keyframe_points:
                    kp.co.y += dloc[fc.array_index]; kp.handle_left.y += dloc[fc.array_index]; kp.handle_right.y += dloc[fc.array_index]
    # drift check of the root over the clip (world XY)
    sc.frame_set(f0); p0 = N.matrix_world @ N.pose.bones['Bip01'].head
    sc.frame_set(f1); p1 = N.matrix_world @ N.pose.bones['Bip01'].head
    info = {'name': clip['name'], 'source': os.path.basename(clip['fbx']), 'frames': f1 - f0 + 1, 'fps': FPS,
            'duration_s': round((f1 - f0) / FPS, 3), 'leg_scale_vs_source': round(r, 3), 'ground_fix_m': round(dz, 4),
            'root_drift_m': round((p1 - p0).xy.length, 3)}
    if speed is not None:
        info['in_place'] = True; info['walk_speed_mps'] = round(speed, 3)
    clips_info.append(info)
    log('clip', json.dumps(info))

if N.animation_data: N.animation_data.action = None
for pb in N.pose.bones:
    pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.rotation_euler = (0, 0, 0); pb.scale = (1, 1, 1)
sc.frame_set(1)

# ------------------------------------------------------------------ export
os.makedirs(os.path.dirname(job['out_glb']), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=job['out_glb'], export_format='GLB', export_yup=True, export_apply=False,
                          export_animations=True, export_animation_mode='ACTIONS', export_force_sampling=True,
                          export_anim_slide_to_zero=True, export_reset_pose_bones=True, export_def_bones=False,
                          export_optimize_animation_size=True, export_skins=True, export_morph=False,
                          export_image_format='AUTO', export_materials='EXPORT', export_cameras=False, export_lights=False,
                          export_extras=False, use_selection=False)
info = {'name': NAME, 'height_m': round(height, 3), 'materials': order, 'shirt_faces': cnt.get('SHIRT', 0),
        'pants_faces': cnt.get('PANTS', 0), 'shirt_uv_bbox': shirt_uv_bbox, 'bones': len(N.data.bones),
        'verts': len(me.vertices), 'tris': sum(len(p.vertices) - 2 for p in me.polygons),
        'source_facing_deg': round(math.degrees(ang), 1), 'clips': clips_info,
        'shirt_ref_lab': np.round(shirt_refs, 1).tolist(), 'skin_ref_lab': np.round(skin, 1).tolist()}
json.dump(info, open(job['out_info'], 'w'), indent=1)
log('wrote', job['out_glb'], os.path.getsize(job['out_glb']) // 1024, 'KB')
