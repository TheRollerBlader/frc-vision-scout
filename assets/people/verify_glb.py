#!/usr/bin/env python3
"""Verify people GLBs (pure json/numpy): skin, materials, clips, rest height, origin at feet, facing +X.
glTF is Y-up: Blender (x, y, z) == glTF (x, z, -y). Facing +X in Blender == +X in glTF."""
import json, struct, sys, glob
import numpy as np
CT = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
def load(p):
    b = open(p, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]; g = json.loads(b[20:20 + n])
    off = 20 + n; bn = struct.unpack('<I', b[off:off + 4])[0]; bin_ = b[off + 8: off + 8 + bn]
    def acc(i):
        a = g['accessors'][i]; bv = g['bufferViews'][a['bufferView']]
        dt = CT[a['componentType']]; nc = NC[a['type']]
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = bv.get('byteStride')
        if stride and stride != nc * np.dtype(dt).itemsize:
            raw = np.frombuffer(bin_, np.uint8, count=stride * a['count'], offset=start).reshape(a['count'], stride)
            return raw[:, :nc * np.dtype(dt).itemsize].copy().view(dt).reshape(a['count'], nc)
        return np.frombuffer(bin_, dt, count=a['count'] * nc, offset=start).reshape(a['count'], nc)
    return g, acc
ok_all = True
rows = []
for p in sys.argv[1:]:
    g, acc = load(p)
    errs = []
    prim = g['meshes'][0]['primitives']
    mats = [g['materials'][pr['material']]['name'] for pr in prim]
    pos = np.concatenate([acc(pr['attributes']['POSITION']) for pr in prim])
    if 'SHIRT' not in mats: errs.append('no SHIRT')
    if not all('JOINTS_0' in pr['attributes'] and 'WEIGHTS_0' in pr['attributes'] for pr in prim): errs.append('not skinned')
    skin = g['skins'][0]
    ibm = acc(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major
    names = [g['nodes'][j]['name'] for j in skin['joints']]
    jw = {nm: np.linalg.inv(m)[:3, 3] for nm, m in zip(names, ibm)}
    # mesh node transform (should be identity / parented to armature root at identity)
    for nd in g['nodes']:
        if 'mesh' in nd and any(k in nd for k in ('translation', 'rotation', 'scale', 'matrix')):
            errs.append('mesh node has transform %s' % {k: nd[k] for k in nd if k in ('translation', 'rotation', 'scale')})
    toe = (jw['Bip01 L Toe0'] + jw['Bip01 R Toe0']) / 2; ank = (jw['Bip01 L Foot'] + jw['Bip01 R Foot']) / 2
    fwd = toe - ank
    facing_ok = fwd[0] > 0.05 and abs(fwd[2]) < 0.03
    feet_c = (jw['Bip01 L Toe0'] + jw['Bip01 R Toe0'] + jw['Bip01 L Foot'] + jw['Bip01 R Foot']) / 4
    if not facing_ok: errs.append(f'facing {fwd}')
    if abs(pos[:, 1].min()) > 0.01: errs.append(f'floor at {pos[:,1].min():.3f}')
    if np.hypot(feet_c[0], feet_c[2]) > 0.01: errs.append(f'feet centre {feet_c}')
    # left foot should be on +Y (Blender) == -Z glTF when facing +X
    if not (jw['Bip01 L Foot'][2] < 0 < jw['Bip01 R Foot'][2]): errs.append('left/right mirrored?')
    clips = {}
    for a in g.get('animations', []):
        clips[a['name']] = round(float(max(acc(s['input']).max() for s in a['samplers'])), 2)
    if len(clips) != 14: errs.append(f'{len(clips)} clips')
    h = float(pos[:, 1].max())
    rows.append((p.split('/')[-1], round(h, 3), mats, len(names), len(clips), errs))
    ok_all &= not errs
    print(f"{p.split('/')[-1]:22s} h={h:.3f} joints={len(names)} mats={mats} clips={len(clips)} fwd=({fwd[0]:+.2f},{-fwd[2]:+.2f}) "
          f"floor={pos[:,1].min():+.3f} feetc=({feet_c[0]:+.3f},{-feet_c[2]:+.3f}) {'OK' if not errs else 'ERR ' + '; '.join(errs)}")
print('ALL OK' if ok_all else 'ERRORS')
