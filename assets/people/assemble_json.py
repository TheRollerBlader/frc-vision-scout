#!/usr/bin/env python3
"""Merge work/people/*.info.json + shirt_stats.json -> models/people/people.json"""
import json, sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from build_people import AVATARS, CLIPS, ROOT, WORK, OUT
stats = json.load(open(WORK / 'shirt_stats.json'))
avs = []
for name, (folder, sex, age, note) in AVATARS.items():
    ip = WORK / f'{name}.info.json'
    if not ip.exists(): continue
    info = json.load(open(ip)); st = stats.get(name, {})
    glb = OUT / 'glb' / f'{name}.glb'
    avs.append({
        'name': name, 'file': f'glb/{name}.glb', 'size_mb': round(glb.stat().st_size / 1e6, 2),
        'source': f'Microsoft-Rocketbox Assets/Avatars/{folder}', 'sex': {'m': 'male', 'f': 'female'}[sex],
        'apparent_age': age, 'outfit': note, 'height_m': info['height_m'],
        'materials': info['materials'], 'shirt_material': 'SHIRT', 'pants_material': 'PANTS' if 'PANTS' in info['materials'] else None,
        **{k: v for k, v in st.items()},
        'shirt_uv_bbox': info['shirt_uv_bbox'], 'verts': info['verts'], 'tris': info['tris'], 'bones': info['bones'],
        'clips': [{k: c[k] for k in ('name', 'source', 'duration_s', 'frames', 'fps') if k in c} |
                  ({'in_place': True, 'walk_speed_mps': c['walk_speed_mps']} if c.get('in_place') else {}) for c in info['clips']],
    })
doc = {
    'description': 'Microsoft Rocketbox avatars (MIT) converted for the FRC synthetic dataset. One GLB per avatar: skinned mesh + '
                   'skeleton + all clips as named glTF animations.',
    'license': 'MIT (Microsoft Rocketbox, https://github.com/microsoft/Microsoft-Rocketbox, see LICENSE_ROCKETBOX.md)',
    'units': 'meters',
    'up_axis': 'glTF Y-up storage; Z-up after Blender import (same as robot/stadium GLBs)',
    'origin': 'on the floor (z=0, lowest sole vertex in rest pose), centred between the feet (mean of both ankle and toe joints)',
    'facing': '+X in the Z-up (Blender) scene frame == robot front convention; glTF storage: +X as well (Y-up)',
    'rest_pose': 'T-pose (bind pose). Clips are posed relative to it.',
    'skeleton': '3ds Max Biped naming: root bone "Bip01" (COM, carries root motion), "Bip01 Pelvis", "Bip01 L UpperArm" ... 81 bones incl. facial bones (facial bones are not animated)',
    'fps': 30,
    'clip_notes': 'walk is in-place (forward root motion removed; move the object at walk_speed_mps along +X to match the stride). '
                  'Other clips are stationary. Clips come from adult male (m_) or female (f_) Rocketbox animations matching the avatar sex; '
                  'for children the root/pelvis motion is scaled by leg-length ratio and feet are re-grounded.',
    'materials_doc': {'BODY': 'skin of arms/hands + shoes and other non-shirt/non-pants parts of the body atlas',
                      'SHIRT': 'upper-body clothing (torso, sleeves, collar/hood), split from the body atlas; recolour this',
                      'PANTS': 'trousers/shorts (best effort split)', 'HEAD': 'head/face/hair texture', 'HAIR_ALPHA': 'alpha-blended hair cards/eyelashes (only some avatars)'},
    'clip_names': [c[0] for c in CLIPS],
    'avatars': avs,
}
json.dump(doc, open(OUT / 'people.json', 'w'), indent=1)
print('avatars', len(avs))
