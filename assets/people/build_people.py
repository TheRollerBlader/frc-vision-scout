#!/usr/bin/env python3
"""Build /workspace/frc/models/people/glb/*.glb (one avatar at a time, sequential) + people.json.
Usage: python3 build_people.py [names...]   (default: all in AVATARS)"""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path('/workspace/frc')
RB = ROOT / 'src/rocketbox/repo/Assets'
OUT = ROOT / 'models/people'
WORK = ROOT / 'work/people'
BLENDER = os.environ.get('BLENDER', '/workspace/tools/blender-4.5.9-linux-x64/blender')
# name -> (rocketbox folder, sex, apparent age group, note)
AVATARS = {
    'male_adult_06':   ('Adults/Male_Adult_06', 'm', 'adult', 'red long-sleeve henley, dark jeans'),
    'male_adult_08':   ('Adults/Male_Adult_08', 'm', 'adult', 'light-blue button shirt, dark trousers'),
    'male_adult_09':   ('Adults/Male_Adult_09', 'm', 'adult', 'navy t-shirt, jeans'),
    'male_adult_12':   ('Adults/Male_Adult_12', 'm', 'adult', 'darker skin, denim jacket, jeans'),
    'male_adult_16':   ('Adults/Male_Adult_16', 'm', 'adult', 'blue t-shirt, jeans'),
    'male_adult_17':   ('Adults/Male_Adult_17', 'm', 'adult', 'brown skin, blue/white track jacket, jeans'),
    'female_adult_01': ('Adults/Female_Adult_01', 'f', 'adult', 'pink blouse, jeans'),
    'female_adult_03': ('Adults/Female_Adult_03', 'f', 'adult', 'purple top, black trousers'),
    'female_adult_08': ('Adults/Female_Adult_08', 'f', 'adult', 'grey t-shirt, jeans'),
    'female_adult_12': ('Adults/Female_Adult_12', 'f', 'adult', 'black hoodie, denim shorts'),
    'female_adult_17': ('Adults/Female_Adult_17', 'f', 'adult', 'green t-shirt, light jeans'),
    'female_child_01': ('Children/Female_Child_01', 'f', 'child/teen', 'pink t-shirt, jeans'),
    'male_child_01':   ('Children/Male_Child_01', 'm', 'child/teen', 'blue sweatshirt, jeans'),
    'male_child_02':   ('Children/Male_Child_02', 'm', 'child/teen', 'dark skin, orange #38 jersey, shorts'),
}
S = 'Animations/all_animations_max_motextr_static'
XY = 'Animations/all_animations_max_motextr_xy'
CLIPS = [  # clip name, source (without sex prefix), folder, in_place
    ('idle_neutral_01', 'idle_neutral_01', S, False),
    ('idle_neutral_02', 'idle_neutral_02', S, False),
    ('idle_look_around', 'idle_look_around_01', S, False),
    ('idle_waiting', 'idle_waiting_01', S, False),
    ('walk', 'walk_neutral_01', XY, True),
    ('talk_neutral', 'gestic_talk_neutral_01', S, False),
    ('talk_excited', 'gestic_talk_excited_01', S, False),
    ('clap', 'claphands_01', S, False),
    ('cheer_01', 'cheer_01', S, False),
    ('cheer_03', 'cheer_03', S, False),
    ('point_left', 'gestic_presentation_left_01', S, False),
    ('point_right', 'gestic_presentation_right_01', S, False),
    ('emcee_moderate', 'moderate_01', S, False),
    ('wave', 'wave_01', S, False),
]

def main():
    names = sys.argv[1:] or list(AVATARS)
    WORK.mkdir(parents=True, exist_ok=True); (OUT / 'glb').mkdir(parents=True, exist_ok=True)
    for name in names:
        folder, sex, age, note = AVATARS[name]
        d = RB / 'Avatars' / folder
        fbx = d / 'Export' / (d.name + '.fbx')
        job = {'name': name, 'fbx': str(fbx), 'texdir': str(d / 'Textures'), 'tex_cache': str(WORK / 'tex'),
               'out_glb': str(OUT / 'glb' / f'{name}.glb'), 'out_info': str(WORK / f'{name}.info.json'),
               'clips': [{'name': c, 'fbx': str(RB / fo / f'{sex}_{src}.max.fbx'), 'in_place': ip} for c, src, fo, ip in CLIPS]}
        if os.environ.get('PEOPLE_CLIPS'):
            keep = os.environ['PEOPLE_CLIPS'].split(',')
            job['clips'] = [c for c in job['clips'] if c['name'] in keep]
        jp = WORK / f'{name}.job.json'; jp.write_text(json.dumps(job, indent=1))
        t = time.time()
        with open(WORK / f'{name}.log', 'w') as lf:
            rc = subprocess.run([BLENDER, '-b', '--factory-startup', '-noaudio', '--python', str(ROOT / 'tools/people/convert_avatar.py'), '--', str(jp)],
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        print(name, 'rc', rc, f'{time.time()-t:.0f}s', flush=True)

if __name__ == '__main__':
    main()
