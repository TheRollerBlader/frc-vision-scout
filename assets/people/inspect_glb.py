"""python3 inspect_glb.py file.glb -> prints nodes/skins/materials/animations summary (pure json parse)"""
import json, struct, sys
for p in sys.argv[1:]:
    b = open(p, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    g = json.loads(b[20:20 + n])
    acc = g['accessors']
    anims = []
    for a in g.get('animations', []):
        tmax = max(acc[s['input']]['max'][0] for s in a['samplers'])
        anims.append((a['name'], round(tmax, 2), len(a['channels'])))
    print(p, 'nodes', len(g['nodes']), 'skins', [len(s['joints']) for s in g.get('skins', [])], 'materials', [m['name'] for m in g['materials']],
          'images', len(g.get('images', [])), 'meshes', len(g['meshes']))
    print('  anims', anims)
