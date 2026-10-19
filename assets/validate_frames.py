#!/usr/bin/env python3
"""Independent legality check of frc-frame/v1 JSONs against field_maps.npz.
Usage: .venv/bin/python tools/validate_frames.py frames/*.json"""
import json, sys, collections, numpy as np, shapely
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
M = np.load('/workspace/frc/field/field_maps.npz'); C = M['clearance']; FZ = M['floor_z']; S = M['floor_slope']; RES = float(M['res'])
L, W = [float(v) for v in M['size']]
bad = collections.Counter(); st = collections.Counter(); n = 0
for p in sys.argv[1:]:
    f = json.load(open(p)); n += 1; st[f['phase']] += 1
    polys = []
    assert len(f['robots']) == 6 and sum(r['alliance'] == 'red' for r in f['robots']) == 3
    assert len({r['team_number'] for r in f['robots']}) == 6
    for r in f['robots']:
        poly = Polygon(r['footprint']['outline_xy']); polys.append(poly)
        x0, y0, x1, y1 = poly.bounds
        if x0 < 0 or y0 < 0 or x1 > L or y1 > W: bad['robot_outside'] += 1; continue
        IX, IY = np.meshgrid(np.arange(int(x0 / RES), int(x1 / RES) + 1), np.arange(int(y0 / RES), int(y1 / RES) + 1), indexing='ij')
        IX = np.minimum(IX, C.shape[0] - 1); IY = np.minimum(IY, C.shape[1] - 1)
        m = shapely.contains_xy(poly, (IX + .5) * RES, (IY + .5) * RES)
        if not np.all(C[IX[m], IY[m]] > r['height'] + 0.02): bad['robot_blocked'] += 1
        st['zone:' + r['zone']] += 1; st['on_bump'] += r['on_bump']; st['under_trench'] += r['under_trench']
        st['moving'] += (abs(r['velocity']['vx']) + abs(r['velocity']['vy']) > 0)
        if r['pose']['z'] < FZ[int(r['pose']['x'] / RES), int(r['pose']['y'] / RES)] - 0.03: bad['robot_sunk'] += 1
    for i in range(6):
        for j in range(i + 1, 6):
            if polys[i].intersection(polys[j]).area > 1e-6: bad['robot_overlap'] += 1
    g = np.array([[b['x'], b['y'], b['z']] for b in f['fuel'] if b['state'] == 'ground'])
    a = [b for b in f['fuel'] if b['state'] == 'airborne']
    st['ground'] += len(g); st['airborne'] += len(a)
    if len(g):
        ix = (g[:, 0] / RES).astype(int); iy = (g[:, 1] / RES).astype(int)
        bad['fuel_low_clear'] += int((C[ix, iy] <= 0.16).sum())
        bad['fuel_on_slope'] += int((S[ix, iy] >= 0.05).sum())
        bad['fuel_z_wrong'] += int((np.abs(g[:, 2] - FZ[ix, iy] - 0.075) > 0.01).sum())
        if len(g) > 1:
            d, _ = cKDTree(g[:, :2]).query(g[:, :2], k=2); bad['fuel_overlap'] += int((d[:, 1] < 0.15 - 1e-4).sum())
        for poly in polys: bad['fuel_in_robot'] += int(shapely.contains_xy(poly, g[:, 0], g[:, 1]).sum())
    for b in a:
        if not (0 < b['x'] < L and 0 < b['y'] < W): bad['air_outside'] += 1
    st['warnings'] += len(f['warnings'])
print('frames', n); print('violations', dict(bad) or 'none'); print('stats', dict(st))
