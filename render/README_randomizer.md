# REBUILT (FRC 2026) frame randomizer

Builds field placement maps from the field CAD and writes one JSON file per "frame" (a frozen
moment of a match): 6 robots (3 red, 3 blue) and randomized fuel. Rendering is a separate step.

## Setup
```
python3 -m venv /workspace/frc/.venv
/workspace/frc/.venv/bin/pip install cadquery-ocp trimesh numpy shapely matplotlib scipy
```

## Pipeline
| step | command | output |
|---|---|---|
| field maps (about 60 s, under 0.5 GB RAM) | `.venv/bin/python tools/build_maps.py` | `field/field_maps.npz`, `field/regions.json` |
| map check image | `.venv/bin/python tools/plot_maps.py` | `field/map_check.png` |
| frames | `.venv/bin/python tools/generate_frame.py --n-frames 5 --seed 1 --plot` | `frames/frame_000001.json` (+ `.png`) ... |
| plot existing frames | `.venv/bin/python tools/plot_frame.py frames/*.json` | `.png` next to each JSON |
| legality check | `.venv/bin/python tools/validate_frames.py frames/*.json` | prints any rule violations |

`build_maps.py` needs `field/field_tris_cad_mm.npz`, `field/field_tris_meta.json` (made by
`tools/build_field_map.py` from the STEP file) and `tools/field_tree.json`.

### generate_frame.py CLI
```
--n-frames N        number of frames (default 1)
--seed S            frame k uses seed S+k (frame_id = frame_<seed>); same seed -> same JSON
--out-dir DIR       default /workspace/frc/frames
--manifest PATH     default /workspace/frc/manifest.json; if missing -> placeholder box robots
--fuel-prob P       per-candidate keep probability (default 0.6)
--max-fuel N        max teleop fuel candidates (default 300)
--config FILE.json  deep-merged override of any tunable (see --dump-default-config)
--plot              also write a top-down PNG per frame
```
Speed: about 0.03 s per frame plus about 1 s per plot. Loading the maps takes about 0.2 s.

## Coordinate frame (used by every output)
Units are meters and radians, Z up. The origin is at the blue alliance wall, at the right-hand
corner as the blue drivers see it. +X points toward the red wall and +Y points to the left.
The field is 16.54 x 8.07 m.
CAD (mm) -> field (m): `x_f=(8270-x_cad)/1000, y_f=(4035-y_cad)/1000, z_f=z_cad/1000`. This is a
180° rotation about Z plus a shift, with no mirroring. As a 4x4 matrix (`cad_to_field`, stored in
the npz, regions.json and every frame):
```
[[-0.001, 0, 0, 8.27], [0, -0.001, 0, 4.035], [0, 0, 0.001, 0], [0, 0, 0, 1]]
```
Blue hub center is about (4.755, 4.035) and red hub center about (11.785, 4.035). The blue
alliance zone is x < 4.028 and the red alliance zone is x > 12.512. The bump/trench/hub barrier
bands are x 4.028–5.222 and 11.318–12.512.

Robot pose: the origin is the center of the frame footprint on the floor, and the front is +X.
`R = Rz(yaw)·Ry(pitch)·Rx(roll)`, the same as Blender rotation_mode 'XYZ' with
`rotation_euler=(roll, pitch, yaw)`.

## field_maps.npz (1 cm grid, `array[ix, iy]`, cell = [ix*0.01, (ix+1)*0.01) x [iy*0.01, ...))
- `floor_z`: height of the drivable surface (carpet 0, bump ramps up to 0.165, tower base about 0.01, depot rim 0.029)
- `clearance`: vertical gap from the floor to the lowest obstacle underside (inf = open,
  0 = solid). Trench arms read 0.565 and tower rungs about 0.88. The hub interior is enclosed, so it is set to 0.
- `obstacle_category`: index into `obstacle_category_names`
  (none, hub, tower, trench, outpost, perimeter, scoring_table, other)
- `zone`: index into `zone_names` (blue_alliance, red_alliance, neutral, barrier_band_blue, barrier_band_red)
- `floor_slope` (rise/run), `reachable`, `res`, `size`, `cad_to_field`

## Frame JSON (schema `frc-frame/v1`)
```
frame_id, seed, phase ("teleop" | "match_start"), coordinate_frame{...},
field{model, model_units:"mm", cad_to_field, maps},
robots[6]: {id ("red_1"..), alliance, bumper_color, team_number (random unique 1..11999),
            manifest_key, model_file | "placeholder", height,
            footprint{length, width, bumper_depth, outline_xy[[x,y]..] (field frame, incl. bumpers)},
            pose{x,y,z,roll,pitch,yaw}, velocity{vx,vy,vz,wz}, zone, on_bump, under_trench, floor_fit_rms}
fuel[]: {id, x,y,z (ball center), state "ground"|"airborne", vx,vy,vz, source | shooter+shot}
fuel_stats{...}, warnings[], config{... full config used, robot_source}
```

## Rules applied
- **Robot legality.** Every 1 cm cell under the bumpered footprint must be inside the field and
  have clearance > height + 0.02 m. Hubs, tower uprights and walls are always blocked; only robots
  shorter than about 0.545 m fit under a trench arm. Bumper outlines must be at least 5 mm apart.
  A plane is fitted to `floor_z` under the footprint, and the pose is rejected if the RMS
  residual is above 0.03 m. z is the plane at the center, raised to the highest support.
- **Where robots go.** The target zone is drawn by weight: own alliance zone 0.35, neutral 0.45,
  opponent 0.15, barrier band 0.05. Within the barrier band, the target is a trench 0.35, a bump
  0.45 or anywhere 0.20. Yaw is random; robots under a trench are squared to the field axes.
- **Velocity.** 25% of robots are stationary. Speed is triangular(0.3, 2.5, 4.5) m/s. Direction is
  independent of yaw with p=0.7 (swerve); otherwise it follows the heading. wz ~ N(0,2) rad/s,
  clipped to ±6. On a bump or under a trench, motion is along ±X. The velocity is resampled if
  the footprint 0.4 s ahead would be illegal.
- **match_start (10% of frames).** Uses the 408 CAD-staged balls (360 in the neutral pile and 48 in
  the depots), jittered by ≤1.2 mm. Robots sit 0–0.8 m behind their starting line, inside their
  alliance zone.
- **teleop.** Candidates = round(max_fuel x U(0.4, 1)), drawn from a mixture: Gaussian clusters
  0.55 (around hub exits on the neutral side, depots, a leftover center pile, the neutral zone and
  alliance zones), piles along the guardrails and alliance walls 0.20, and uniform scatter 0.25.
  Each candidate is kept with --fuel-prob. A ball is allowed only where clearance > 0.16, the
  slope is < 0.05 (so not on bump ramps; the bumps have no flat top) and the ball does not touch
  a wall. Each ball sits on the floor (z = floor + 0.075). Balls are a single layer at least 0.15
  m apart, not under robots, and about 10% are rolling.
- **Airborne fuel.** Each robot shoots with p=0.3, firing 1–6 balls on a ballistic arc. Launch
  height is 0.5 m and each ball leaves from where the robot was when it fired. Hub shots aim at
  the top opening (1.83 m) of the robot's own hub and only happen from the alliance-wall side of
  that hub, within 7.5 m, because the hub net blocks the neutral side. Robots in the neutral zone
  may instead lob the ball into their own alliance zone without crossing the hub. Robots under a
  trench do not shoot.

## manifest.json expected by the randomizer (written by the robot-model step)
```json
{"robots": [
  {"key": "unique_name",
   "model_file": "models/robots/xyz.glb",
   "partial": false,
   "frame_length_m": 0.71, "frame_width_m": 0.66, "height_m": 0.98,
   "frame_outline_m": [[x, y], ...],
   "bumper_depth_m": 0.085}
]}
```
- **Required:** `key`, `model_file`, `frame_length_m`, `frame_width_m` and `height_m` (meters, in
  the robot frame, with the origin at the footprint center on the floor, +X front, Z up), plus
  `partial: true` for incomplete models. Partial models are skipped.
- **Optional:** `frame_outline_m`, the frame perimeter polygon without bumpers (otherwise a
  rectangle is used), and `bumper_depth_m` (default 0.085).
- **Also accepted:** `length`/`width`/`height`, or `bbox_m: {min:[x,y,z], max:[x,y,z]}`; a
  top-level list instead of `{"robots": [...]}`; and `complete: false` or `status: "partial"` as
  the partial flag.
- `height_m` must be the maximum height in the robot's stowed or driving configuration, because
  it decides whether the robot fits under a trench.
