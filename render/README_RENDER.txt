FRC REBUILT synthetic render pipeline (render/)
==============================================
setup:   bash render/setup.sh [bundle]          (Blender 4.5 + venv)
test:    bash render/render.sh --test           (50 frames -> out_test/, previews with boxes in out_test/preview/)
full:    bash render/render.sh --full 10000     (resumable: re-run the same command)

render.sh options
  --seed S            first frame seed (default 100000)
  --samples K         Cycles samples (default 64)
  --res WxH           render size (default 1920x1080)
  --chunk C           frames per Blender process (default 25). Chunks are grouped by stadium, so the heavy
                      per-stadium prep (field-area clearing, floor split, seat detection, crowd + stand-light setup)
                      runs once per chunk; seat spots are additionally cached in cache/stadium/<name>_<hash>.npz.
  --gpus G            number of GPUs to use (default: nvidia-smi count, min 1)
  --per-gpu N         Blender processes per GPU (default 1). G*N workers pull chunks from one shared queue
                      (atomic mkdir claims); worker w gets CUDA_VISIBLE_DEVICES = w % G. N=2 usually helps keep a big
                      GPU busy during the CPU-bound scene setup of the other process. Watch host RAM (~2-4 GB/process).
  --shard i/n         render every n-th chunk starting at i (several machines)
  --budget USD --price USD_PER_HR
                      stop before the next chunk once elapsed_hours*price >= budget, then `runpodctl stop pod`
                      if $RUNPOD_POD_ID is set. Finished frames are kept; re-run to continue.
Resuming: frames with labels/<id>.json are skipped; chunks whose frames are all done are not started at all.

Per frame (cheap only): material colour jitter, floor tint, light rig + stand lights + auto exposure, robots/fuel/people
placement (people GLBs are imported once per process and copied), camera, crowd attributes (occupancy 30-90 %, proxy,
yaw, height, shirt/pants colours, fans that could cover the field volume from this camera are hidden).
Per stadium (once per Blender process): duplicate back-face twins removed (they caused shadow acne = black floor /
stand fronts), field-area clear-out, arena floor material, floor patch + big floor slab 0.4 m below the floor under the
whole bowl (no black void anywhere), seat detection (cached), crowd instancer, stand lights + floor-ring fill lights
(positions found by raycasts: below overhangs, never inside geometry).
Camera (render/blender_render.py camera_pose): ALWAYS head-on like the broadcast: on the frame's camera_side, centred
on midfield (x = 8.27 +-0.5 m), looking straight across (yaw +-2.5 deg), 8.5-20.5 m behind the near guardrail,
11-25 deg elevation (~3-9 m up), far guardrail at 84-98 % of the frame width, near guardrail at/just below the bottom
edge, roll +-1 deg, tilt jitter; robots kept in frame (zoom out / re-tilt only) in 97 % of frames; DOF (60 %),
operator pan/tilt motion blur (35 %). Fast projection-only check:
  blender -b -P render/blender_render.py -- --frames f*.json --out /tmp/x --res 1920x1080 --camera-check stats.json
Crowd: render/stadium_prep.py + render/assets/crowd/crowd_proxies.blend (rebuild with
  blender -b -P render/crowd/build_crowd_proxies.py). Flags for A/B checks: --no-crowd, --no-people.
Textures + licences: render/assets/TEXTURES.md (all CC0).
