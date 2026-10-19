#!/usr/bin/env bash
# Render the synthetic FRC dataset.
#   render.sh --test                 50 frames -> out_test/ (with preview/ images that have the boxes drawn)
#   render.sh --full [N]             N frames (default 10000) -> out_full/   (resumable: re-run the same command)
# options: --seed S (first seed, default 100000)  --samples K (default 64)  --res WxH (default 1920x1080)
#          --chunk C (frames per Blender run, default 25)  --gpus G (parallel Blender processes, default = #GPUs)
#          --shard i/n (render every n-th chunk starting at i, for several machines)
#          --per-gpu N (Blender processes per GPU, default 1; all workers pull chunks from one queue)
#          --budget USD --price USD_PER_HR   stop rendering once this run has cost ~USD (then stops the Runpod pod if possible)
# stadium prep (clearing, seat detection) runs once per Blender process; seat spots are cached in cache/stadium/.
set -euo pipefail
ROOT=/workspace/frc; BL=/workspace/tools/blender-4.5.9-linux-x64/blender; PY=$ROOT/.venv/bin/python
PER_GPU=1; BUDGET=""; PRICE=""; MODE=""; N=10000; SEED=100000; SAMPLES=64; RES=1920x1080; CHUNK=25; SHARD="0/1"
GPUS=$(nvidia-smi -L 2>/dev/null | grep -c GPU || true); [ "${GPUS:-0}" -ge 1 ] || GPUS=1
while [ $# -gt 0 ]; do case "$1" in
  --test) MODE=test; N=50; shift;; --full) MODE=full; shift; [[ "${1:-}" =~ ^[0-9]+$ ]] && { N=$1; shift; };;
  --seed) SEED=$2; shift 2;; --samples) SAMPLES=$2; shift 2;; --res) RES=$2; shift 2;; --chunk) CHUNK=$2; shift 2;;
  --gpus) GPUS=$2; shift 2;; --per-gpu) PER_GPU=$2; shift 2;; -h|--help) sed -n 2,11p "$0"; exit 0;; --budget) BUDGET=$2; shift 2;; --price) PRICE=$2; shift 2;; --shard) SHARD=$2; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done
[ -n "$MODE" ] || { sed -n 2,11p "$0"; exit 1; }
[[ "$PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "--per-gpu needs a positive integer"; exit 1; }
OUT=$ROOT/out_$MODE; mkdir -p "$OUT/frames" "$OUT/chunks" "$OUT/logs"
[ "$MODE" = test ] && SEED=900000
cd "$ROOT"
# 1) scene descriptions (cheap, deterministic per seed)
have=$(ls "$OUT/frames" | grep -c '^frame_.*json$' || true)
if [ "$have" -lt "$N" ]; then $PY tools/generate_frame.py --n-frames "$N" --seed "$SEED" --out-dir "$OUT/frames" 2>"$OUT/logs/generate.log" >/dev/null; fi
# 2) chunks grouped by stadium (stadium loads once per chunk)
$PY - "$OUT" "$CHUNK" <<'PY'
import json, glob, sys, os
out, chunk = sys.argv[1], int(sys.argv[2])
fr = sorted(glob.glob(f"{out}/frames/frame_*.json"))
by = {}
for f in fr: by.setdefault(json.load(open(f))["venue"]["stadium"], []).append(f)
k = 0
for st in sorted(by):
    L = by[st]
    for i in range(0, len(L), chunk):
        open(f"{out}/chunks/chunk_{k:05d}.txt", "w").write("\n".join(L[i:i + chunk])); k += 1
print(f"{len(fr)} frames, {k} chunks, stadiums: " + ", ".join(f"{s}={len(v)}" for s, v in sorted(by.items())))
PY
# 3) render chunks: GPUS x PER_GPU worker processes (worker w pinned to GPU w % GPUS) pull chunks from one queue
#    (atomic mkdir claims); finished frames are skipped on re-run, fully finished chunks are not even started
SI=${SHARD%/*}; SN=${SHARD#*/}
mapfile -t CH < <(ls "$OUT/chunks"/chunk_*.txt | awk -v i="$SI" -v n="$SN" '(NR-1)%n==i')
CLAIMS=$(mktemp -d "$OUT/.claims.XXXXXX"); trap 'rm -rf "$CLAIMS"' EXIT
chunk_done() { local f; for f in $(cat "$1"); do [ -f "$OUT/labels/$(basename "$f" .json).json" ] || return 1; done; return 0; }
run_chunk() { local c=$1 g=$2; local name=$(basename "$c" .txt)
  CUDA_VISIBLE_DEVICES=$g "$BL" -b -P "$ROOT/render/blender_render.py" -- --frames $(cat "$c") --out "$OUT" \
     --res "$RES" --samples "$SAMPLES" > "$OUT/logs/$name.log" 2>&1 || echo "chunk $name failed (see logs/$name.log)"
  grep '\[render\] frame_' "$OUT/logs/$name.log" | tail -n +1 | sed "s/^/[gpu$g] /" || true; }
T0=$(date +%s)
over_budget() { [ -n "$BUDGET" ] && [ -n "$PRICE" ] || return 1
  awk -v t="$(( $(date +%s) - T0 ))" -v p="$PRICE" -v b="$BUDGET" 'BEGIN{exit !((t/3600.0)*p >= b*0.97)}'; }
worker() { local g=$1 c name
  for c in "${CH[@]}"; do
    name=$(basename "$c" .txt)
    mkdir "$CLAIMS/$name" 2>/dev/null || continue          # another worker has it
    chunk_done "$c" && continue
    if over_budget; then echo "BUDGET: ~\$$BUDGET reached after $(( ($(date +%s)-T0)/60 )) min - stopping (finished frames are kept; re-run to continue)"; touch "$CLAIMS/.stopped"; return; fi
    run_chunk "$c" "$g"
  done; }
NW=$(( GPUS * PER_GPU )); echo "render: ${#CH[@]} chunks, $GPUS GPU(s) x $PER_GPU process(es) per GPU = $NW workers"
for ((w=0; w<NW; w++)); do worker $((w % GPUS)) & done; wait
STOPPED=0; [ -f "$CLAIMS/.stopped" ] && STOPPED=1
# 4) degrade + YOLO labels (+ previews for the test run)
if [ "$MODE" = test ]; then $PY render/post.py "$OUT" --preview; else $PY render/post.py "$OUT"; fi
if [ "$STOPPED" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
  echo "stopping Runpod pod $RUNPOD_POD_ID to stop GPU billing (the /workspace volume and results stay)"; runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
echo "done: $OUT/images, $OUT/labels_yolo (classes 0 bumper_red, 1 bumper_blue, 2 fuel), $OUT/dataset.yaml"
[ "$MODE" = test ] && echo "check the boxes in $OUT/preview/"
