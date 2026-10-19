#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU machine (RunPod / Vast / Lambda / any Ubuntu + NVIDIA driver).
#   bash setup.sh [ASSETS]      ASSETS = folder with the frc_render_bundle.tar.part* files, a .tar path, or an http(s) URL
#                               (default: ./frc_render_bundle.tar)
set -euo pipefail
ASSETS="${1:-./frc_render_bundle.tar}"
ROOT=/workspace/frc
mkdir -p /workspace && cd /workspace
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || echo "WARNING: no nvidia-smi (CPU fallback will be slow)"
if command -v apt-get >/dev/null; then (apt-get update -qq && apt-get install -y -qq wget xz-utils python3-venv libxi6 libxkbcommon0 libxrender1 libgl1 libsm6 >/dev/null) || true; fi
if [[ "$ASSETS" =~ ^https?:// ]]; then wget -q --show-progress -O /workspace/frc_render_bundle.tar "$ASSETS"; ASSETS=/workspace/frc_render_bundle.tar; fi
# the Drive copy is split into frc_render_bundle.tar.part00..13 (Drive upload limit): pass the folder holding them
if [ -d "$ASSETS" ]; then (cd "$ASSETS" && { [ -f SHA256SUMS.txt ] && sha256sum -c --quiet SHA256SUMS.txt || true; }; cat "$ASSETS"/frc_render_bundle.tar.part* > /workspace/frc_render_bundle.tar); ASSETS=/workspace/frc_render_bundle.tar; fi
[ -d "$ROOT/models" ] || tar -xf "$ASSETS" -C /workspace
BL=/workspace/tools/blender-4.5.9-linux-x64
if [ ! -x "$BL/blender" ]; then
  mkdir -p /workspace/tools && cd /workspace/tools
  wget -q --show-progress https://download.blender.org/release/Blender4.5/blender-4.5.9-linux-x64.tar.xz || \
    wget -q --show-progress "$(wget -qO- https://download.blender.org/release/Blender4.5/ | grep -o 'blender-4\.5\.[0-9]*-linux-x64\.tar\.xz' | sort -V | tail -1 | sed 's|^|https://download.blender.org/release/Blender4.5/|')"
  tar -xf blender-4.5.*-linux-x64.tar.xz && rm blender-4.5.*-linux-x64.tar.xz
  [ -x "$BL/blender" ] || ln -s "$(ls -d /workspace/tools/blender-4.5.*-linux-x64 | head -1)" "$BL"
fi
cd "$ROOT"
python3 -m venv .venv && .venv/bin/pip -q install numpy shapely scipy pillow matplotlib
echo "setup OK. Next: bash $ROOT/render/render.sh --test"
