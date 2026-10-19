#!/usr/bin/env bash
# Pack everything the cloud machine needs into one tar (run on Eddie's computer).
set -euo pipefail
cd /workspace
tar -cf /workspace/frc_render_bundle.tar \
  frc/render frc/tools frc/manifest.json frc/field/field_maps.npz frc/field/regions.json \
  frc/models/robots frc/models/people/glb frc/models/people/people.json frc/models/field/rebuilt_field.glb \
  frc/models/field/carpet_placeholder.glb frc/models/field/carpet_placeholder.json \
  frc/models/stadiums/glb frc/models/stadiums/anchors frc/render/setup.sh
ls -la /workspace/frc_render_bundle.tar
