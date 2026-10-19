FRC synthetic bumper/fuel dataset - cloud GPU render kit (Eddie, Oct 9 2026)

1. Start any Linux GPU machine with an NVIDIA driver (RunPod / Vast / Lambda, Ubuntu). One RTX 4090/A6000/A100-class GPU is plenty.
2. Put all 14 files frc_render_bundle.tar.part00 ... part13 (+ SHA256SUMS.txt) and setup.sh in one folder on it, e.g. /workspace/parts
   (download them from this Drive folder, or upload them from your computer with scp / the provider's file upload).
3. bash /workspace/parts/setup.sh /workspace/parts          (installs Blender 4.5 + Python deps, unpacks to /workspace/frc)
4. bash /workspace/frc/render/render.sh --test              (50 frames -> /workspace/frc/out_test, check preview/ for the boxes)
5. bash /workspace/frc/render/render.sh --full 10000        (resumable: if the machine stops, run the same command again)

Output: out_full/images/*.jpg, out_full/labels_yolo/*.txt (YOLO: 0 bumper_red, 1 bumper_blue, 2 fuel), out_full/dataset.yaml,
plus out_full/labels/*.json with full detail (team numbers, amodal boxes, camera, lighting).
Options: --samples 64 --res 1920x1080 --gpus N --shard i/n (split across several machines: i = 0..n-1).
