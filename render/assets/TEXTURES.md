# Textures / assets added to render/assets (all CC0)

| file | used for | source | license |
|---|---|---|---|
| carpet/neyland_ii_20.jpg (1024 px, tile 0.5 m in carpet_tile_m.txt) | field carpet (CARPET_NEYLAND_II_20 base colour + bump from luminance) | ambientCG **Carpet001** (loop-pile carpet), Color map of the 2K-JPG pack: https://ambientcg.com/view?id=Carpet001 , download https://ambientcg.com/get?file=Carpet001_2K-JPG.zip ; desaturated to 85 %, contrast x1.6, re-centred to sRGB mean ~(0.30, 0.31, 0.33) (dark-mid grey like Shaw Neyland II 20), resized to 1024 | CC0 1.0 (ambientCG, https://docs.ambientcg.com/license/) |
| metal/brushed_alu_color.jpg, metal/brushed_alu_roughness.jpg (1024 px) | field ALU_* materials and robot r_alu materials (box-projected in object space, 0.4 m tile; colour modulates the base tint, roughness map drives roughness +-0.12) | ambientCG **Metal011** (brushed metal), 1K-JPG: https://ambientcg.com/view?id=Metal011 , https://ambientcg.com/get?file=Metal011_1K-JPG.zip | CC0 1.0 (ambientCG) |
| concrete/smooth_concrete_floor_diff_1k.jpg | ARENA_FLOOR (arena floor outside the carpet; luminance x random dark tint per frame) | Poly Haven **smooth_concrete_floor** (Dimitrios Savva), https://polyhaven.com/a/smooth_concrete_floor , file https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/smooth_concrete_floor/smooth_concrete_floor_diff_1k.jpg | CC0 1.0 (Poly Haven, https://polyhaven.com/license) |
| crowd/crowd_proxies.blend (2.3 MB) | stand crowd proxies (42 = 14 avatars x idle/cheer/clap, ~1.4k tris, flat colours, shirt/pants per-instance) | built from models/people/glb (Microsoft Rocketbox, MIT, see models/people/LICENSE_ROCKETBOX.md) by render/crowd/build_crowd_proxies.py | MIT (Rocketbox) |

Downloaded Oct 9 2026 by the render pipeline maintainer; no other files from those packs are shipped.
