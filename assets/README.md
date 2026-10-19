# Assets

STEP→GLB converters and validators for robots, stadiums, and people.

## Directory Structure

```
assets/
├── convert_robots.py       # STEP→GLB robot converter with decimation + bumper detection
├── convert_stadiums.py     # Stadium model converter
├── build_field_map.py      # Build REBUILT field map from CAD
├── build_maps.py           # Build field region/placement maps
├── generate_frame.py       # Seeded scene-layout generator
├── validate_frames.py      # Validate generated frame JSONs
├── validate_manifest.py    # Validate asset manifest
├── verify_stadiums.py      # Verify converted stadium GLBs
├── inspect_stadiums.py     # Inspect stadium models
├── inspect_step.py         # Inspect STEP file structure
├── people_venue.py         # Place field staff in venue
├── plot_frame.py           # Plot scene layout
├── plot_maps.py            # Plot field maps
├── preview_stadium.py      # Render stadium previews
├── stadium_anchor.py       # Align field to stadium anchors
└── people/                 # Avatar/field-staff pipeline
    ├── convert_avatar.py   # Avatar GLB converter
    ├── build_people.py     # Build people library
    ├── assemble_json.py    # Assemble people manifest
    ├── people_blender.py   # Blender people import
    ├── render_preview.py   # Render people previews
    ├── render_sheet.py     # Render contact sheet
    ├── shirt_stats.py      # Shirt color statistics
    ├── verify_glb.py       # Verify GLB models
    └── inspect_glb.py      # Inspect GLB structure
```

## Robot Conversion

Convert STEP files from Onshape/GrabCAD to GLB:

```bash
python convert_robots.py \
    --input robots/*.step \
    --output models/robots/ \
    --manifest manifest.json
```

Features:
- Automatic mesh decimation for render performance
- Bumper detection and material assignment
- Frame dimensions extraction
- Manifest generation for the randomizer

## Stadium Conversion

```bash
python convert_stadiums.py \
    --input stadiums/*.step \
    --output models/stadiums/
```

## Field Maps

Build field placement maps from the REBUILT field CAD:

```bash
# Build field geometry from STEP
python build_field_map.py --input field.step

# Generate placement maps (1cm grid)
python build_maps.py
```

Output:
- `field_maps.npz` — Floor height, clearance, zones, obstacles
- `regions.json` — Alliance zones, barrier bands, hub positions

## Scene Generation

Generate randomized scene layouts:

```bash
python generate_frame.py \
    --n-frames 100 \
    --seed 42 \
    --manifest manifest.json \
    --out-dir frames/
```

Each frame JSON includes:
- 6 robot poses (3 red, 3 blue)
- Fuel positions
- Match phase (teleop/match_start)
- Camera parameters

## People Pipeline

The `people/` directory handles field staff and crowd avatars:

```bash
# Convert avatar models
python people/convert_avatar.py --input avatars/*.fbx

# Build people library
python people/build_people.py

# Preview
python people/render_preview.py
```

## Coordinate System

All outputs use the field coordinate frame:
- **Origin**: Blue alliance wall, driver-right corner
- **+X**: Toward red wall
- **+Y**: Left (as blue drivers see it)
- **+Z**: Up
- **Units**: Meters and radians

Field dimensions: 16.54 × 8.07 m
