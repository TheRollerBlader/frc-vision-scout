# Data Collection

Tools for collecting FRC robot CAD models and match videos.

## Directory Structure

```
data_collection/
├── frc_cad/              # Robot CAD scraper & downloader
│   ├── frc_cad_downloader.py   # Main scraper (Onshape/GrabCAD)
│   ├── onshape_relogin.py      # Onshape session refresh
│   ├── full_run.py             # Full-scale stager
│   ├── full_upload.py          # Drive upload helper
│   ├── finalize_index.py       # Build robots_final.json
│   ├── robots_final.json       # Index of collected robot CAD
│   └── onshape_inventory.tsv   # Onshape STEP export inventory
└── tba/
    └── tba_match_videos.ipynb  # TBA API match video downloader
```

## Robot CAD Collection

The `frc_cad/` directory contains scripts for scraping FRC robot CAD from public sources:

- **Spectrum 3847's CAD collection** — Community-maintained library of FRC robot CAD
- **GrabCAD** — User-uploaded FRC robot models
- **Onshape** — Team-shared public documents

### Usage

```bash
# Set credentials
export ONSHAPE_EMAIL="your-email"
export ONSHAPE_PASSWORD="your-password"
export ONSHAPE_COOKIE="session-cookie"
export GRABCAD_COOKIE="session-cookie"

# Run the scraper
python frc_cad/frc_cad_downloader.py
```

The scraper exports STEP files which are then converted to GLB using `assets/convert_robots.py`.

### Output

- `robots_final.json` — Index of all collected robot CAD with metadata
- `onshape_inventory.tsv` — Inventory of Onshape exports with Drive links

## Match Video Collection

The `tba/tba_match_videos.ipynb` notebook queries The Blue Alliance API for match videos.

### Usage

```python
# Set TBA API key
import os
os.environ['TBA_KEY'] = 'your-tba-key'

# Or use getpass for interactive input
```

The notebook:
1. Queries TBA for 2026 season matches
2. Extracts YouTube video URLs
3. Downloads match videos for validation

### TBA API

Get your API key at [thebluealliance.com/account](https://www.thebluealliance.com/account).

## Notes

- Robot CAD belongs to the original FRC teams who created it
- Match broadcasts are property of FIRST and its partners
- This repository does not re-host CAD or videos
