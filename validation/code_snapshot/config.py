"""Configuration and environment management for FRC Match Video Downloader."""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


def load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """Simple .env file loader without third-party dependencies."""
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.is_file():
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"[Warning] Failed to load .env file: {e}", file=sys.stderr)


def detect_gdrive_directory() -> Optional[Path]:
    """Attempts to auto-detect the local Google Drive sync folder on macOS/Linux/Windows."""
    home = Path.home()

    # Priority 1: Common macOS CloudStorage paths (macOS 12.3+ live Google Drive Desktop mount)
    cloud_storage = home / "Library" / "CloudStorage"
    if cloud_storage.is_dir():
        try:
            for item in cloud_storage.glob("GoogleDrive*"):
                my_drive = item / "My Drive"
                if my_drive.is_dir():
                    return my_drive
                if item.is_dir():
                    return item
        except Exception:
            pass

    # Priority 2: Classic macOS sync folder "~/Google Drive/My Drive"
    mac_gdrive_mydrive = home / "Google Drive" / "My Drive"
    if mac_gdrive_mydrive.is_dir():
        return mac_gdrive_mydrive

    mac_gdrive_root = home / "Google Drive"
    if mac_gdrive_root.is_dir():
        return mac_gdrive_root

    # Priority 3: Classic Volumes mount
    volumes_gdrive = Path("/Volumes/GoogleDrive/My Drive")
    if volumes_gdrive.is_dir():
        return volumes_gdrive

    volumes_gdrive_root = Path("/Volumes/GoogleDrive")
    if volumes_gdrive_root.is_dir():
        return volumes_gdrive_root

    # Priority 4: Windows common drive letters
    for drive in ["G", "H", "I", "D", "E"]:
        win_drive = Path(f"{drive}:/My Drive")
        if win_drive.is_dir():
            return win_drive

    return None


def is_writable(path: Path) -> bool:
    """Checks whether path (or its nearest existing parent) is writable."""
    try:
        curr = path
        while not curr.exists() and curr != curr.parent:
            curr = curr.parent
        if not curr.exists():
            return False
        test_file = curr / f".write_test_{os.getpid()}"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        return True
    except Exception:
        return False


@dataclass
class AppConfig:
    tba_api_key: str = ""
    gdrive_dir: Path = field(default_factory=lambda: Path("./downloads/gdrive"))
    current_year: int = 2026
    multi_years: List[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    max_videos_this_year: int = 100
    max_videos_multi_year: int = 100
    video_quality: str = "best"  # "best", "1080p", "720p", "480p"
    cache_dir: Path = field(default_factory=lambda: Path(".cache/tba"))
    dry_run: bool = False
    extract_frames: bool = True
    yt_dlp_path: str = "yt-dlp"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    @property
    def this_year_dir(self) -> Path:
        return self.gdrive_dir / "frc_this_year"

    @property
    def multi_year_dir(self) -> Path:
        return self.gdrive_dir / "frc_multi_year"

    @property
    def frames_dir(self) -> Path:
        return self.gdrive_dir / "frc_extracted_frames"

    @classmethod
    def from_env_and_args(cls, args=None) -> "AppConfig":
        load_dotenv()

        api_key = (
            (args and getattr(args, "api_key", None))
            or os.environ.get("TBA_AUTH_KEY")
            or os.environ.get("TBA_API_KEY")
            or ""
        )

        # Google Drive dir detection / resolution
        gdrive_arg = args and getattr(args, "gdrive_dir", None)
        env_gdrive = os.environ.get("GDRIVE_DIR")
        if gdrive_arg:
            gdrive_dir = Path(gdrive_arg).expanduser().resolve()
        elif env_gdrive:
            gdrive_dir = Path(env_gdrive).expanduser().resolve()
        else:
            detected = detect_gdrive_directory()
            if detected:
                gdrive_dir = detected / "FRC_Match_Videos"
            else:
                gdrive_dir = Path("./downloads/gdrive").resolve()

        # Check writability of target gdrive_dir; if not writable (e.g. sandbox restriction),
        # warn and fall back to local workspace downloads if user didn't explicitly override
        if not is_writable(gdrive_dir) and not gdrive_arg:
            local_fallback = Path("./downloads/gdrive").resolve()
            print(f"[Notice] Google Drive folder '{gdrive_dir}' is currently not directly writable.")
            print(f"         Using local folder '{local_fallback}'. (To write directly to Google Drive, run outside sandbox).")
            gdrive_dir = local_fallback

        # Years: 2026 for this year, 2021-2025 for multi-year
        curr_year_val = (
            (args and getattr(args, "this_year", None))
            or os.environ.get("FRC_CURRENT_YEAR")
            or 2026
        )
        current_year = int(curr_year_val)

        multi_years_str = (
            (args and getattr(args, "multi_years", None))
            or os.environ.get("FRC_MULTI_YEARS")
            or "2021,2022,2023,2024,2025"
        )
        if isinstance(multi_years_str, str):
            multi_years = [int(y.strip()) for y in multi_years_str.split(",") if y.strip().isdigit()]
        else:
            multi_years = list(multi_years_str)

        # Max counts: default 100
        max_this = int(
            (args and getattr(args, "max_this_year", None))
            or os.environ.get("MAX_VIDEOS_THIS_YEAR")
            or 100
        )
        max_multi = int(
            (args and getattr(args, "max_multi_year", None))
            or os.environ.get("MAX_VIDEOS_MULTI_YEAR")
            or 100
        )

        quality = (
            (args and getattr(args, "quality", None))
            or os.environ.get("VIDEO_QUALITY")
            or "best"
        )

        dry_run = bool(args and getattr(args, "dry_run", False))
        extract_frames = not bool(args and getattr(args, "no_frames", False))

        # yt-dlp path
        yt_dlp = (args and getattr(args, "yt_dlp_path", None)) or os.environ.get("YT_DLP_PATH") or "/opt/homebrew/bin/yt-dlp"
        for candidate in [yt_dlp, "/opt/homebrew/bin/yt-dlp", "/usr/local/bin/yt-dlp", "/opt/anaconda3/bin/yt-dlp"]:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                yt_dlp = candidate
                break

        # ffmpeg & ffprobe paths
        ffmpeg = os.environ.get("FFMPEG_PATH") or "ffmpeg"
        for candidate in [ffmpeg, "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                ffmpeg = candidate
                break

        ffprobe = os.environ.get("FFPROBE_PATH") or "ffprobe"
        for candidate in [ffprobe, "/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                ffprobe = candidate
                break

        return cls(
            tba_api_key=api_key,
            gdrive_dir=gdrive_dir,
            current_year=current_year,
            multi_years=multi_years,
            max_videos_this_year=max_this,
            max_videos_multi_year=max_multi,
            video_quality=quality,
            cache_dir=Path(".cache/tba").resolve(),
            dry_run=dry_run,
            extract_frames=extract_frames,
            yt_dlp_path=yt_dlp,
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
        )
