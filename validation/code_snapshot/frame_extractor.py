"""Frame extractor for FRC match videos using ffmpeg/ffprobe with micro-progress updates."""

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

from progress import MicroProgressTracker


class FrameExtractor:
    """Extracts 5 key frames (start, middle 1, middle 2, middle 3, end) from videos."""

    # Relative position points across the video duration (0.0 to 1.0)
    FRAME_POINTS = [
        ("start", 0.05),     # 5% in to skip intro/black cards
        ("middle_1", 0.275), # Autonomous / early teleop
        ("middle_2", 0.50),  # Mid-match
        ("middle_3", 0.725), # Late teleop / prep
        ("end", 0.95),       # Endgame / final buzzer
    ]

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        progress_callback: Optional[Callable[[str], None]] = None,
        tracker: Optional[MicroProgressTracker] = None,
    ):
        self.ffmpeg_path = self._resolve_binary(ffmpeg_path, ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"])
        self.ffprobe_path = self._resolve_binary(ffprobe_path, ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"])
        self.tracker = tracker
        if progress_callback is not None:
            self.log = progress_callback
        elif tracker is not None:
            self.log = tracker.log
        else:
            self.log = print

    def _resolve_binary(self, name: str, candidates: List[str]) -> str:
        resolved = shutil.which(name)
        if resolved and os.access(resolved, os.X_OK):
            return resolved
        for c in candidates:
            if Path(c).is_file() and os.access(c, os.X_OK):
                return c
        return name

    def get_duration(self, video_path: Path) -> Optional[float]:
        """Gets video duration in seconds using ffprobe."""
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except Exception:
            pass

        # Fallback to ffmpeg -i parsing
        try:
            cmd = [self.ffmpeg_path, "-i", str(video_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            for line in res.stderr.splitlines():
                if "Duration:" in line:
                    parts = line.split("Duration:")[1].split(",")[0].strip()
                    # format HH:MM:SS.xx
                    h, m, s = parts.split(":")
                    return float(h) * 3600 + float(m) * 60 + float(s)
        except Exception:
            pass

        return None

    def extract_frames_for_video(
        self,
        video_path: Path,
        output_subfolder: Path,
        video_metadata: Optional[dict] = None,
        force_reextract: bool = False,
        update_tracker_fraction: bool = True,
    ) -> List[dict]:
        """Extracts the 5 designated frames from a single video file."""
        output_subfolder.mkdir(parents=True, exist_ok=True)
        duration = self.get_duration(video_path)

        if duration is None or duration <= 0:
            self.log(f"  [Warning] Could not determine duration for {video_path.name}; defaulting to 150s.")
            duration = 150.0  # Standard FRC match length is 2m30s = 150s

        extracted_records = []
        total_frames = len(self.FRAME_POINTS)

        for idx, (label, pct) in enumerate(self.FRAME_POINTS, start=1):
            timestamp = max(0.0, round(duration * pct, 2))
            filename = f"frame_{idx}_{label}.jpg"
            image_path = output_subfolder / filename

            if self.tracker and update_tracker_fraction:
                self.tracker.update_item(
                    fraction=(idx - 1) / total_frames,
                    sub_detail=f"Frame {idx}/{total_frames}: {label} (@ {timestamp:.1f}s)",
                )

            rec = {
                "source_video": video_path.name,
                "video_path": str(video_path),
                "frame_index": idx,
                "frame_label": label,
                "timestamp_seconds": timestamp,
                "image_filename": filename,
                "image_path": str(image_path),
            }
            if video_metadata:
                for k in ["year", "event_key", "event_name", "location", "match_key", "youtube_id", "youtube_url"]:
                    if k in video_metadata:
                        rec[k] = video_metadata[k]

            if image_path.is_file() and image_path.stat().st_size > 0 and not force_reextract:
                rec["status"] = "already_exists"
                extracted_records.append(rec)
                if self.tracker:
                    self.tracker.record_frame_extracted(1)
                    if update_tracker_fraction:
                        self.tracker.update_item(
                            fraction=idx / total_frames,
                            sub_detail=f"Frame {idx}/{total_frames}: {label} (cached)",
                        )
                continue

            if image_path.exists():
                try:
                    image_path.unlink()
                except Exception:
                    pass

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(image_path),
            ]

            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and image_path.is_file():
                rec["status"] = "extracted"
            else:
                rec["status"] = "failed"
                rec["error"] = res.stderr.strip().splitlines()[-1] if res.stderr else "ffmpeg error"

            if self.tracker:
                self.tracker.record_frame_extracted(1)
                if update_tracker_fraction:
                    self.tracker.update_item(
                        fraction=idx / total_frames,
                        sub_detail=f"Frame {idx}/{total_frames}: {label} done",
                    )

            extracted_records.append(rec)

        return extracted_records

    def process_folder(
        self,
        video_folder: Path,
        frames_root_dir: Path,
        folder_tag: str,
        manifest_path: Optional[Path] = None,
        item_weight: float = 2.5,
    ) -> List[dict]:
        """Extracts 5 frames for every video in video_folder into frames_root_dir/folder_tag/<video_stem>/."""
        target_frames_folder = frames_root_dir / folder_tag
        target_frames_folder.mkdir(parents=True, exist_ok=True)

        # Load video metadata manifest if available
        meta_by_filename: Dict[str, dict] = {}
        if manifest_path and manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    for item in manifest_data:
                        fp = item.get("file_path")
                        if fp:
                            meta_by_filename[Path(fp).name] = item
            except Exception:
                pass

        video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
        video_files = [
            f for f in sorted(video_folder.iterdir())
            if f.is_file() and f.suffix.lower() in video_extensions and not f.name.endswith((".part", ".ytdl"))
        ]

        self.log(f"\n[Frames] Extracting 5 frames per video for {len(video_files)} videos in {folder_tag}...")
        self.log(f"Destination: {target_frames_folder}")

        all_frame_results = []
        for i, vfile in enumerate(video_files, start=1):
            subfolder = target_frames_folder / vfile.stem
            vmeta = meta_by_filename.get(vfile.name)
            if self.tracker:
                self.tracker.start_item(
                    phase_label=f"Frames ({folder_tag})",
                    item_label=f"[{i}/{len(video_files)}] {vfile.stem}",
                    weight=item_weight,
                    sub_detail="Probing duration...",
                    meta=vmeta,
                )
            frame_records = self.extract_frames_for_video(vfile, subfolder, video_metadata=vmeta)
            if self.tracker:
                self.tracker.finish_item(sub_detail="5/5 frames extracted")
            self.log(f"[{i}/{len(video_files)}] ✓ Extracted 5 frames for {vfile.stem}")
            all_frame_results.extend(frame_records)

        # Write frames manifest
        self._write_manifest(target_frames_folder, all_frame_results)
        return all_frame_results

    def _write_manifest(self, folder: Path, records: List[dict]) -> None:
        if not records:
            return
        json_path = folder / "frames_manifest.json"
        csv_path = folder / "frames_manifest.csv"
        try:
            if json_path.exists():
                json_path.unlink()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

            fieldnames = list(records[0].keys())
            for r in records:
                for k in r.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)

            if csv_path.exists():
                csv_path.unlink()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

            self.log(f"[Frames] Saved frames manifest to {json_path.name} and {csv_path.name}")
        except Exception as e:
            self.log(f"[Warning] Failed to write frames manifest: {e}")
