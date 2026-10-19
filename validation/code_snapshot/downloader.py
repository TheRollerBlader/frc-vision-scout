"""Video downloader using yt-dlp with real-time micro-progress streaming and manifest generation."""

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from progress import MicroProgressTracker
from sampler import MatchVideoRecord


class VideoDownloader:
    """Manages downloading YouTube match videos using yt-dlp with live micro-progress."""

    QUALITY_MAP = {
        "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "best": "bestvideo+bestaudio/best",
    }

    # Matches yt-dlp progress lines like:
    # [download]  45.2% of   12.62MiB at    9.58MiB/s ETA 00:01
    # [download]  45.2% of ~  12.62MiB at    9.58MiB/s ETA 00:01
    PROGRESS_RE = re.compile(
        r"\[download\]\s+(\d+(?:\.\d+)?)%\s+of\s+~?\s*(\S+)(?:\s+at\s+(\S+))?(?:\s+ETA\s+(\S+))?"
    )
    FORMAT_INFO_RE = re.compile(r"\[info\]\s+\S+:\s+Downloading\s+\d+\s+format\(s\):\s+(\S+)")

    def __init__(
        self,
        yt_dlp_path: str = "yt-dlp",
        quality: str = "best",
        ffmpeg_path: str = "ffmpeg",
        progress_callback: Optional[Callable[[str], None]] = None,
        tracker: Optional[MicroProgressTracker] = None,
    ):
        self.yt_dlp_path = self._resolve_yt_dlp(yt_dlp_path)
        self.ffmpeg_path = self._resolve_ffmpeg(ffmpeg_path)
        self.quality_format = self.QUALITY_MAP.get(quality, quality)
        self.js_runtime_arg = self._find_js_runtime()
        self.tracker = tracker
        if progress_callback is not None:
            self.log = progress_callback
        elif tracker is not None:
            self.log = tracker.log
        else:
            self.log = print

    def _resolve_ffmpeg(self, path: str) -> Optional[str]:
        """Finds executable ffmpeg in PATH or common locations."""
        resolved = shutil.which(path)
        if resolved and os.access(resolved, os.X_OK):
            return resolved
        for fallback in [
            path,
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/anaconda3/bin/ffmpeg",
        ]:
            if Path(fallback).is_file() and os.access(fallback, os.X_OK):
                return fallback
        return None

    def _resolve_yt_dlp(self, path: str) -> str:
        """Finds executable yt-dlp in PATH or specified path."""
        for fallback in [
            path if Path(path).is_absolute() else "",
            "/opt/homebrew/bin/yt-dlp",
            "/usr/local/bin/yt-dlp",
            "/opt/anaconda3/bin/yt-dlp",
            str(Path.home() / ".local/bin/yt-dlp"),
        ]:
            if fallback and Path(fallback).is_file() and os.access(fallback, os.X_OK):
                return fallback

        resolved = shutil.which(path)
        if resolved and os.access(resolved, os.X_OK):
            return resolved

        return path

    def _find_js_runtime(self) -> Optional[str]:
        """Detects available JavaScript runtime for yt-dlp (e.g. node, deno)."""
        node_bin = shutil.which("node") or "/opt/homebrew/bin/node" or "/usr/local/bin/node"
        if Path(node_bin).is_file() and os.access(node_bin, os.X_OK):
            return f"node:{node_bin}"
        deno_bin = shutil.which("deno")
        if deno_bin and os.access(deno_bin, os.X_OK):
            return f"deno:{deno_bin}"
        return None

    def verify_yt_dlp(self) -> bool:
        """Checks if yt-dlp runs properly."""
        try:
            res = subprocess.run(
                [self.yt_dlp_path, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def parse_yt_dlp_line(
        cls,
        line: str,
        is_multi_stream: bool,
        stream_index: int,
    ) -> tuple[Optional[float], Optional[str], bool, int]:
        """Parses a single yt-dlp stdout line and returns (micro_fraction, sub_detail, is_multi_stream, stream_index)."""
        stripped = line.strip()
        if not stripped:
            return None, None, is_multi_stream, stream_index

        fmt_match = cls.FORMAT_INFO_RE.search(stripped)
        if fmt_match:
            fmt_spec = fmt_match.group(1)
            is_multi_stream = "+" in fmt_spec
            return 0.02, f"Format {fmt_spec}", is_multi_stream, stream_index

        if stripped.startswith("[youtube]") and "Downloading" in stripped:
            return 0.01, "Fetching stream metadata", is_multi_stream, stream_index

        if stripped.startswith("[download] Destination:"):
            stream_index += 1
            stream_type = "Video stream" if (not is_multi_stream or stream_index == 1) else "Audio stream"
            base_frac = 0.02 if stream_index <= 1 else 0.85
            return base_frac, f"Starting {stream_type}", is_multi_stream, stream_index

        prog_match = cls.PROGRESS_RE.search(stripped)
        if prog_match:
            raw_pct = float(prog_match.group(1))
            pct = max(0.0, min(1.0, raw_pct / 100.0))
            total_size = prog_match.group(2) or ""
            speed = prog_match.group(3) or ""
            eta = prog_match.group(4) or ""

            if not is_multi_stream:
                micro_frac = pct
                stream_tag = ""
            else:
                if stream_index <= 1:
                    # Video stream accounts for 0% -> 85% of the video download
                    micro_frac = pct * 0.85
                    stream_tag = "vid "
                else:
                    # Audio stream accounts for 85% -> 97% of the video download
                    micro_frac = 0.85 + (pct * 0.12)
                    stream_tag = "aud "

            detail_parts = [f"{stream_tag}{raw_pct:.1f}% of {total_size}"]
            if speed and speed != "Unknown":
                detail_parts.append(f"@ {speed}")
            if eta and eta != "Unknown":
                detail_parts.append(f"ETA {eta}")
            return micro_frac, " ".join(detail_parts), is_multi_stream, stream_index

        if stripped.startswith("[Merger]"):
            return 0.98, "Merging video & audio -> MP4", is_multi_stream, stream_index

        return None, None, is_multi_stream, stream_index

    def download_records(
        self,
        records: List[MatchVideoRecord],
        output_dir: Path,
        dry_run: bool = False,
        phase_label: str = "Download",
        force_redownload: bool = False,
        cloud_uploader=None,
        remote_folder_tag: str = "",
        frame_extractor=None,
        frames_root_dir: Optional[Path] = None,
        delete_local_after_upload: bool = False,
    ) -> List[dict]:
        """Downloads videos for given records into output_dir with real-time micro-progress.

        When cloud_uploader is provided, each video (and its 5 extracted frames) is uploaded
        directly to Google Drive in the cloud via HTTPS right after download, and optionally
        removed from local disk immediately so local storage is never exhausted.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        folder_tag = remote_folder_tag or output_dir.name

        results = []
        all_frame_records = []
        total = len(records)
        item_weight = 0.2 if dry_run else (12.5 if (cloud_uploader and frame_extractor) else 10.0)

        self.log(f"\nTarget Folder: {output_dir}")
        if cloud_uploader:
            self.log(f"Direct Cloud Destination: {cloud_uploader.remote_spec(folder_tag)}")
        self.log(f"Preparing to process {total} videos (dry_run={dry_run}, quality={self.quality_format})...")

        for idx, rec in enumerate(records, start=1):
            filename_stem = f"{rec.year}_{rec.event_key}_{rec.match_key}_{rec.youtube_id}"
            out_template = str(output_dir / f"{filename_stem}.%(ext)s")
            item_desc = f"[{idx}/{total}] {rec.match_key} ({rec.location})"
            result_entry = rec.to_dict()

            if self.tracker:
                self.tracker.pop_upcoming_video()
                self.tracker.start_item(
                    phase_label=phase_label,
                    item_label=item_desc,
                    weight=item_weight,
                    sub_detail="Checking local files",
                    meta=result_entry,
                )

            # Check if video already exists with any extension (.mp4, .mkv, .webm)
            existing = list(output_dir.glob(f"{filename_stem}.*"))
            existing_completed = [
                f for f in existing if not f.name.endswith((".part", ".ytdl", ".temp"))
            ]

            if existing_completed and not force_redownload:
                fsize = existing_completed[0].stat().st_size if existing_completed[0].is_file() else 0
                fsize_mib = fsize / (1024.0 * 1024.0)
                self.log(f"[{idx}/{total}] [SKIP] Already downloaded: {existing_completed[0].name} ({fsize_mib:.2f} MiB)")
                result_entry["status"] = "already_exists"
                result_entry["file_path"] = str(existing_completed[0])
                result_entry["file_size_bytes"] = fsize
                result_entry["file_size_mib"] = round(fsize_mib, 2)
                if cloud_uploader and not dry_run:
                    cloud_uploader.upload_file(
                        existing_completed[0],
                        f"{folder_tag}/{existing_completed[0].name}",
                        progress_start=0.5,
                        progress_end=1.0,
                        delete_local_after=delete_local_after_upload,
                    )
                results.append(result_entry)
                if self.tracker:
                    self.tracker.record_video_Completed(
                        result_entry, "already_exists", existing_completed[0].name, file_size_bytes=fsize
                    )
                    self.tracker.finish_item(sub_detail=f"Cached ({fsize_mib:.2f} MiB)")
                continue

            if existing and force_redownload and not dry_run:
                for old_f in existing:
                    try:
                        old_f.unlink()
                    except Exception:
                        pass

            if dry_run:
                self.log(f"[{idx}/{total}] [DRY-RUN] Would download: {rec.youtube_url} ({rec.location} - {rec.event_name})")
                result_entry["status"] = "dry_run"
                result_entry["file_path"] = str(output_dir / f"{filename_stem}.mp4")
                results.append(result_entry)
                if self.tracker:
                    self.tracker.record_video_Completed(result_entry, "dry_run", f"{filename_stem}.mp4")
                    self.tracker.finish_item(sub_detail="Dry-run")
                continue

            cmd = [
                self.yt_dlp_path,
                "--no-update",
                "--no-playlist",
                "--progress",
                "--newline",
                "-f", self.quality_format,
                "--merge-output-format", "mp4",
                "--output", out_template,
                "--socket-timeout", "30",
                "--retries", "3",
            ]
            if self.ffmpeg_path:
                cmd.extend(["--ffmpeg-location", self.ffmpeg_path])
            if self.js_runtime_arg:
                cmd.extend(["--js-runtimes", self.js_runtime_arg])

            cmd.append(rec.youtube_url)

            # Scale yt-dlp download progress to [0.0, 0.65] if cloud upload follows, else [0.0, 1.0]
            dl_scale = 0.65 if cloud_uploader else 1.0

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                is_multi_stream = False
                stream_index = 0
                output_lines: List[str] = []

                if proc.stdout:
                    for raw_line in proc.stdout:
                        output_lines.append(raw_line.rstrip("\n"))
                        micro_frac, detail, is_multi_stream, stream_index = self.parse_yt_dlp_line(
                            raw_line, is_multi_stream, stream_index
                        )
                        if self.tracker and micro_frac is not None:
                            self.tracker.update_item(micro_frac * dl_scale, sub_detail=detail)

                retcode = proc.wait()

                if retcode == 0:
                    found = list(output_dir.glob(f"{filename_stem}.*"))
                    completed_files = [f for f in found if not f.name.endswith((".part", ".ytdl"))]
                    downloaded_file = completed_files[0] if completed_files else output_dir / f"{filename_stem}.mp4"
                    fsize = downloaded_file.stat().st_size if downloaded_file.is_file() else 0
                    fsize_mib = fsize / (1024.0 * 1024.0)
                    self.log(f"[{idx}/{total}] ✓ Downloaded: {downloaded_file.name} ({fsize_mib:.2f} MiB | {rec.location})")
                    result_entry["status"] = "downloaded"
                    result_entry["file_path"] = str(downloaded_file)
                    result_entry["file_size_bytes"] = fsize
                    result_entry["file_size_mib"] = round(fsize_mib, 2)

                    # Optional per-video frame extraction + direct Google Drive cloud upload
                    if frame_extractor and frames_root_dir and downloaded_file.is_file():
                        if self.tracker:
                            self.tracker.update_item(0.68, sub_detail="Extracting 5 frames...")
                        subfolder = frames_root_dir / folder_tag / downloaded_file.stem
                        f_recs = frame_extractor.extract_frames_for_video(
                            downloaded_file,
                            subfolder,
                            video_metadata=result_entry,
                            force_reextract=force_redownload,
                            update_tracker_fraction=False,
                        )
                        all_frame_records.extend(f_recs)
                        if cloud_uploader:
                            if self.tracker:
                                self.tracker.update_item(0.74, sub_detail="Uploading 5 frames to Google Drive...")
                            cloud_uploader.sync_directory_to_cloud(
                                subfolder,
                                f"frc_extracted_frames/{folder_tag}/{downloaded_file.stem}",
                                progress_start=0.74,
                                progress_end=0.80,
                                delete_local_files_after=delete_local_after_upload,
                            )

                    if cloud_uploader and downloaded_file.is_file():
                        if self.tracker:
                            self.tracker.update_item(0.80, sub_detail="Uploading video to Google Drive...")
                        uploaded_ok = cloud_uploader.upload_file(
                            downloaded_file,
                            f"{folder_tag}/{downloaded_file.name}",
                            progress_start=0.80,
                            progress_end=0.99,
                            delete_local_after=delete_local_after_upload,
                        )
                        if uploaded_ok:
                            result_entry["cloud_path"] = cloud_uploader.remote_spec(f"{folder_tag}/{downloaded_file.name}")
                            self.log(f"[{idx}/{total}] ☁ Uploaded to Google Drive: {result_entry['cloud_path']}")

                    if self.tracker:
                        self.tracker.record_video_Completed(
                            result_entry, "downloaded", downloaded_file.name, file_size_bytes=fsize
                        )
                        self.tracker.finish_item(sub_detail=f"100% ({fsize_mib:.2f} MiB)")
                else:
                    err_lines = [l for l in output_lines if "ERROR:" in l] or output_lines
                    err_snippet = err_lines[-1] if err_lines else "Unknown error"
                    self.log(f"[{idx}/{total}] ✗ Failed ({rec.match_key}): {err_snippet}")
                    result_entry["status"] = "failed"
                    result_entry["error"] = err_snippet
                    if self.tracker:
                        self.tracker.record_video_Completed(result_entry, "failed", err_snippet[:60])
                        self.tracker.finish_item(sub_detail="Failed")
            except Exception as e:
                self.log(f"[{idx}/{total}] ✗ Error executing yt-dlp: {e}")
                result_entry["status"] = "error"
                result_entry["error"] = str(e)
                if self.tracker:
                    self.tracker.record_video_Completed(result_entry, "error", str(e)[:60])
                    self.tracker.finish_item(sub_detail="Error")

            results.append(result_entry)

        # Write manifest.json and manifest.csv (and upload them to cloud if cloud_uploader is active)
        self._write_manifests(output_dir, results)
        if cloud_uploader and not dry_run:
            for mf in ("manifest.json", "manifest.csv"):
                mp = output_dir / mf
                if mp.is_file():
                    cloud_uploader.upload_file(mp, f"{folder_tag}/{mf}")
            if frame_extractor and frames_root_dir and all_frame_records:
                target_frames_folder = frames_root_dir / folder_tag
                target_frames_folder.mkdir(parents=True, exist_ok=True)
                frame_extractor._write_manifest(target_frames_folder, all_frame_records)
                for mf in ("frames_manifest.json", "frames_manifest.csv"):
                    mp = target_frames_folder / mf
                    if mp.is_file():
                        cloud_uploader.upload_file(mp, f"frc_extracted_frames/{folder_tag}/{mf}")
        return results

    def _write_manifests(self, output_dir: Path, results: List[dict]) -> None:
        """Saves metadata manifest in JSON and CSV formats."""
        json_path = output_dir / "manifest.json"
        csv_path = output_dir / "manifest.csv"

        try:
            if json_path.exists():
                json_path.unlink()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            if results:
                fieldnames = list(results[0].keys())
                for r in results:
                    for k in r.keys():
                        if k not in fieldnames:
                            fieldnames.append(k)

                if csv_path.exists():
                    csv_path.unlink()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)

            self.log(f"[Manifest] Saved manifests to {json_path.name} and {csv_path.name}")
        except Exception as e:
            self.log(f"[Warning] Failed to write manifests: {e}")
