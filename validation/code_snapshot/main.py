"""Main entrypoint for FRC Match Video Downloader with Frame Extraction, Google Drive output, and Localhost Live Dashboard."""

import argparse
import collections
import shutil
import sys
import time
from pathlib import Path

from config import AppConfig
from dashboard import DashboardServer
from downloader import VideoDownloader
from frame_extractor import FrameExtractor
from gdrive_uploader import GoogleDriveCloudUploader
from progress import MicroProgressTracker
from sampler import FRCSampler, MatchVideoRecord
from tba_client import TBAClient


def print_banner():
    print("=" * 70)
    print("   FRC Match Video Downloader & Frame Extractor (TBA + yt-dlp)")
    print("=" * 70)


def print_distribution_summary(
    title: str,
    records: list[MatchVideoRecord],
    log_fn=print,
):
    lines = [f"\n{'-' * 20} {title} Distribution {'-' * 20}"]
    if not records:
        lines.append("No videos selected.")
        log_fn("\n".join(lines))
        return

    by_year = collections.defaultdict(int)
    by_loc = collections.defaultdict(int)
    for r in records:
        by_year[r.year] += 1
        by_loc[r.location] += 1

    if len(by_year) > 1:
        lines.append("By Year:")
        for y in sorted(by_year.keys()):
            lines.append(f"  {y}: {by_year[y]} videos")

    lines.append(f"By Location ({len(by_loc)} distinct locations):")
    for loc, count in sorted(by_loc.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {loc:<30} : {count} videos")
    lines.append(f"Total: {len(records)} videos")
    lines.append("-" * 65)
    log_fn("\n".join(lines))


def count_videos_in_folder(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    return sum(
        1
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in exts and not f.name.endswith((".part", ".ytdl"))
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download FRC match videos from The Blue Alliance evenly sampled by year and location into Google Drive, with 5 key frames extracted per video and a live localhost web dashboard."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The Blue Alliance API v3 Auth Key (defaults to TBA_AUTH_KEY in .env)",
    )
    parser.add_argument(
        "--gdrive-dir",
        type=str,
        default=None,
        help="Target Google Drive folder path (defaults to auto-detected local Google Drive Desktop folder)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "this-year", "multi-year"],
        default="all",
        help="Which dataset to sample and download (default: 'all')",
    )
    parser.add_argument(
        "--this-year",
        type=int,
        default=None,
        help="Season year for 'this year' dataset (default: 2026)",
    )
    parser.add_argument(
        "--multi-years",
        type=str,
        default=None,
        help="Comma-separated years for 'multi-year' dataset (default: '2021,2022,2023,2024,2025')",
    )
    parser.add_argument(
        "--max-this-year",
        type=int,
        default=None,
        help="Max videos to sample for this year (default: 100)",
    )
    parser.add_argument(
        "--max-multi-year",
        type=int,
        default=None,
        help="Max videos to sample across multiple years (default: 100)",
    )
    parser.add_argument(
        "--quality",
        choices=["720p", "1080p", "480p", "best"],
        default=None,
        help="Video resolution limit for yt-dlp (default: best)",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-downloading videos and re-extracting frames even if files already exist locally",
    )
    parser.add_argument(
        "--cloud-upload",
        action="store_true",
        help="Upload videos and frames directly to Google Drive in the cloud via rclone (bypasses macOS Google Drive sync app)",
    )
    parser.add_argument(
        "--delete-local-after-upload",
        action="store_true",
        help="Delete each local video file immediately after uploading it to Google Drive in the cloud to save Mac disk space",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip extracting 5 key frames per video",
    )
    parser.add_argument(
        "--extract-frames-only",
        action="store_true",
        help="Only extract frames from already downloaded videos without querying TBA or downloading new videos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query TBA and preview sampled videos without downloading with yt-dlp",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached TBA API responses before running",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the localhost live web dashboard (default: 8765)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable starting the localhost web dashboard server",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the live dashboard in your default web browser",
    )
    parser.add_argument(
        "--keep-dashboard",
        action="store_true",
        help="Keep the localhost web dashboard server running after completion until Ctrl+C",
    )
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()
    config = AppConfig.from_env_and_args(args)

    if args.clear_cache and config.cache_dir.is_dir():
        print(f"[Cache] Clearing {config.cache_dir}...")
        shutil.rmtree(config.cache_dir, ignore_errors=True)

    target_display = (
        f"gdrive:FRC_Match_Videos (Direct Cloud Upload | local staging: {config.gdrive_dir})"
        if args.cloud_upload
        else str(config.gdrive_dir)
    )
    print(f"Target Google Drive Folder : {target_display}")
    print(f"Folder 1 (This Year {config.current_year}) : {config.this_year_dir}")
    print(f"Folder 2 (Multi-Year {config.multi_years}): {config.multi_year_dir}")
    print(f"Folder 3 (Extracted Frames): {config.frames_dir}")
    print(f"Selected Quality           : {config.video_quality}")
    print(f"Extract 5 Frames per Video : {config.extract_frames}")
    print(f"Direct Cloud Upload        : {args.cloud_upload} (delete_local={args.delete_local_after_upload})")
    print(f"Execution Mode             : {args.mode} (dry_run={config.dry_run})")

    tracker = MicroProgressTracker(enabled=True, gdrive_dir=target_display)

    cloud_uploader = None
    if args.cloud_upload:
        cloud_uploader = GoogleDriveCloudUploader(
            remote_name="gdrive",
            remote_root="FRC_Match_Videos",
            tracker=tracker,
        )

    dashboard_server = None
    if not args.no_dashboard:
        dashboard_server = DashboardServer(tracker=tracker, port=args.port)
        dash_url = dashboard_server.start(open_browser=args.open_browser)
        print(f"Live Web Dashboard         : {dash_url}")
        print("-" * 70)

    frame_extractor = FrameExtractor(
        ffmpeg_path=config.ffmpeg_path,
        ffprobe_path=config.ffprobe_path,
        tracker=tracker,
    )

    try:
        # Standalone mode: extract frames only
        if args.extract_frames_only:
            n1 = count_videos_in_folder(config.this_year_dir)
            n2 = count_videos_in_folder(config.multi_year_dir)
            tracker.configure_plan(extract_only_videos=n1 + n2)
            tracker.log("\n=== Running Frame Extraction on Existing Videos ===")
            if config.this_year_dir.is_dir():
                tracker.set_stage_status("frames_this", "active")
                frame_extractor.process_folder(
                    video_folder=config.this_year_dir,
                    frames_root_dir=config.frames_dir,
                    folder_tag="frc_this_year",
                    manifest_path=config.this_year_dir / "manifest.json",
                    item_weight=5.0,
                )
                tracker.set_stage_status("frames_this", "done")
            if config.multi_year_dir.is_dir():
                tracker.set_stage_status("frames_multi", "active")
                frame_extractor.process_folder(
                    video_folder=config.multi_year_dir,
                    frames_root_dir=config.frames_dir,
                    folder_tag="frc_multi_year",
                    manifest_path=config.multi_year_dir / "manifest.json",
                    item_weight=5.0,
                )
                tracker.set_stage_status("frames_multi", "done")
            tracker.finish_all()
            print("\nFrame extraction completed!")
            return

        # Check API Key
        if not config.tba_api_key:
            print("\n" + "!" * 70)
            print("ERROR: No TBA API key found!")
            print("Please set TBA_AUTH_KEY in your .env file or pass --api-key <YOUR_KEY>.")
            print("Get your free API key at: https://www.thebluealliance.com/account")
            print("!" * 70 + "\n")
            sys.exit(1)

        # Configure micro-progress plan
        planned_this_year = config.max_videos_this_year if args.mode in ("all", "this-year") else 0
        planned_multi_year = config.max_videos_multi_year if args.mode in ("all", "multi-year") else 0
        years_label = f"{min(config.multi_years)}-{max(config.multi_years)}" if config.multi_years else "Multi-Year"
        tracker.configure_plan(
            num_this_year_videos=planned_this_year,
            num_multi_year_videos=planned_multi_year,
            num_multi_years=len(config.multi_years),
            extract_frames=config.extract_frames,
            dry_run=config.dry_run,
            this_year_label=config.current_year,
            multi_years_label=years_label,
        )

        # Initialize TBA Client, Downloader, and Sampler with shared MicroProgressTracker
        client = TBAClient(api_key=config.tba_api_key, cache_dir=config.cache_dir)
        downloader = VideoDownloader(
            yt_dlp_path=config.yt_dlp_path,
            quality=config.video_quality,
            ffmpeg_path=config.ffmpeg_path,
            tracker=tracker,
        )

        if not config.dry_run:
            if not downloader.verify_yt_dlp():
                print(f"\n[Warning] yt-dlp executable not verified at: {downloader.yt_dlp_path}")
                print("Please ensure yt-dlp is installed, or pass --dry-run to preview.")
                sys.exit(1)

        sampler = FRCSampler(client=client, tracker=tracker)
        per_video_unit = 0.2 if config.dry_run else (12.5 if config.extract_frames else 10.0)

        # Mode 1: This Year
        if args.mode in ("all", "this-year"):
            tracker.set_stage_status("discover_this", "active")
            this_year_records = sampler.sample_this_year(
                year=config.current_year,
                target_count=config.max_videos_this_year,
            )
            tracker.set_stage_status("discover_this", "done", detail=f"{len(this_year_records)} sampled")
            tracker.enqueue_upcoming_videos(this_year_records, f"This Year ({config.current_year})")

            diff = len(this_year_records) - planned_this_year
            if diff != 0:
                tracker.add_total_units(diff * per_video_unit)
                tracker.videos_total += diff
                if config.extract_frames and not config.dry_run:
                    tracker.frames_total += diff * 5

            print_distribution_summary(
                f"This Year ({config.current_year})",
                this_year_records,
                log_fn=tracker.log,
            )

            tracker.set_stage_status("download_this", "active")
            downloader.download_records(
                records=this_year_records,
                output_dir=config.this_year_dir,
                dry_run=config.dry_run,
                phase_label=f"Download {config.current_year}",
                force_redownload=args.force_redownload,
                cloud_uploader=cloud_uploader,
                remote_folder_tag="frc_this_year",
                frame_extractor=frame_extractor if (cloud_uploader and config.extract_frames) else None,
                frames_root_dir=config.frames_dir if (cloud_uploader and config.extract_frames) else None,
                delete_local_after_upload=args.delete_local_after_upload,
            )
            tracker.set_stage_status("download_this", "done")

            if config.extract_frames and not config.dry_run:
                if cloud_uploader:
                    tracker.set_stage_status("frames_this", "done", detail="Uploaded per-video")
                else:
                    tracker.set_stage_status("frames_this", "active")
                    actual_vids = count_videos_in_folder(config.this_year_dir)
                    frame_diff = actual_vids - len(this_year_records)
                    if frame_diff != 0:
                        tracker.add_total_units(frame_diff * 2.5)
                        tracker.frames_total += frame_diff * 5
                    frame_extractor.process_folder(
                        video_folder=config.this_year_dir,
                        frames_root_dir=config.frames_dir,
                        folder_tag="frc_this_year",
                        manifest_path=config.this_year_dir / "manifest.json",
                        item_weight=2.5,
                    )
                    tracker.set_stage_status("frames_this", "done")

        # Mode 2: Multi-Year
        if args.mode in ("all", "multi-year"):
            tracker.set_stage_status("discover_multi", "active")
            multi_year_records = sampler.sample_multi_year(
                years=config.multi_years,
                target_count=config.max_videos_multi_year,
            )
            tracker.set_stage_status("discover_multi", "done", detail=f"{len(multi_year_records)} sampled")
            tracker.enqueue_upcoming_videos(multi_year_records, "Multi-Year")

            diff = len(multi_year_records) - planned_multi_year
            if diff != 0:
                tracker.add_total_units(diff * per_video_unit)
                tracker.videos_total += diff
                if config.extract_frames and not config.dry_run:
                    tracker.frames_total += diff * 5

            print_distribution_summary(
                "Multi-Year",
                multi_year_records,
                log_fn=tracker.log,
            )

            tracker.set_stage_status("download_multi", "active")
            downloader.download_records(
                records=multi_year_records,
                output_dir=config.multi_year_dir,
                dry_run=config.dry_run,
                phase_label="Download Multi-Year",
                force_redownload=args.force_redownload,
                cloud_uploader=cloud_uploader,
                remote_folder_tag="frc_multi_year",
                frame_extractor=frame_extractor if (cloud_uploader and config.extract_frames) else None,
                frames_root_dir=config.frames_dir if (cloud_uploader and config.extract_frames) else None,
                delete_local_after_upload=args.delete_local_after_upload,
            )
            tracker.set_stage_status("download_multi", "done")

            if config.extract_frames and not config.dry_run:
                if cloud_uploader:
                    tracker.set_stage_status("frames_multi", "done", detail="Uploaded per-video")
                else:
                    tracker.set_stage_status("frames_multi", "active")
                    actual_vids = count_videos_in_folder(config.multi_year_dir)
                    frame_diff = actual_vids - len(multi_year_records)
                    if frame_diff != 0:
                        tracker.add_total_units(frame_diff * 2.5)
                        tracker.frames_total += frame_diff * 5
                    frame_extractor.process_folder(
                        video_folder=config.multi_year_dir,
                        frames_root_dir=config.frames_dir,
                        folder_tag="frc_multi_year",
                        manifest_path=config.multi_year_dir / "manifest.json",
                        item_weight=2.5,
                    )
                    tracker.set_stage_status("frames_multi", "done")

        tracker.finish_all()

        print("\n" + "=" * 70)
        print("Process completed successfully!")
        print(f"Folder 1 (This Year Videos) : {config.this_year_dir}")
        print(f"Folder 2 (Multi-Year Videos): {config.multi_year_dir}")
        if config.extract_frames and not config.dry_run:
            print(f"Folder 3 (Extracted Frames) : {config.frames_dir}")
        if dashboard_server:
            print(f"Live Web Dashboard          : {dashboard_server.url}")
        print("=" * 70)

        if dashboard_server and args.keep_dashboard:
            print(f"\n[Dashboard] Server remaining open at {dashboard_server.url} (Press Ctrl+C to exit)...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Interrupted] Shutting down...")
    finally:
        if dashboard_server:
            dashboard_server.stop()


if __name__ == "__main__":
    main()
