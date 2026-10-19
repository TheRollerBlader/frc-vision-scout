"""Master End-to-End Orchestrator for FRC YOLO11s Evaluation, Robot Identification & Diagnostics.

Executes the full pipeline:
1. `evaluator.py`: Zero-label FRC rule-invariant evaluation on all extracted frames + annotated JPGs.
2. `robot_identifier.py`: Closed-Set TBA Robot Identification (`frcXXXX`) + BoT-SORT/Kalman Re-ID
   tracking + annotated MP4 videos with 2D Top-Down Tactical Minimap Radar.
3. All-Match Team Cataloger: Expands frame-level vision telemetry to every team (`frcXXXX`) across
   all downloaded matches using cached TBA rosters.
4. `frc_diagnostics.py`: Generates 13 high-resolution charts, ML K-Means/PCA role clustering,
   per-robot scouting cards, and GitHub Markdown reports (`ANALYTICS_REPORT.md` &
   `ROBOT_SCOUTING_ENCYCLOPEDIA.md`).
5. Code & Artifact Backup: Saves a complete snapshot of all Python scripts, configs, charts,
   and reports to `<FRC_DATA_ROOT>/frc_vision_analytics/`.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluator import ZeroLabelEvaluator
from frc_diagnostics import FRCDiagnosticsSuite
from robot_identifier import MatchVideoIdentifier


def build_all_matches_team_appearances(
    df_frames: pd.DataFrame,
    tba_cache_dir: Path,
    repo_data_dir: Path,
    ssd_data_dir: Path,
) -> pd.DataFrame:
    """Catalogs every FRC robot (`frcXXXX`) across ALL evaluated matches (both 2026 and 2021-2025)
    by joining frame-level vision diagnostics with TBA alliance rosters and score breakdowns.
    """
    if df_frames.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (split, year, event_key, match_key), grp in df_frames.groupby(["split", "year", "event_key", "match_key"]):
        cache_file = tba_cache_dir / f"event_{event_key}_matches.json"
        tba_match = None
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    matches = json.load(f)
                if isinstance(matches, list):
                    for m in matches:
                        if m.get("key") == match_key:
                            tba_match = m
                            break
            except Exception:
                pass

        if not tba_match:
            continue

        alliances = tba_match.get("alliances") or {}
        sb = tba_match.get("score_breakdown") or {}
        winning_alliance = tba_match.get("winning_alliance") or ""

        for alliance_str in ["red", "blue"]:
            al_obj = alliances.get(alliance_str) or {}
            team_keys = al_obj.get("team_keys") or []
            score = int(al_obj.get("score") or 0)
            al_sb = sb.get(alliance_str) or {}
            hub_score = al_sb.get("hubScore") or {}

            bot_col = "red_robot_clusters" if alliance_str == "red" else "blue_robot_clusters"
            inc_col = "territory_incursion_red" if alliance_str == "red" else "territory_incursion_blue"
            x_col = "red_mean_norm_x" if alliance_str == "red" else "blue_mean_norm_x"

            for slot_idx, team_key in enumerate(team_keys, start=1):
                rows.append({
                    "team_key": team_key,
                    "team_number": team_key.replace("frc", ""),
                    "year": int(year),
                    "split": split,
                    "event_key": event_key,
                    "match_key": match_key,
                    "alliance": alliance_str,
                    "station_slot": slot_idx,
                    "alliance_score": score,
                    "alliance_won": winning_alliance == alliance_str,
                    "tba_auto_points": int(al_sb.get("totalAutoPoints") or 0),
                    "tba_teleop_points": int(al_sb.get("totalTeleopPoints") or 0),
                    "tba_hub_total_fuel": int(hub_score.get("totalCount") or 0),
                    "tba_auto_tower_robot": str(al_sb.get(f"autoTowerRobot{slot_idx}", "None")),
                    "tba_endgame_tower_robot": str(al_sb.get(f"endGameTowerRobot{slot_idx}", "None")),
                    "vision_alliance_mean_bots_detected": round(float(grp[bot_col].mean()), 2),
                    "vision_alliance_mean_x": round(float(grp[x_col].mean()), 4),
                    "vision_alliance_incursions": int(grp[inc_col].sum()),
                    "vision_match_defense_engagements": int(grp["defense_engagements"].sum()),
                    "vision_match_mean_fuel_on_carpet": round(float(grp["raw_fuel_count"].mean()), 2),
                    "vision_match_mean_conf": round(float(grp["mean_bumper_conf"].mean()), 4),
                    "venue_brightness": round(float(grp["brightness_mean"].mean()), 2),
                    "venue_glare_ratio": round(float(grp["glare_ratio"].mean()), 5),
                })

    df_all_robots = pd.DataFrame(rows)
    if not df_all_robots.empty:
        for target_dir in [repo_data_dir, ssd_data_dir]:
            df_all_robots.to_csv(target_dir / "all_matches_every_robot_catalog.csv", index=False)
            df_all_robots.to_json(target_dir / "all_matches_every_robot_catalog.json", orient="records", indent=2)
        print(
            f"[Pipeline] Cataloged {len(df_all_robots)} robot-match appearances across "
            f"{df_all_robots['team_key'].nunique()} unique FRC teams in all extracted matches."
        )
    return df_all_robots


def backup_code_to_ssd(repo_root: Path, ssd_analytics_dir: Path) -> None:
    """Copies all project Python scripts, requirements, and documentation to Extreme SSD."""
    code_backup_dir = ssd_analytics_dir / "code_snapshot"
    code_backup_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "evaluator.py",
        "robot_identifier.py",
        "frc_diagnostics.py",
        "run_vision_pipeline.py",
        "config.py",
        "tba_client.py",
        "downloader.py",
        "frame_extractor.py",
        "sampler.py",
        "dashboard.py",
        "main.py",
        "requirements.txt",
        "README.md",
    ]
    copied = 0
    for fname in files_to_copy:
        src = repo_root / fname
        if src.is_file():
            shutil.copy2(src, code_backup_dir / fname)
            copied += 1
    print(f"[Pipeline] Backed up {copied} source code & documentation files to {code_backup_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Master FRC Vision Evaluation, Robot ID & Diagnostics Pipeline.")
    parser.add_argument("--model", type=str, default="models/frc_yolo11s_best.pt")
    parser.add_argument("--ssd-root", type=str, default="./data")
    parser.add_argument("--repo-out", type=str, default="analytics_outputs")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=15,
        help="Number of full match videos to track & render annotated MP4s in this batch (0 = all downloaded videos)",
    )
    parser.add_argument("--video-stride", type=int, default=5, help="Frame stride for full-video tracking")
    parser.add_argument("--skip-frames-eval", action="store_true", help="Reuse existing frame evaluation CSV")
    parser.add_argument("--skip-video-tracking", action="store_true", help="Reuse existing video tracking CSV")
    args = parser.parse_args()

    t_start = time.time()
    repo_root = Path.cwd()
    ssd_root = Path(args.ssd_root)
    ssd_analytics_dir = ssd_root / "frc_vision_analytics"
    repo_out_dir = Path(args.repo_out)

    # Step 1: Zero-Label Frame Evaluation across all extracted frames
    if not args.skip_frames_eval:
        print("\n=== STEP 1/4: Running Zero-Label Frame Accuracy & FRC Invariant Evaluation ===")
        evaluator = ZeroLabelEvaluator(
            model_path=Path(args.model),
            frames_root=ssd_root / "frc_extracted_frames",
            ssd_output_dir=ssd_analytics_dir,
            repo_output_dir=repo_out_dir,
            save_annotated_frames=True,
        )
        evaluator.evaluate_all_frames(batch_size=16)

    # Step 2: Closed-Set TBA Robot Identification & Annotated Video Rendering
    if not args.skip_video_tracking:
        print("\n=== STEP 2/4: Running Closed-Set TBA Robot Identification & Annotated Video Rendering ===")
        identifier = MatchVideoIdentifier(
            model_path=Path(args.model),
            tba_cache_dir=Path(".cache/tba"),
            ssd_output_dir=ssd_analytics_dir,
            repo_output_dir=repo_out_dir,
        )
        identifier.run_batch(
            videos_root=ssd_root,
            max_videos=args.max_videos,
            frame_stride=args.video_stride,
            render_videos=True,
        )

    # Step 3: Generate Charts, ML Role Clusters, All-Match Robot Catalog & Reports
    print("\n=== STEP 3/4: Generating Charts, ML Role Clusters, Robot Profiles & Markdown Reports ===")
    suite = FRCDiagnosticsSuite(
        ssd_analytics_dir=ssd_analytics_dir,
        repo_output_dir=repo_out_dir,
        tba_cache_dir=Path(".cache/tba"),
    )
    df_frames, df_boxes, df_robots, df_traj = suite.load_datasets()
    build_all_matches_team_appearances(
        df_frames=df_frames,
        tba_cache_dir=Path(".cache/tba"),
        repo_data_dir=suite.repo_data_dir,
        ssd_data_dir=suite.ssd_data_dir,
    )
    df_matches = suite.enrich_with_tba_match_data(df_frames)
    stats = suite.generate_all_charts(df_frames, df_boxes, df_robots, df_traj, df_matches)
    suite.write_markdown_report(df_frames, df_boxes, df_robots, df_matches, stats)

    # Step 4: Backup all source code to Extreme SSD
    print("\n=== STEP 4/4: Saving Code Snapshot & Syncing Artifacts to Extreme SSD ===")
    backup_code_to_ssd(repo_root, ssd_analytics_dir)

    elapsed = time.time() - t_start
    print(f"\n[Pipeline Complete] Finished in {elapsed:.1f}s!")
    print(f" -> GitHub Analytics Output: {repo_out_dir.resolve()}")
    print(f" -> Extreme SSD Output (Annotated Videos, Frames, Charts, Data, Code): {ssd_analytics_dir}")


if __name__ == "__main__":
    main()
