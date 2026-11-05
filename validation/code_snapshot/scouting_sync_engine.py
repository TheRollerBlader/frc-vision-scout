"""FRC Vision Scout Ecosystem Sync Engine.

Integrates The Blue Alliance (TBA) and Statbotics EPA (Expected Points Added)
analytics with vision-scouted robot trajectories, bumper tracking, and cycle counts.
Generates correlated scouting records and exportable JSON/CSV payloads.
"""

from __future__ import annotations
import csv
import io
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StatboticsEPA:
    team_number: int
    year: int
    epa_total: float
    epa_auto: float
    epa_teleop: float
    epa_endgame: float
    winrate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchScoutingRecord:
    match_key: str
    team_number: int
    alliance: str  # 'red' or 'blue'
    detected_auto_cycles: int
    detected_teleop_cycles: int
    estimated_scoring_points: float
    vision_tracking_confidence: float
    official_alliance_score: Optional[int] = None
    statbotics_epa: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScoutingSyncEngine:
    """Connects computer vision match detections with external FRC data sources."""

    STATBOTICS_BASE = "https://api.statbotics.io/v3"

    def __init__(self, cache_dir: Optional[Path] = None, enable_cache: bool = True):
        self.cache_dir = cache_dir or Path(".cache/scouting_sync")
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)
        return self.cache_dir / f"{sanitized}.json"

    def _read_cache(self, key: str) -> Optional[Any]:
        if not self.enable_cache:
            return None
        cp = self._get_cache_path(key)
        if cp.is_file():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, data: Any) -> None:
        if not self.enable_cache:
            return
        cp = self._get_cache_path(key)
        try:
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def parse_statbotics_payload(self, data: Dict[str, Any]) -> Optional[StatboticsEPA]:
        """Parses raw Statbotics team_year JSON response into a typed StatboticsEPA."""
        if not data or not isinstance(data, dict):
            return None
        try:
            epa_dict = data.get("epa", {}) or {}
            record_dict = data.get("record", {}) or {}
            return StatboticsEPA(
                team_number=int(data.get("team", 0)),
                year=int(data.get("year", 0)),
                epa_total=float(epa_dict.get("breakdown", {}).get("total_points", epa_dict.get("unitless", 0.0))),
                epa_auto=float(epa_dict.get("breakdown", {}).get("auto_points", 0.0)),
                epa_teleop=float(epa_dict.get("breakdown", {}).get("teleop_points", 0.0)),
                epa_endgame=float(epa_dict.get("breakdown", {}).get("endgame_points", 0.0)),
                winrate=float(record_dict.get("winrate", 0.0)),
            )
        except (ValueError, TypeError, KeyError):
            return None

    def fetch_team_epa(self, team: int, year: int) -> Optional[StatboticsEPA]:
        """Fetches EPA analytics for a given team and season with local caching."""
        cache_key = f"statbotics_team_{team}_{year}"
        cached = self._read_cache(cache_key)
        if cached:
            return self.parse_statbotics_payload(cached)

        url = f"{self.STATBOTICS_BASE}/team_year/{team}/{year}"
        req = urllib.request.Request(url, headers={"User-Agent": "FRCVisionScout/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self._write_cache(cache_key, data)
                    return self.parse_statbotics_payload(data)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None

        return None

    def correlate_match(
        self,
        match_key: str,
        tba_match_data: Optional[Dict[str, Any]],
        vision_detected_cycles: Dict[int, Dict[str, Any]],
        statbotics_cache: Optional[Dict[int, StatboticsEPA]] = None,
    ) -> List[MatchScoutingRecord]:
        """Joins vision tracking observations with official TBA match alliances and Statbotics EPA."""
        records: List[MatchScoutingRecord] = []
        alliances = tba_match_data.get("alliances", {}) if tba_match_data else {}

        # Build team alliance map
        team_alliance_map: Dict[int, str] = {}
        team_score_map: Dict[int, Optional[int]] = {}

        for color in ("red", "blue"):
            alliance_info = alliances.get(color, {})
            score = alliance_info.get("score")
            team_keys = alliance_info.get("team_keys", [])
            for tk in team_keys:
                try:
                    num = int(tk.replace("frc", ""))
                    team_alliance_map[num] = color
                    team_score_map[num] = score
                except ValueError:
                    continue

        for team_num, det in vision_detected_cycles.items():
            alliance = team_alliance_map.get(team_num, det.get("alliance", "unknown"))
            official_score = team_score_map.get(team_num)
            auto_cycles = int(det.get("auto_cycles", 0))
            teleop_cycles = int(det.get("teleop_cycles", 0))
            # Typical FRC coral / note scoring value ~3-5 pts
            est_points = float(auto_cycles * 5.0 + teleop_cycles * 3.5)
            confidence = float(det.get("confidence", 0.90))

            epa_val = None
            if statbotics_cache and team_num in statbotics_cache:
                epa_val = statbotics_cache[team_num].epa_total

            records.append(
                MatchScoutingRecord(
                    match_key=match_key,
                    team_number=team_num,
                    alliance=alliance,
                    detected_auto_cycles=auto_cycles,
                    detected_teleop_cycles=teleop_cycles,
                    estimated_scoring_points=round(est_points, 1),
                    vision_tracking_confidence=round(confidence, 3),
                    official_alliance_score=official_score,
                    statbotics_epa=round(epa_val, 2) if epa_val is not None else None,
                )
            )

        return sorted(records, key=lambda r: (r.alliance, r.team_number))

    def export_records_csv(self, records: List[MatchScoutingRecord]) -> str:
        """Exports scouting records to standard RFC 4180 CSV."""
        output = io.StringIO()
        fieldnames = [
            "match_key",
            "team_number",
            "alliance",
            "detected_auto_cycles",
            "detected_teleop_cycles",
            "estimated_scoring_points",
            "vision_tracking_confidence",
            "official_alliance_score",
            "statbotics_epa",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())
        return output.getvalue()
