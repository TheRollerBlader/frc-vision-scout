"""Streaming Match Schedule Ingestion and Pre-Match EPA Prediction Engine.

Monitors The Blue Alliance (TBA) event feeds, extracts six-robot match
rosters, correlates historical Statbotics EPA metrics, and auto-dispatches
vision tracking job manifests for FRC scouting analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time


@dataclass
class RobotMatchAssignment:
    team_number: int
    alliance: str  # 'red' or 'blue'
    station: int  # 1, 2, or 3
    bumper_color: str
    epa_total: float = 0.0
    epa_auto: float = 0.0
    epa_teleop: float = 0.0


@dataclass
class ScheduledMatchJob:
    event_key: str
    match_key: str
    match_number: int
    comp_level: str  # 'qm', 'qf', 'sf', 'f'
    scheduled_time: Optional[int]
    red_teams: List[int]
    blue_teams: List[int]
    predicted_red_score: float = 0.0
    predicted_blue_score: float = 0.0
    win_probability_red: float = 0.5
    vision_ready: bool = False
    roster: List[RobotMatchAssignment] = field(default_factory=list)


class MatchScheduleStream:
    """Ingests raw TBA match arrays and generates calibrated vision tracking queue jobs."""

    def __init__(self, event_key: str):
        self.event_key = event_key
        self.cached_jobs: Dict[str, ScheduledMatchJob] = {}

    def process_tba_matches(
        self,
        tba_matches: List[Dict[str, Any]],
        team_epa_map: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> List[ScheduledMatchJob]:
        team_epa_map = team_epa_map or {}
        jobs: List[ScheduledMatchJob] = []

        for m in tba_matches:
            key = m.get("key", "")
            if not key:
                continue

            alliances = m.get("alliances", {})
            red_alliance = alliances.get("red", {})
            blue_alliance = alliances.get("blue", {})

            red_teams = [
                int(k.replace("frc", ""))
                for k in red_alliance.get("team_keys", [])
                if k.replace("frc", "").isdigit()
            ]
            blue_teams = [
                int(k.replace("frc", ""))
                for k in blue_alliance.get("team_keys", [])
                if k.replace("frc", "").isdigit()
            ]

            roster: List[RobotMatchAssignment] = []
            red_predicted = 0.0
            blue_predicted = 0.0

            for idx, team in enumerate(red_teams, 1):
                epa = team_epa_map.get(team, {})
                tot = epa.get("total", 25.0)
                red_predicted += tot
                roster.append(
                    RobotMatchAssignment(
                        team_number=team,
                        alliance="red",
                        station=idx,
                        bumper_color="red",
                        epa_total=tot,
                        epa_auto=epa.get("auto", 6.0),
                        epa_teleop=epa.get("teleop", 15.0),
                    )
                )

            for idx, team in enumerate(blue_teams, 1):
                epa = team_epa_map.get(team, {})
                tot = epa.get("total", 25.0)
                blue_predicted += tot
                roster.append(
                    RobotMatchAssignment(
                        team_number=team,
                        alliance="blue",
                        station=idx,
                        bumper_color="blue",
                        epa_total=tot,
                        epa_auto=epa.get("auto", 6.0),
                        epa_teleop=epa.get("teleop", 15.0),
                    )
                )

            # Log-odds win probability approximation
            diff = red_predicted - blue_predicted
            win_prob_red = 1.0 / (1.0 + 10.0 ** (-diff / 25.0))

            job = ScheduledMatchJob(
                event_key=self.event_key,
                match_key=key,
                match_number=m.get("match_number", 0),
                comp_level=m.get("comp_level", "qm"),
                scheduled_time=m.get("time"),
                red_teams=red_teams,
                blue_teams=blue_teams,
                predicted_red_score=round(red_predicted, 1),
                predicted_blue_score=round(blue_predicted, 1),
                win_probability_red=round(win_prob_red, 3),
                vision_ready=(len(red_teams) == 3 and len(blue_teams) == 3),
                roster=roster,
            )

            self.cached_jobs[key] = job
            jobs.append(job)

        jobs.sort(key=lambda j: (j.comp_level != "qm", j.match_number))
        return jobs

    def get_upcoming_unscouted_matches(self, scouted_keys: List[str]) -> List[ScheduledMatchJob]:
        scouted_set = set(scouted_keys)
        return [
            job
            for job in self.cached_jobs.values()
            if job.match_key not in scouted_set and job.vision_ready
        ]
