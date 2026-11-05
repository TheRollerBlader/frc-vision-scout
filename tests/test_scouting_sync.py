import json
import tempfile
from pathlib import Path
import pytest
from scouting_sync_engine import (
    ScoutingSyncEngine,
    StatboticsEPA,
    MatchScoutingRecord,
)


class TestScoutingSyncEngine:
    def test_parse_statbotics_payload(self):
        engine = ScoutingSyncEngine(enable_cache=False)
        mock_payload = {
            "team": 254,
            "year": 2026,
            "epa": {
                "breakdown": {
                    "total_points": 48.5,
                    "auto_points": 14.0,
                    "teleop_points": 26.5,
                    "endgame_points": 8.0,
                }
            },
            "record": {"winrate": 0.89},
        }

        parsed = engine.parse_statbotics_payload(mock_payload)
        assert parsed is not None
        assert parsed.team_number == 254
        assert parsed.year == 2026
        assert parsed.epa_total == 48.5
        assert parsed.epa_auto == 14.0
        assert parsed.winrate == 0.89

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ScoutingSyncEngine(cache_dir=Path(tmpdir), enable_cache=True)
            mock_data = {"team": 1678, "year": 2026, "epa": {"breakdown": {"total_points": 45.2}}}
            engine._write_cache("team_1678", mock_data)

            cached = engine._read_cache("team_1678")
            assert cached == mock_data

    def test_correlate_match_vision_with_tba_and_statbotics(self):
        engine = ScoutingSyncEngine(enable_cache=False)
        mock_tba_match = {
            "key": "2026txirv_qm12",
            "alliances": {
                "blue": {"score": 94, "team_keys": ["frc254", "frc1678", "frc5411"]},
                "red": {"score": 82, "team_keys": ["frc118", "frc148", "frc3310"]},
            },
        }

        vision_detections = {
            254: {"auto_cycles": 3, "teleop_cycles": 8, "confidence": 0.98},
            5411: {"auto_cycles": 1, "teleop_cycles": 5, "confidence": 0.92},
            118: {"auto_cycles": 2, "teleop_cycles": 6, "confidence": 0.95},
        }

        statbotics_cache = {
            254: StatboticsEPA(254, 2026, 48.5, 14.0, 26.5, 8.0, 0.89),
            5411: StatboticsEPA(5411, 2026, 31.2, 8.0, 18.2, 5.0, 0.65),
            118: StatboticsEPA(118, 2026, 42.0, 12.0, 24.0, 6.0, 0.82),
        }

        records = engine.correlate_match(
            match_key="2026txirv_qm12",
            tba_match_data=mock_tba_match,
            vision_detected_cycles=vision_detections,
            statbotics_cache=statbotics_cache,
        )

        assert len(records) == 3

        # Check 254 (blue alliance)
        rec_254 = next(r for r in records if r.team_number == 254)
        assert rec_254.alliance == "blue"
        assert rec_254.official_alliance_score == 94
        assert rec_254.detected_auto_cycles == 3
        assert rec_254.detected_teleop_cycles == 8
        assert rec_254.statbotics_epa == 48.5

        # Check 118 (red alliance)
        rec_118 = next(r for r in records if r.team_number == 118)
        assert rec_118.alliance == "red"
        assert rec_118.official_alliance_score == 82
        assert rec_118.statbotics_epa == 42.0

    def test_export_records_csv(self):
        engine = ScoutingSyncEngine(enable_cache=False)
        record = MatchScoutingRecord(
            match_key="2026txirv_qm1",
            team_number=5411,
            alliance="blue",
            detected_auto_cycles=2,
            detected_teleop_cycles=6,
            estimated_scoring_points=31.0,
            vision_tracking_confidence=0.94,
            official_alliance_score=85,
            statbotics_epa=32.5,
        )
        csv_output = engine.export_records_csv([record])
        assert "match_key,team_number" in csv_output
        assert "2026txirv_qm1,5411,blue,2,6,31.0,0.94,85,32.5" in csv_output
