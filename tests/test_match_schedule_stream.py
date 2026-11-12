import pytest
from match_schedule_stream import MatchScheduleStream, ScheduledMatchJob


class TestMatchScheduleStream:
    def test_process_tba_matches_roster_extraction(self):
        stream = MatchScheduleStream(event_key="2026txirv")

        sample_matches = [
            {
                "key": "2026txirv_qm1",
                "comp_level": "qm",
                "match_number": 1,
                "time": 1742000000,
                "alliances": {
                    "red": {"team_keys": ["frc254", "frc1678", "frc5411"]},
                    "blue": {"team_keys": ["frc118", "frc148", "frc3310"]},
                },
            }
        ]

        epa_data = {
            254: {"total": 48.0, "auto": 14.0, "teleop": 26.0},
            1678: {"total": 45.0, "auto": 12.0, "teleop": 25.0},
            5411: {"total": 32.0, "auto": 8.0, "teleop": 18.0},
            118: {"total": 42.0, "auto": 11.0, "teleop": 24.0},
            148: {"total": 38.0, "auto": 10.0, "teleop": 20.0},
            3310: {"total": 35.0, "auto": 9.0, "teleop": 19.0},
        }

        jobs = stream.process_tba_matches(sample_matches, team_epa_map=epa_data)
        assert len(jobs) == 1

        job = jobs[0]
        assert job.match_key == "2026txirv_qm1"
        assert job.red_teams == [254, 1678, 5411]
        assert job.blue_teams == [118, 148, 3310]
        assert job.predicted_red_score == 48.0 + 45.0 + 32.0  # 125.0
        assert job.predicted_blue_score == 42.0 + 38.0 + 35.0  # 115.0
        assert job.win_probability_red > 0.5
        assert job.vision_ready is True
        assert len(job.roster) == 6

        # Check team 254 assignment
        team_254 = next(r for r in job.roster if r.team_number == 254)
        assert team_254.alliance == "red"
        assert team_254.station == 1
        assert team_254.epa_total == 48.0

    def test_unscouted_queue_filtering(self):
        stream = MatchScheduleStream(event_key="2026txirv")
        matches = [
            {
                "key": "2026txirv_qm1",
                "comp_level": "qm",
                "match_number": 1,
                "alliances": {
                    "red": {"team_keys": ["frc254", "frc1678", "frc5411"]},
                    "blue": {"team_keys": ["frc118", "frc148", "frc3310"]},
                },
            },
            {
                "key": "2026txirv_qm2",
                "comp_level": "qm",
                "match_number": 2,
                "alliances": {
                    "red": {"team_keys": ["frc971", "frc1323", "frc4414"]},
                    "blue": {"team_keys": ["frc2910", "frc2056", "frc1114"]},
                },
            },
        ]
        stream.process_tba_matches(matches)

        # qm1 already scouted
        unscouted = stream.get_upcoming_unscouted_matches(scouted_keys=["2026txirv_qm1"])
        assert len(unscouted) == 1
        assert unscouted[0].match_key == "2026txirv_qm2"
