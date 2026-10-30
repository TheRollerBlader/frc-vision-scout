import os
import tempfile
from pathlib import Path
import pytest
from config import AppConfig, load_dotenv, is_writable, detect_gdrive_directory


class TestConfig:
    def test_load_dotenv_creates_env_vars(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as f:
            f.write("TEST_FRC_VAR=high_leverage_token\n")
            f.write("# Comment line\n")
            f.write("ANOTHER_VAR='quoted_value'\n")
            temp_path = Path(f.name)

        try:
            load_dotenv(temp_path)
            assert os.environ.get("TEST_FRC_VAR") == "high_leverage_token"
            assert os.environ.get("ANOTHER_VAR") == "quoted_value"
        finally:
            temp_path.unlink()
            os.environ.pop("TEST_FRC_VAR", None)
            os.environ.pop("ANOTHER_VAR", None)

    def test_is_writable_on_valid_tempdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert is_writable(Path(tmpdir)) is True

    def test_app_config_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AppConfig(
                tba_api_key="mock_key_123",
                gdrive_dir=Path(tmpdir),
                cache_dir=Path(tmpdir) / ".cache",
                multi_years=[2025, 2026],
            )
            assert cfg.tba_api_key == "mock_key_123"
            assert cfg.current_year == 2026
            assert 2025 in cfg.multi_years
            assert cfg.video_quality == "best"
            assert cfg.dry_run is False
