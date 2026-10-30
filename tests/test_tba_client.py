import json
import tempfile
from pathlib import Path
import pytest
from tba_client import TBAClient


class TestTBAClient:
    def test_cache_file_generation_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir)
            client = TBAClient(api_key="mock_key", cache_dir=cache_path, cache_enabled=True)

            endpoint = "event/2026txirv/matches"
            mock_data = [{"key": "2026txirv_qm1", "alliances": {"blue": {"score": 42}}}]

            # Write cache
            client._write_cache(endpoint, mock_data)

            # Read back cache
            cached = client._read_cache(endpoint)
            assert cached == mock_data

    def test_missing_api_key_raises_value_error(self):
        client = TBAClient(api_key="", cache_enabled=False)
        with pytest.raises(ValueError, match="TBA API key is missing"):
            client.request("events/2026")

    def test_disabled_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = TBAClient(api_key="mock_key", cache_dir=Path(tmpdir), cache_enabled=False)
            assert client._read_cache("any_endpoint") is None
