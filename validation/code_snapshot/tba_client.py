"""The Blue Alliance (TBA) API v3 Client with disk caching and rate limiting."""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class TBAClient:
    """Client for The Blue Alliance API v3."""

    BASE_URL = "https://www.thebluealliance.com/api/v3"

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None, cache_enabled: bool = True):
        self.api_key = api_key.strip()
        self.cache_dir = cache_dir or Path(".cache/tba")
        self.cache_enabled = cache_enabled
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, endpoint: str) -> Path:
        sanitized = endpoint.strip("/").replace("/", "_")
        if len(sanitized) > 100:
            h = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
            sanitized = f"{sanitized[:80]}_{h}"
        return self.cache_dir / f"{sanitized}.json"

    def _read_cache(self, endpoint: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
        cache_file = self._get_cache_path(endpoint)
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _write_cache(self, endpoint: str, data: Any) -> None:
        if not self.cache_enabled:
            return
        cache_file = self._get_cache_path(endpoint)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Warning] Failed to write cache for {endpoint}: {e}", file=sys.stderr)

    def request(self, endpoint: str, max_retries: int = 3) -> Any:
        """Makes an authenticated GET request to TBA API v3."""
        cached = self._read_cache(endpoint)
        if cached is not None:
            return cached

        if not self.api_key:
            raise ValueError(
                "TBA API key is missing. Set TBA_AUTH_KEY in your environment or .env, "
                "or pass --api-key <YOUR_KEY>."
            )

        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers = {
            "X-TBA-Auth-Key": self.api_key,
            "User-Agent": "FRC-Vision-VideoDownloader/1.0 (urllib)",
            "Accept": "application/json",
        }

        req = urllib.request.Request(url, headers=headers)

        last_error = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if resp.status == 200:
                        content = resp.read().decode("utf-8")
                        data = json.loads(content)
                        self._write_cache(endpoint, data)
                        return data
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 401:
                    raise PermissionError(
                        "TBA API returned 401 Unauthorized. Please verify your TBA API key."
                    )
                elif e.code == 404:
                    return None
                elif e.code == 429:
                    wait_time = (attempt + 1) * 3
                    print(f"[TBA] Rate limited (429). Retrying in {wait_time}s...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    time.sleep(1)
            except Exception as e:
                last_error = e
                time.sleep(1)

        raise RuntimeError(f"TBA API request to '{endpoint}' failed after {max_retries} attempts: {last_error}")

    def get_status(self) -> Dict[str, Any]:
        """Gets current season and API status."""
        data = self.request("status")
        return data or {}

    def get_events(self, year: int, official_only: bool = True) -> List[Dict[str, Any]]:
        """Fetches events for a given season year."""
        data = self.request(f"events/{year}")
        if not data or not isinstance(data, list):
            return []

        if not official_only:
            return data

        # Official event types:
        # 0: Regional, 1: District, 2: District CMP, 3: CMP Division, 4: CMP Finals,
        # 5: District Division, 6: Festival of Champions
        # Offseason (99), Preseason (100) are typically excluded by official_only
        official_types = {0, 1, 2, 3, 4, 5, 6}
        return [e for e in data if e.get("event_type") in official_types]

    def get_event_matches(self, event_key: str) -> List[Dict[str, Any]]:
        """Fetches all matches for an event."""
        data = self.request(f"event/{event_key}/matches")
        if not data or not isinstance(data, list):
            return []
        return data

    @staticmethod
    def extract_youtube_videos(match: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extracts list of YouTube video dictionaries: [{'id': ..., 'url': ...}]"""
        videos = match.get("videos") or []
        res = []
        for v in videos:
            if v.get("type") == "youtube" and v.get("key"):
                yt_id = v["key"].strip()
                res.append({
                    "id": yt_id,
                    "url": f"https://www.youtube.com/watch?v={yt_id}"
                })
        return res

    @staticmethod
    def format_location(event: Dict[str, Any]) -> str:
        """Extracts a normalized, readable geographical location from an event."""
        state_prov = (event.get("state_prov") or "").strip()
        country = (event.get("country") or "").strip()
        city = (event.get("city") or "").strip()

        if state_prov and country:
            return f"{state_prov}, {country}"
        elif country:
            return country
        elif state_prov:
            return state_prov
        elif city:
            return city
        return "Unknown Location"
