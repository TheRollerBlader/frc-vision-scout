"""Resilient HTTP and JSON ingestion stream client for frc-vision-scout.
Scheduled Milestone: Integration & Data Ingest (2026-11-12).
"""

import time
from typing import Dict, Any, Optional
import urllib.request
import json


class ResilientStreamClient:
    """Client for polling live streaming endpoints with exponential backoff and timeout boundaries."""
    def __init__(self, endpoint_url: str, timeout_sec: float = 5.0, max_retries: int = 3):
        self.endpoint_url = endpoint_url
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def fetch_payload(self) -> Optional[Dict[str, Any]]:
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(self.endpoint_url, headers={"User-Agent": "frc-vision-scout-stream/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode("utf-8"))
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
        return None
