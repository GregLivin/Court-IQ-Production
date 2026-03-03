from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class NBAHttpConfig:
    base_url: str = "https://stats.nba.com/stats"
    timeout_s: int = 90          # was 30
    sleep_s: float = 1.25        # gentle spacing between calls
    retries: int = 4             # NEW
    backoff_s: float = 2.0       # NEW


def default_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }


class NBAHttpClient:
    def __init__(self, cfg: Optional[NBAHttpConfig] = None):
        self.cfg = cfg or NBAHttpConfig()
        self.session = requests.Session()
        self.session.headers.update(default_headers())

    def get_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}/{endpoint.lstrip('/')}"

        # retry loop for timeouts/transient failures
        for attempt in range(1, self.cfg.retries + 1):
            time.sleep(self.cfg.sleep_s)
            try:
                r = self.session.get(url, params=params, timeout=self.cfg.timeout_s)

                if r.status_code in (403, 429):
                    raise RuntimeError(
                        f"NBA stats blocked the request (status={r.status_code}). "
                        f"Try increasing sleep_s, and ensure headers are intact (no VPN)."
                    )

                r.raise_for_status()
                return r.json()

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                if attempt >= self.cfg.retries:
                    raise
                # exponential-ish backoff
                wait = self.cfg.backoff_s * attempt
                print(f"[NBAHttp] Attempt {attempt} failed ({type(e).__name__}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
