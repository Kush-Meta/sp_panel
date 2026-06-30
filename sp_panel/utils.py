"""Shared HTTP plumbing: a token-bucket rate limiter and a configured session
for SEC requests (correct User-Agent, retries with exponential backoff)."""
import time
import threading
import requests

from . import config


class RateLimiter:
    """Simple thread-safe limiter: at most `rps` calls per second."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = self.min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


_sec_limiter = RateLimiter(config.SEC_MAX_RPS)


def sec_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": config.USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    })
    return s


def sec_get_json(session: requests.Session, url: str):
    """GET a SEC JSON endpoint with rate limiting + retries.

    Returns parsed JSON, or None on a definitive 404 (e.g. no companyfacts).
    Raises on repeated hard failures so the caller can log and move on.
    """
    last_err = None
    for attempt in range(config.SEC_RETRIES):
        _sec_limiter.wait()
        try:
            # Host header must match the URL's host.
            host = url.split("/")[2]
            resp = session.get(url, headers={"Host": host}, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 429, 500, 502, 503):
                last_err = f"HTTP {resp.status_code}"
                time.sleep(config.SEC_BACKOFF * (2 ** attempt))
                continue
            resp.raise_for_status()
        except (requests.RequestException, ValueError) as e:
            last_err = str(e)
            time.sleep(config.SEC_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"SEC request failed after {config.SEC_RETRIES} tries: {url} ({last_err})")
