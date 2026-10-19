#!/usr/bin/env python3
"""
frc_cad_downloader.py - Download real STEP files for every robot in the
Spectrum 3847 "FRC CAD Collection" (https://spectrum3847.org/resources/frc-cad-collection/)
that has CAD on Onshape or GrabCAD, and write JSON metadata for each robot.

Requirements:  Python 3.8+,  pip install requests beautifulsoup4
Optional:      pip install playwright   (only for the --login helper; it can
               use your installed Google Chrome, or run `playwright install chromium`)

------------------------------------------------------------------------------
WHERE THE DATA COMES FROM
------------------------------------------------------------------------------
The page itself is an Astro site. The robot list is NOT an HTML table: it is
(a) embedded as JSON in <script type="application/json" id="cad-entries-snapshot">
and (b) refreshed by the page's JS from a Google Apps Script endpoint
(".../macros/s/<id>/exec?action=entries"), which returns
{"success": true, "entries": [{rowIndex, team, year, teamName, description,
format, links[]}]}. This script finds that endpoint in the page's
CadCollectionBrowser JS bundle and uses it, falling back to the embedded
snapshot (--data-source auto|live|snapshot).

------------------------------------------------------------------------------
ONSHAPE  (needs a logged-in Onshape session - a FREE account is fine)
------------------------------------------------------------------------------
Reading public/link-shared documents (name, elements) works anonymously, but
Onshape refuses every export/translation call without authentication
("401 Unauthenticated API request"), even for public documents. Choose one:

 1. Browser session (recommended, no API keys needed). Either:
    a) python3 frc_cad_downloader.py --login onshape
       -> a Chrome window opens, you sign in to Onshape normally, the script
          saves the session cookies to ~/.frc_cad/onshape_cookies.json and
          reuses them on later runs automatically. (Needs `pip install playwright`.)
    b) or copy the Cookie header yourself: sign in at https://cad.onshape.com,
       open DevTools > Network, click any request to cad.onshape.com/api/...,
       copy the full "Cookie:" request header value, and either
         export ONSHAPE_COOKIE='on=...; XSRF-TOKEN=...; ...'
       or save it to a file and pass --onshape-cookie-file FILE
       (raw header text, Netscape cookies.txt and JSON cookie exports all work).
    The session must contain the "on" and "XSRF-TOKEN" cookies (the script
    sends X-XSRF-TOKEN automatically). Sessions expire; re-run --login if you
    see "onshape session invalid/expired".
 2. Optional: API keys, if you ever have them:
       export ONSHAPE_ACCESS_KEY=...  ONSHAPE_SECRET_KEY=...
    (sent as HTTP Basic auth). Note: API keys on free/EDU plans are capped
    at 2,500 calls/year; each robot costs roughly 3-15 calls.

Export flow: POST /api/v10/{assemblies|partstudios}/d/{did}/{w|v|m}/{id}/e/{eid}/translations
{"formatName":"STEP","storeInDocument":false}  ->  poll GET /api/v10/translations/{tid}
until DONE  ->  GET /api/v10/documents/d/{did}/externaldata/{fid}.
Element choice: the element in the link if it is an assembly/part studio,
otherwise the best-named assembly ("robot", "full", "comp", ...; mechanism
names like "intake"/"swerve" are de-prioritised), otherwise a part studio.
--onshape-all-assemblies exports every assembly in the document instead.
If the owner disabled export for non-owners you get a 403;
--onshape-copy-fallback then copies the document into YOUR account and
exports from the copy (the copy is left in your Onshape "My documents").

------------------------------------------------------------------------------
GRABCAD
------------------------------------------------------------------------------
Model info and file lists come from GrabCAD's JSON API
(/community/api/v1/models/<slug> and .../files?folder_id=N), which works
without login. The official per-file download endpoint
(.../files/download?cadid=...) returns 401 "You need to log in" unless you are
signed in; when signed in it returns JSON {"url": ".../cads/files/<cadid>/original.<ext>"}
(the 87-byte JSON "placeholder" files you may have seen) and the script then
follows that URL to the real file. Provide a session with:
    python3 frc_cad_downloader.py --login grabcad          (Playwright helper), or
    export GRABCAD_COOKIE='...'   /   --grabcad-cookie-file FILE
--grabcad-direct (opt-in) skips the login step and fetches the
.../cads/files/<cadid>/original.<ext> file URL directly. In testing that URL
served the real file without a session, but GrabCAD's site asks users to log
in to download, so use it at your own discretion.
STEP/STP files are kept; .zip files are downloaded (by default only when the
model has no loose STEP file) and their STEP members extracted.

------------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------------
<out>/robots.json                         master index (all Onshape/GrabCAD robots)
<out>/<year>/<team#>_<TeamName>/metadata.json
<out>/<year>/<team#>_<TeamName>/onshape/<Document> - <Element>.step
<out>/<year>/<team#>_<TeamName>/grabcad/<slug>/<file>.stp
<out>/frc_cad_downloader.log
Every downloaded file is validated: it must start with "ISO-10303-21" (or be a
zip that contains such files). HTML, JSON placeholders etc. are rejected.
Re-running resumes: files that already exist and validate are skipped.

Examples:
  python3 frc_cad_downloader.py --dry-run                     # index only
  python3 frc_cad_downloader.py --dry-run --resolve --year 2025 --limit 5
  python3 frc_cad_downloader.py --login onshape
  python3 frc_cad_downloader.py --year 2025 --source onshape
  python3 frc_cad_downloader.py --source grabcad --grabcad-cookie-file gc.txt
  python3 frc_cad_downloader.py --repair-placeholders ~/old_step_folder
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.stderr.write("This script needs: pip install requests beautifulsoup4\n")
    sys.exit(2)

__version__ = "1.1.0"

PAGE_URL = "https://spectrum3847.org/resources/frc-cad-collection/"
ONSHAPE_BASE = "https://cad.onshape.com"
ONSHAPE_API = "/api/v10"
GRABCAD_BASE = "https://grabcad.com"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/129.0 Safari/537.36 frc-cad-downloader/" + __version__)
# GrabCAD's CDN rejects requests without a browser-like User-Agent (CloudFront 403).

STEP_EXTS = {"step", "stp"}
ZIP_EXTS = {"zip"}
DEFAULT_STATE_DIR = Path.home() / ".frc_cad"

log = logging.getLogger("frc_cad")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name(s: str, maxlen: int = 80) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w.\- ]+", "", s).strip().replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("._")
    return (s[:maxlen].rstrip("._")) or "unnamed"


def safe_filename(s: str, maxlen: int = 150) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", s).strip(" .")
    if len(s) > maxlen:
        stem, dot, ext = s.rpartition(".")
        s = (stem[: maxlen - len(ext) - 1] + "." + ext) if dot and len(ext) < 8 else s[:maxlen]
    return s or "file"


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def sniff(head: bytes) -> str:
    """Classify the first bytes of a file: step | zip | json | html | empty | unknown."""
    h = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not h:
        return "empty"
    if h.startswith(b"ISO-10303-21"):
        return "step"
    if h.startswith(b"PK\x03\x04") or h.startswith(b"PK\x07\x08PK\x03\x04") or h.startswith(b"PK\x05\x06"):
        return "zip"  # incl. single-segment "spanned" marker PK0708
    if h[:1] in (b"{", b"["):
        return "json"
    low = h[:300].lower()
    if low.startswith(b"<!doctype") or b"<html" in low or low.startswith(b"<?xml"):
        return "html"
    return "unknown"


def file_kind(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return sniff(f.read(4096))
    except OSError:
        return "missing"


def is_valid_step(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 200 and file_kind(path) == "step"


def ext_of(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in (name or "") else ""


# ----------------------------------------------------------------------------
# cookies
# ----------------------------------------------------------------------------
def parse_cookie_text(text: str) -> Dict[str, str]:
    """Accept a raw Cookie header, a Netscape cookies.txt, or a JSON cookie export."""
    text = (text or "").strip()
    if not text:
        return {}
    if text[:1] in "[{":
        data = json.loads(text)
        if isinstance(data, dict) and "cookies" in data:  # Playwright storage_state
            data = data["cookies"]
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {c["name"]: c["value"] for c in data if "name" in c and "value" in c}
    lines = [l for l in text.splitlines()
             if l.strip() and (not l.startswith("#") or l.startswith("#HttpOnly_"))]
    if lines and all(len(l.split("\t")) >= 7 for l in lines):  # Netscape cookies.txt
        return {l.split("\t")[5]: l.split("\t")[6].strip() for l in lines}
    if text.lower().startswith("cookie:"):
        text = text[7:]
    out = {}
    for part in text.replace("\n", ";").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def load_cookies(env_var: str, file_arg: Optional[str], default_file: Path) -> Tuple[Dict[str, str], str]:
    if file_arg:
        p = Path(file_arg).expanduser()
        return parse_cookie_text(p.read_text(encoding="utf-8")), f"file:{p}"
    if os.environ.get(env_var):
        return parse_cookie_text(os.environ[env_var]), f"env:{env_var}"
    if default_file.is_file():
        return parse_cookie_text(default_file.read_text(encoding="utf-8")), f"file:{default_file}"
    return {}, ""


# ----------------------------------------------------------------------------
# HTTP with polite rate limiting + retries
# ----------------------------------------------------------------------------
class DownloadError(Exception):
    def __init__(self, msg, skipped=False, kind=None):
        super().__init__(msg)
        self.skipped = skipped
        self.kind = kind


class Http:
    def __init__(self, delay: float = 1.0, retries: int = 4, timeout: float = 60.0):
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.calls: Dict[str, int] = {}

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        with self._lock:
            wait = self._last.get(host, 0) + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()
            self.calls[host] = self.calls.get(host, 0) + 1

    def request(self, method: str, url: str, *, retry_on=(429, 500, 502, 503, 504),
                session: Optional[requests.Session] = None, **kw) -> requests.Response:
        """Send with retries/backoff. Any status not in retry_on is returned as-is."""
        s = session or self.session
        kw.setdefault("timeout", self.timeout)
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._throttle(url)
            try:
                r = s.request(method, url, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                log.debug("%s %s -> %s (attempt %d)", method, url, e, attempt + 1)
                if attempt < self.retries:
                    time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            if r.status_code not in retry_on or attempt == self.retries:
                return r
            ra = r.headers.get("Retry-After")
            log.debug("%s %s -> HTTP %s (attempt %d), retrying", method, url, r.status_code, attempt + 1)
            r.close()
            time.sleep(float(ra) if ra and ra.isdigit() else min(60, 2 ** (attempt + 1)))
        raise last_exc or RuntimeError("request failed")

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def download(self, url: str, dest: Path, *, session=None, headers=None,
                 max_bytes: Optional[int] = None, expect=("step", "zip")) -> Dict[str, Any]:
        """Stream url -> dest (via .part) and validate the bytes. Raises DownloadError on failure."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        r = self.get(url, stream=True, session=session, headers=headers or {}, allow_redirects=True)
        try:
            if r.status_code != 200:
                raise DownloadError(f"HTTP {r.status_code} from {url}: {r.text[:200]!r}")
            clen = int(r.headers.get("Content-Length") or 0)
            if max_bytes and clen > max_bytes:
                raise DownloadError(f"file is {clen / 1e6:.1f} MB > --max-file-mb", skipped=True)
            first, size, checked = b"", 0, False
            with open(part, "wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    if not chunk:
                        continue
                    if not checked:
                        first += chunk[: 4096 - len(first)]
                        if len(first) >= 64:
                            checked = True
                            kind = sniff(first)
                            if kind not in expect:
                                raise DownloadError(f"content is {kind!r}, not STEP/zip (starts {first[:80]!r})",
                                                    kind=kind)
                    f.write(chunk)
                    size += len(chunk)
                    if max_bytes and size > max_bytes:
                        raise DownloadError("exceeded --max-file-mb", skipped=True)
            kind = sniff(first)
            if kind not in expect:
                raise DownloadError(f"content is {kind!r}, not STEP/zip (starts {first[:80]!r})", kind=kind)
            if clen and size != clen and not r.headers.get("Content-Encoding"):
                raise DownloadError(f"truncated download ({size} of {clen} bytes)")
            os.replace(part, dest)
            return {"path": str(dest), "bytes": size, "kind": kind}
        except BaseException:
            try:
                part.unlink()
            except OSError:
                pass
            raise
        finally:
            r.close()


# ----------------------------------------------------------------------------
# page parsing
# ----------------------------------------------------------------------------
def fetch_entries(http: Http, mode: str, page_url: str) -> Dict[str, Any]:
    r = http.get(page_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    snap_tag = soup.find("script", id="cad-entries-snapshot")
    snapshot = json.loads(snap_tag.string) if snap_tag and snap_tag.string else []
    endpoint, live, live_err = None, None, None
    if mode in ("auto", "live"):
        for s in soup.find_all("script", src=True):
            if "CadCollection" in s["src"]:
                js = http.get(urljoin(page_url, s["src"])).text
                m = re.search(r"https://script\.google\.com/macros/s/[\w-]+/exec", js)
                endpoint = m.group(0) if m else None
                break
        if endpoint:
            try:
                j = http.get(endpoint + "?action=entries", timeout=90).json()
                if j.get("success") and isinstance(j.get("entries"), list) and j["entries"]:
                    live = j["entries"]
                else:
                    live_err = f"endpoint returned no entries: {str(j)[:200]}"
            except Exception as e:  # noqa
                live_err = str(e)
        else:
            live_err = "could not find the Apps Script endpoint in the page JS"
    if mode == "live" and live is None:
        raise SystemExit(f"--data-source live failed: {live_err}")
    if live is not None:
        entries, used = live, "live_apps_script"
    else:
        if mode == "auto":
            log.warning("live data unavailable (%s); using embedded snapshot", live_err)
        entries, used = snapshot, "embedded_snapshot"
    if not entries:
        raise SystemExit("No CAD entries found on the page - the page structure may have changed.")
    return {"entries": entries, "data_source": used, "endpoint": endpoint,
            "snapshot_count": len(snapshot), "live_count": len(live) if live else None}


ONSHAPE_RE = re.compile(
    r"https?://(?P<host>[a-z0-9-]+\.onshape\.com)/documents/(?P<did>[0-9a-f]{24})(?=[/?#]|$)"
    r"(?:/(?P<wvm>[wvm])/(?P<wvmid>[0-9a-f]{24})(?=[/?#]|$))?"
    r"(?:/e/(?P<eid>[0-9a-f]{24})(?=[/?#]|$))?", re.I)
GRABCAD_RE = re.compile(r"https?://(?:www\.)?grabcad\.com/library/(?P<slug>[^/?#\s]+)", re.I)


def classify_link(raw: str) -> Dict[str, Any]:
    link = (raw or "").strip()
    m = re.search(r"https?://\S+", link)
    url = m.group(0) if m else link
    info: Dict[str, Any] = {"url": url}
    if link != url:
        info["raw_link"] = raw
    host = urlparse(url).netloc.lower()
    if host.endswith("onshape.com"):
        om = ONSHAPE_RE.match(url)
        if om and link == url:
            cfg = parse_qs(urlparse(url).query).get("configuration", [None])[0]
            info.update(type="onshape", host=om.group("host").lower(), document_id=om.group("did").lower(),
                        wvm=om.group("wvm"), wvm_id=om.group("wvmid"), element_id=om.group("eid"),
                        configuration=(unquote(cfg) if cfg and cfg.lower() != "default" else None))
        else:
            info.update(type="onshape", unsupported="not a usable /documents/<24-hex id> link "
                                                    "(Onshape publication or malformed URL on the page)")
    elif host.endswith("grabcad.com"):
        gm = GRABCAD_RE.match(url)
        info["type"] = "grabcad"
        if gm:
            info["slug"] = gm.group("slug")
        else:
            info["unsupported"] = "not a grabcad.com/library/<slug> link"
    else:
        info.update(type="other", host=host)
    return info


def guess_robot_name(desc: str) -> Optional[str]:
    d = (desc or "").strip()
    m = re.match(r"^(?:comp(?:etition)?\s+|offseason\s+|off-season\s+|practice\s+)?robot\s*[–—-]\s*(.+)$", d, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"[\"“]([^\"“”]{2,40})[\"”]", d)
    return m.group(1).strip() if m else None


def build_index(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    robots = []
    for i, e in enumerate(entries):
        links = [classify_link(l) for l in (e.get("links") or []) if str(l).strip()]
        cad = [l for l in links if l["type"] in ("onshape", "grabcad")]
        if not cad:
            continue
        year = str(e.get("year") or "").strip()
        year_dir = year if re.fullmatch(r"(19|20)\d\d", year) else "unknown_year"
        team = str(e.get("team") or "").strip()
        row = e.get("rowIndex", i + 2)
        types = sorted({l["type"] for l in cad})
        name = guess_robot_name(e.get("description", ""))
        robots.append({
            "id": f"{year_dir}-{team or 'x'}-r{row}",
            "row_index": row,
            "team_number": int(team) if team.isdigit() else team,
            "team_name": (e.get("teamName") or "").strip(),
            "year": int(year) if re.fullmatch(r"(19|20)\d\d", year) else (year or None),
            "robot_name": name,
            "robot_name_source": "page_description" if name else None,
            "description": (e.get("description") or "").strip(),
            "page_format_field": (e.get("format") or "").strip(),
            "source_types": types,
            "source_type": types[0] if len(types) == 1 else "mixed",
            "page_links": [str(l).strip() for l in (e.get("links") or [])],
            "other_links": [l["url"] for l in links if l["type"] == "other"],
            "cad_sources": [dict(l, status="pending") for l in cad],
            "status": "pending",
            "_year_dir": year_dir,
        })
    # folder names <year>/<team#>_<TeamName>; disambiguated when a team has several entries in a year
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in robots:
        base = f"{r['team_number']}_{safe_name(r['team_name'], 50)}" if r["team_name"] else f"{r['team_number']}"
        groups.setdefault((r["_year_dir"], base), []).append(r)
    for (yd, base), rs in groups.items():
        for r in rs:
            r["folder"] = f"{yd}/{base}" if len(rs) == 1 else f"{yd}/{base}__r{r['row_index']}"
            del r["_year_dir"]
    return robots


# ----------------------------------------------------------------------------
# Onshape
# ----------------------------------------------------------------------------
ASM_GOOD = [("full robot", 6), ("robot", 5), ("full", 4), ("comp", 3), ("main", 3), ("top", 3),
            ("final", 2), ("release", 2), ("master assembly", 2), ("assembly", 1), ("asm", 1)]
ASM_BAD = ["intake", "shooter", "elevator", "arm", "climb", "drivetrain", "drive train", "swerve", "module",
           "gearbox", "bumper", "field", "test", "old", "archive", "wip", "sub", "sketch", "prototype",
           "wrist", "turret", "hood", "indexer", "feeder", "pivot", "mechanism", "superstructure", "copy",
           "cart", "frame", "practice", "electronics", "electrical"]


def score_element(name: str) -> int:
    n = (name or "").lower()
    s = sum(w for k, w in ASM_GOOD if k in n)
    s -= sum(3 for k in ASM_BAD if re.search(r"\b" + re.escape(k), n))
    return s


class Onshape:
    def __init__(self, http: Http, base: str, cookies: Dict[str, str], cookie_src: str,
                 access_key: Optional[str], secret_key: Optional[str], max_wait: float):
        self.http = http
        self.base = base.rstrip("/")
        self.max_wait = max_wait
        hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        self.anon = requests.Session()
        self.anon.headers.update(hdrs)
        self.auth_session: Optional[requests.Session] = None
        self.auth_mode: Optional[str] = None
        self.auth_ok, self.auth_msg = False, "no Onshape session cookie (or API keys) provided"
        self.auth_calls = 0
        self.user = None
        if cookies:
            s = requests.Session()
            s.headers.update(hdrs)
            s.headers.update({"Origin": self.base, "Referer": self.base + "/"})
            for k, v in cookies.items():
                s.cookies.set(k, v, domain=".onshape.com")
            if "XSRF-TOKEN" in cookies:  # web-session CSRF protection for POSTs
                s.headers["X-XSRF-TOKEN"] = unquote(cookies["XSRF-TOKEN"])
            self.auth_session, self.auth_mode = s, f"browser session cookie ({cookie_src})"
        elif access_key and secret_key:
            s = requests.Session()
            s.headers.update(hdrs)
            s.auth = (access_key, secret_key)  # Onshape accepts API keys as HTTP Basic auth
            self.auth_session, self.auth_mode = s, "API keys (ONSHAPE_ACCESS_KEY/SECRET_KEY)"

    def api(self, path: str) -> str:
        return f"{self.base}{ONSHAPE_API}{path}"

    def check_auth(self) -> Tuple[bool, str]:
        if not self.auth_session:
            return False, self.auth_msg
        r = self.http.get(self.api("/users/sessioninfo"), session=self.auth_session)
        self.auth_calls += 1
        if r.status_code == 200 and len(r.content) > 2:
            try:
                j = r.json()
                self.user = j.get("name") or j.get("email") or j.get("id")
            except ValueError:
                pass
            self.auth_ok, self.auth_msg = True, f"logged in via {self.auth_mode} as {self.user}"
        else:
            self.auth_msg = (f"Onshape session invalid/expired (sessioninfo HTTP {r.status_code}, "
                             f"{self.auth_mode}); re-run --login onshape")
        return self.auth_ok, self.auth_msg

    def get_json(self, path: str) -> Any:
        """Reads: anonymous first (works for public/link-shared docs), then the logged-in session."""
        r = self.http.get(self.api(path), session=self.anon)
        if r.status_code in (401, 403, 404) and self.auth_ok:
            r = self.http.get(self.api(path), session=self.auth_session)
            self.auth_calls += 1
        if r.headers.get("x-amzn-waf-action") or (r.status_code == 202 and not r.text.strip()):
            raise OnshapeBlocked(f"Onshape bot protection challenge on GET (HTTP {r.status_code})")
        if r.status_code != 200:
            raise RuntimeError(f"GET {path} -> HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def resolve(self, src: Dict[str, Any]) -> None:
        did = src["document_id"]
        doc = self.get_json(f"/documents/{did}")
        src.update(document_name=doc.get("name"), document_owner=(doc.get("owner") or {}).get("name"),
                   document_public=doc.get("public"), document_created_at=doc.get("createdAt"),
                   document_modified_at=doc.get("modifiedAt"), default_element_id=doc.get("defaultElementId"))
        if not src.get("wvm"):
            src["wvm"], src["wvm_id"] = "w", (doc.get("defaultWorkspace") or {}).get("id")
            src["wvm_inferred_from_default_workspace"] = True
        els = self.get_json(f"/documents/d/{did}/{src['wvm']}/{src['wvm_id']}/elements")
        src["elements"] = [{"id": e["id"], "name": e.get("name"), "type": e.get("elementType")}
                           for e in els if e.get("elementType") in ("ASSEMBLY", "PARTSTUDIO")]

    @staticmethod
    def choose_elements(src: Dict[str, Any], all_assemblies: bool) -> List[Dict[str, Any]]:
        els = src.get("elements") or []
        asms = [e for e in els if e["type"] == "ASSEMBLY"]
        pss = [e for e in els if e["type"] == "PARTSTUDIO"]
        if all_assemblies and asms:
            return [dict(e, chosen_because="--onshape-all-assemblies") for e in asms]
        if src.get("element_id"):
            for e in els:
                if e["id"] == src["element_id"]:
                    return [dict(e, chosen_because="element linked from the page")]
        pool = asms or pss
        if not pool:
            return []
        dflt = src.get("default_element_id")
        best = max(pool, key=lambda e: (score_element(e["name"]), e["id"] == dflt, -pool.index(e)))
        return [dict(best, chosen_because="best-named assembly" if asms else "no assemblies; best-named part studio")]

    def export_step(self, did: str, wvm: str, wvmid: str, el: Dict[str, Any], dest: Path,
                    configuration: Optional[str]) -> Dict[str, Any]:
        kind = "assemblies" if el["type"] == "ASSEMBLY" else "partstudios"
        body: Dict[str, Any] = {"formatName": "STEP", "storeInDocument": False, "allowFaultyParts": True,
                                "flattenAssemblies": False, "yAxisIsUp": False}
        if configuration:
            body["configuration"] = configuration
        s = self.auth_session
        r = self.http.post(self.api(f"/{kind}/d/{did}/{wvm}/{wvmid}/e/{el['id']}/translations"),
                           session=s, json=body)
        self.auth_calls += 1
        if r.headers.get("x-amzn-waf-action") or (r.status_code == 202 and not r.text.strip()):
            raise OnshapeBlocked(f"Onshape bot protection challenge (HTTP {r.status_code}, "
                                 f"waf={r.headers.get('x-amzn-waf-action')})")
        if r.status_code in (401, 403):
            raise PermissionError(f"translation request HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise RuntimeError(f"translation request HTTP {r.status_code}: {r.text[:300]}")
        tr = r.json()
        tid = tr["id"]
        t0, wait = time.monotonic(), 8.0
        while tr.get("requestState") == "ACTIVE":
            if time.monotonic() - t0 > self.max_wait:
                raise RuntimeError(f"translation {tid} still ACTIVE after {self.max_wait:.0f}s")
            time.sleep(wait)
            wait = min(wait * 1.5, 45)
            pr = self.http.get(self.api(f"/translations/{tid}"), session=s)
            self.auth_calls += 1
            if pr.status_code != 200:
                raise RuntimeError(f"poll translation HTTP {pr.status_code}: {pr.text[:200]}")
            tr = pr.json()
        if tr.get("requestState") != "DONE":
            raise RuntimeError(f"translation {tr.get('requestState')}: {tr.get('failureReason')}")
        fids = tr.get("resultExternalDataIds") or []
        if not fids:
            raise RuntimeError(f"translation DONE but no resultExternalDataIds: {str(tr)[:300]}")
        info = self.http.download(self.api(f"/documents/d/{did}/externaldata/{fids[0]}"), dest,
                                  session=s, headers={"Accept": "application/octet-stream"})
        self.auth_calls += 1
        info["translation_id"] = tid
        return info

    def copy_workspace(self, did: str, wid: str, name: str) -> Tuple[str, str]:
        r = self.http.post(self.api(f"/documents/{did}/workspaces/{wid}/copy"), session=self.auth_session,
                           json={"newName": name, "isPublic": False})
        self.auth_calls += 1
        if r.status_code != 200:
            raise RuntimeError(f"copy document HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        return j["newDocumentId"], j["newWorkspaceId"]


def process_onshape(osh: Onshape, robot: Dict, src: Dict, robot_dir: Path, args) -> None:
    if src.get("unsupported"):
        src.update(status="unsupported", error=src["unsupported"])
        return
    try:
        if "elements" not in src or args.refresh:
            osh.resolve(src)
    except OnshapeBlocked:
        raise
    except Exception as e:  # noqa
        src.update(status="error", error=f"could not read Onshape document: {e}")
        return
    if not robot.get("robot_name") and src.get("document_name"):
        robot["robot_name"], robot["robot_name_source"] = src["document_name"], "onshape_document_name"
    chosen = osh.choose_elements(src, args.onshape_all_assemblies)
    if not chosen:
        src.update(status="error", error="document has no assembly or part studio")
        return
    old = {x["element_id"]: x for x in src.get("exports", [])}
    exports = []
    for el in chosen:
        fname = safe_filename(f"{src.get('document_name') or src['document_id']} - {el['name']}.step")
        x = {**old.get(el["id"], {}), "element_id": el["id"], "element_name": el["name"],
             "element_type": el["type"], "chosen_because": el["chosen_because"],
             "path": str(robot_dir / "onshape" / fname), "status": "pending"}
        x.pop("error", None)
        exports.append(x)
    src["exports"] = exports
    src["error"] = None
    if args.dry_run:
        src["status"] = "dry_run"
        return
    for x in exports:
        if is_valid_step(Path(x["path"])):
            x.update(status="exists", bytes=Path(x["path"]).stat().st_size)
        elif Path(x["path"] + ".synced").exists():   # already moved to Google Drive by the sync script
            try:
                nbytes = int(Path(x["path"] + ".synced").read_text().strip() or 0)
            except Exception:  # noqa
                nbytes = 0
            x.update(status="exists", bytes=nbytes, synced_to_drive=True)
    if all(x["status"] == "exists" for x in exports):
        src["status"] = "downloaded"
        return
    if not osh.auth_ok:
        for x in exports:
            if x["status"] != "exists":
                x["status"] = "skipped_no_onshape_login"
        src.update(status="skipped_no_onshape_login",
                   error=f"Onshape only exports for logged-in users ({osh.auth_msg}). "
                         f"Run: python3 frc_cad_downloader.py --login onshape  (or set ONSHAPE_COOKIE)")
        return
    errors = []
    for x in exports:
        if x["status"] == "exists":
            continue
        dest = Path(x["path"])
        if args.onshape_max_calls and osh.auth_calls >= args.onshape_max_calls:
            x.update(status="skipped_call_budget", error="--onshape-max-calls reached")
            errors.append(x["error"])
            continue
        el = {"id": x["element_id"], "type": x["element_type"], "name": x["element_name"]}
        try:
            log.info("  onshape: exporting '%s' / '%s' (%s)", src.get("document_name"), el["name"], el["type"])
            info = osh.export_step(src["document_id"], src["wvm"], src["wvm_id"], el, dest, src.get("configuration"))
            x.update(status="downloaded", bytes=info["bytes"], kind=info["kind"],
                     translation_id=info["translation_id"])
        except OnshapeBlocked:
            raise
        except PermissionError as e:
            if not args.onshape_copy_fallback:
                x.update(status="error", error=f"export denied: {e} (owner may block exports; "
                                               f"try --onshape-copy-fallback)")
            elif src["wvm"] != "w":
                x.update(status="error", error=f"export denied: {e}; copy fallback needs a workspace link")
            else:
                try:
                    log.info("  onshape: export denied, copying document into your account")
                    ndid, nwid = osh.copy_workspace(src["document_id"], src["wvm_id"],
                                                    f"frc_cad copy - {src.get('document_name')}")
                    nels = osh.get_json(f"/documents/d/{ndid}/w/{nwid}/elements")
                    match = next((n for n in nels if n.get("name") == el["name"]
                                  and n.get("elementType") == el["type"]), None)
                    if not match:
                        raise RuntimeError("element not found in the copied document")
                    info = osh.export_step(ndid, "w", nwid, dict(el, id=match["id"]), dest, src.get("configuration"))
                    x.update(status="downloaded", bytes=info["bytes"], kind=info["kind"],
                             via_copied_document_id=ndid)
                except OnshapeBlocked:
                    raise
                except Exception as e2:  # noqa
                    x.update(status="error", error=f"export denied ({e}); copy fallback failed: {e2}")
        except DownloadError as e:
            x.update(status="error", error=f"exported file rejected: {e}")
        except Exception as e:  # noqa
            x.update(status="error", error=f"{type(e).__name__}: {e}")
        if x["status"] == "downloaded" and x.get("kind") == "zip":
            x["extracted"] = extract_steps(dest, dest.parent / (dest.stem + "_unzipped"), keep_zip=False)
        if x.get("error"):
            errors.append(x["error"])
    ok = [x for x in exports if x["status"] in ("downloaded", "exists")]
    src["status"] = "downloaded" if len(ok) == len(exports) else ("partial" if ok else "error")
    src["error"] = "; ".join(errors) or None


# ----------------------------------------------------------------------------
# GrabCAD
# ----------------------------------------------------------------------------
class GrabCAD:
    def __init__(self, http: Http, cookies: Dict[str, str], cookie_src: str, direct: bool):
        self.http = http
        self.direct = direct
        self.cookie_src = cookie_src
        self.auth: Optional[requests.Session] = None
        if cookies:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, */*"})
            for k, v in cookies.items():
                s.cookies.set(k, v, domain=".grabcad.com")
            self.auth = s

    @staticmethod
    def api(slug: str, suffix: str = "") -> str:
        return f"{GRABCAD_BASE}/community/api/v1/models/{slug}{suffix}"

    def api_get(self, url: str, **kw) -> requests.Response:
        """GET a GrabCAD JSON API URL. GrabCAD answers bursts of API calls (~10 models in
        ~70 s) with a temporary HTTP 403/429, so back off (30, 60, 90, 120 s) and retry."""
        waits = [30, 60, 90, 120]
        for i in range(len(waits) + 1):
            r = self.http.get(url, **kw)
            if r.status_code not in (403, 429) or i == len(waits):
                return r
            log.info("  grabcad: API HTTP %s (rate limit?), waiting %ss and retrying", r.status_code, waits[i])
            r.close()
            time.sleep(waits[i])
        return r

    def resolve(self, src: Dict[str, Any]) -> None:
        slug = src["slug"]
        r = self.api_get(self.api(slug), headers={"Accept": "application/json"})
        if r.status_code == 404:
            raise RuntimeError("GrabCAD model not found (404) - removed, private or renamed")
        if r.status_code != 200:
            raise RuntimeError(f"GrabCAD model info HTTP {r.status_code}")
        m = r.json()
        src.update(model_name=m.get("name"), author=(m.get("author") or {}).get("name"),
                   model_created_at=m.get("created_at"), downloads_count=m.get("downloads_count"),
                   likes_count=m.get("likes_count"), model_description=m.get("raw_description"),
                   preview_image=m.get("preview_image"), video_link=m.get("video_link"))
        files: List[Dict[str, Any]] = []

        def walk(folder_id: Optional[int], prefix: str, depth: int) -> None:
            if depth > 8:
                return
            fr = self.api_get(self.api(slug, "/files"), params={"folder_id": folder_id} if folder_id else None,
                               headers={"Accept": "application/json"})
            if fr.status_code != 200:
                raise RuntimeError(f"GrabCAD file list HTTP {fr.status_code}")
            j = fr.json()
            for f in j.get("files", []):
                mm = re.search(r"cadid=([0-9a-f]+)", f.get("download_url") or "")
                files.append({"name": f.get("name"), "folder": prefix, "extension": (f.get("extension") or "").lower(),
                              "system": f.get("system"), "cadid": mm.group(1) if mm else None,
                              "download_url": f.get("download_url")})
            for sub in j.get("folders", []):
                walk(sub["id"], f"{prefix}{sub.get('name') or sub['id']}/", depth + 1)

        walk(None, "", 0)
        src["files_listing"] = files

    def file_url(self, f: Dict[str, Any]) -> str:
        """Return the URL of the real file bytes."""
        if self.auth is not None:
            r = self.http.get(f["download_url"], session=self.auth, allow_redirects=False)
            if r.status_code == 401:
                raise DownloadError("GrabCAD session invalid/expired (401 'You need to log in'); "
                                    "re-run --login grabcad")
            if r.status_code in (301, 302, 303, 307, 308):
                return urljoin(f["download_url"], r.headers["Location"])
            if r.status_code == 200:
                try:
                    j = r.json()
                except ValueError:
                    raise DownloadError(f"unexpected download response: {r.text[:200]!r}")
                if isinstance(j, dict) and j.get("url"):
                    return j["url"]  # {"url": "https://grabcad.com/cads/files/<cadid>/original.<ext>"}
                raise DownloadError(f"download endpoint returned no url: {str(j)[:200]}")
            raise DownloadError(f"download endpoint HTTP {r.status_code}: {r.text[:200]!r}")
        if self.direct and f.get("cadid"):
            ext = f["name"].rsplit(".", 1)[-1]  # case-sensitive: must match the uploaded file name
            return f"{GRABCAD_BASE}/cads/files/{f['cadid']}/original.{ext}"
        raise DownloadError("GrabCAD download needs a login session (run --login grabcad or set "
                            "GRABCAD_COOKIE), or opt in with --grabcad-direct", skipped=True)


def extract_steps(zpath: Path, outdir: Path, keep_zip: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"zip": str(zpath), "step_files": [], "errors": []}
    try:
        with zipfile.ZipFile(zpath) as z:
            infos = z.infolist()
            res["zip_member_count"] = len(infos)
            for m in infos:
                if m.is_dir() or ext_of(m.filename) not in STEP_EXTS:
                    continue
                parts = [safe_filename(p) for p in m.filename.replace("\\", "/").split("/")
                         if p not in ("", ".", "..")]
                dest = outdir.joinpath(*parts)  # sanitized: no zip-slip
                if is_valid_step(dest):
                    res["step_files"].append({"path": str(dest), "bytes": dest.stat().st_size, "status": "exists"})
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as zsrc, open(dest, "wb") as out:
                    shutil.copyfileobj(zsrc, out, 1024 * 1024)
                if file_kind(dest) != "step":
                    res["errors"].append(f"{m.filename}: not a real STEP file ({file_kind(dest)})")
                    dest.unlink()
                else:
                    res["step_files"].append({"path": str(dest), "bytes": dest.stat().st_size, "status": "extracted"})
    except zipfile.BadZipFile as e:
        res["errors"].append(f"bad zip: {e}")
    if not res["step_files"]:
        res["errors"].append("zip contained no STEP/STP files")
    if not keep_zip and res["step_files"]:
        try:
            zpath.unlink()
            res["zip_deleted"] = True
        except OSError:
            pass
    return res


# Uploaders often cap single files (e.g. 100 MB). STEP is text and deflates ~5-7x.
UPLOAD_PART_BYTES = 95 * 1000 * 1000


def compress_for_upload(path: Path, over_bytes: int, part_bytes: int = UPLOAD_PART_BYTES,
                        delete_original: bool = True) -> Dict[str, Any]:
    """If `path` is larger than `over_bytes`, zip it (one file per zip, verified with testzip) as
    <name>.zip. If the zip is still larger than `part_bytes`, split it into raw byte parts
    <name>.zip.001, .002, ... (7-Zip opens .001 directly; or `cat name.zip.0* > name.zip`).
    Returns {"files": [paths to upload], "compressed": None|"zip"|"zip_split", ...}."""
    size = path.stat().st_size
    if not over_bytes or size <= over_bytes:
        return {"files": [str(path)], "compressed": None, "original_bytes": size}
    zpath = path.with_name(path.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.write(path, arcname=path.name)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        info = z.infolist()
    if bad or len(info) != 1 or info[0].file_size != size:
        zpath.unlink()
        raise DownloadError(f"zip verification failed for {path.name} (bad member: {bad})")
    zsize = zpath.stat().st_size
    res: Dict[str, Any] = {"original_bytes": size, "zip_bytes": zsize, "compressed": "zip", "files": [str(zpath)]}
    if zsize > part_bytes:
        parts, i = [], 1
        with open(zpath, "rb") as src:
            while True:
                chunk = src.read(part_bytes)
                if not chunk:
                    break
                pp = zpath.with_name(f"{zpath.name}.{i:03d}")
                with open(pp, "wb") as out:
                    out.write(chunk)
                parts.append(str(pp))
                i += 1
        zpath.unlink()
        res.update(compressed="zip_split", files=parts,
                   rejoin="7-Zip: open the .001 part; or: cat '<name>.zip.0'* > '<name>.zip'")
    if delete_original:
        path.unlink()
    return res


IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "bmp", "txt", "pdf", "mp4", "webp", "lck"}


def process_grabcad(gc: GrabCAD, robot: Dict, src: Dict, robot_dir: Path, args) -> None:
    if src.get("unsupported"):
        src.update(status="unsupported", error=src["unsupported"])
        return
    try:
        if "files_listing" not in src or args.refresh:
            gc.resolve(src)
    except Exception as e:  # noqa
        src.update(status="error", error=str(e))
        return
    if not robot.get("robot_name") and src.get("model_name"):
        robot["robot_name"], robot["robot_name_source"] = src["model_name"], "grabcad_model_name"
    listing = src["files_listing"]
    steps = [f for f in listing if ext_of(f["name"]) in STEP_EXTS]
    zips = [f for f in listing if ext_of(f["name"]) in ZIP_EXTS]
    use_zips = args.grabcad_zips == "always" or (args.grabcad_zips == "auto" and not steps)
    src.update(available_step_files=len(steps), available_zip_files=len(zips),
               other_formats=sorted({ext_of(f["name"]) for f in listing} - STEP_EXTS - ZIP_EXTS - IMAGE_EXTS - {""}))
    base = robot_dir / "grabcad" / safe_name(src["slug"], 80)
    old = {f.get("cadid"): f for f in src.get("files", [])}
    out_files = []
    for f in steps + (zips if use_zips else []):
        sub = [safe_filename(p) for p in (f["folder"] or "").split("/") if p]
        prev = old.get(f["cadid"], {})
        out_files.append({**f, "path": str(base.joinpath(*sub, safe_filename(f["name"]))),
                          "is_zip": f in zips, "status": "pending",
                          **({"extracted": prev["extracted"]} if prev.get("extracted") else {})})
    src["files"] = out_files
    src["error"] = None
    if not out_files:
        src.update(status="no_step_files", error="model has no STEP/STP/zip files (formats: "
                                                 f"{', '.join(src['other_formats']) or 'none'})")
        return
    if args.dry_run:
        src["status"] = "dry_run"
        return
    errors = []
    for f in out_files:
        dest = Path(f["path"])
        ex = f.get("extracted") or {}
        if f["is_zip"] and ex.get("step_files") and all(is_valid_step(Path(p["path"])) for p in ex["step_files"]):
            f["status"] = "exists"
            continue
        if not f["is_zip"] and is_valid_step(dest):
            f.update(status="exists", bytes=dest.stat().st_size)
            continue
        try:
            url = gc.file_url(f)
            f["file_url"] = url
            log.info("  grabcad: downloading %s%s", f["folder"], f["name"])
            info = gc.http.download(url, dest, max_bytes=int(args.max_file_mb * 1e6) if args.max_file_mb else None)
            f.update(status="downloaded", bytes=info["bytes"], kind=info["kind"])
            if info["kind"] == "zip":
                f["extracted"] = extract_steps(dest, dest.parent / (dest.stem + "_unzipped"), args.keep_zips)
                if not f["extracted"]["step_files"]:
                    f.update(status="error", error="; ".join(f["extracted"]["errors"]))
            elif getattr(args, "compress_over_mb", 0):
                f["upload"] = compress_for_upload(dest, int(args.compress_over_mb * 1e6))
        except DownloadError as e:
            f.update(status="skipped" if e.skipped else "error", error=str(e))
        except Exception as e:  # noqa
            f.update(status="error", error=f"{type(e).__name__}: {e}")
        if f.get("error"):
            errors.append(f"{f['name']}: {f['error']}")
    ok = [f for f in out_files if f["status"] in ("downloaded", "exists")]
    if len(ok) == len(out_files):
        src["status"] = "downloaded"
    elif ok:
        src["status"] = "partial"
    elif gc.auth is None and not gc.direct:
        src["status"] = "skipped_no_grabcad_login"
    else:
        src["status"] = "error"
    msgs = list(dict.fromkeys(e.split(": ", 1)[-1] for e in errors)) if src["status"].startswith("skipped") else errors
    src["error"] = "; ".join(msgs) or None


# ----------------------------------------------------------------------------
# repair old 87-byte JSON placeholder files ({"url": ".../original.STEP"})
# ----------------------------------------------------------------------------
def repair_placeholders(http: Http, folder: Path) -> int:
    n_ok = n_bad = n_seen = 0
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or ext_of(p.name) not in STEP_EXTS or p.stat().st_size > 4096 or file_kind(p) != "json":
            continue
        try:
            url = json.loads(p.read_text(encoding="utf-8")).get("url")
        except Exception:  # noqa
            url = None
        if not url:
            continue
        n_seen += 1
        tmp = p.with_name(p.name + ".download")
        try:
            info = http.download(url, tmp)
            if info["kind"] == "zip":
                ex = extract_steps(tmp, p.parent / (p.stem + "_unzipped"), keep_zip=False)
                if not ex["step_files"]:
                    raise DownloadError("; ".join(ex["errors"]))
                p.unlink()
                log.info("repaired %s -> zip with %d STEP files", p.name, len(ex["step_files"]))
            else:
                os.replace(tmp, p)
                log.info("repaired %s (%.1f MB)", p.name, info["bytes"] / 1e6)
            n_ok += 1
        except Exception as e:  # noqa
            log.error("could not repair %s: %s", p, e)
            n_bad += 1
    log.info("placeholder repair: %d placeholders found, %d fixed, %d failed", n_seen, n_ok, n_bad)
    return 0 if n_bad == 0 else 1


# ----------------------------------------------------------------------------
# optional Playwright login helper
# ----------------------------------------------------------------------------
GRABCAD_PROBE = ("https://grabcad.com/community/api/v1/models/robonauts-118-2025-robot-firefly-1/"
                 "files/download?cadid=33cc6e651ae17cff08d6ed0057ad7974")


def login_helper(service: str, state_dir: Path, timeout_s: int, headless: bool = False) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("--login needs Playwright: pip install playwright   (it uses Google Chrome if installed, "
                  "otherwise run: playwright install chromium)")
        return 2
    state_dir.mkdir(parents=True, exist_ok=True)
    out = state_dir / f"{service}_cookies.json"
    url = "https://cad.onshape.com/signin" if service == "onshape" else "https://grabcad.com/login"
    with sync_playwright() as pw:
        browser = None
        for kw in ({"channel": "chrome"}, {}):
            try:
                browser = pw.chromium.launch(headless=headless, **kw)
                break
            except Exception as e:  # noqa
                log.debug("browser launch %s failed: %s", kw, e)
        if browser is None:
            log.error("Could not start a browser. Install Google Chrome or run: playwright install chromium")
            return 2
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(url)
        print(f"\nA browser window opened at {url}\nSign in there normally. Waiting up to "
              f"{max(1, timeout_s // 60)} min (this window closes by itself once you're logged in)...\n", flush=True)
        t0, ok = time.time(), False
        while time.time() - t0 < timeout_s and not ok:
            time.sleep(3)
            try:
                if service == "onshape":
                    res = ctx.request.get("https://cad.onshape.com/api/v10/users/sessioninfo")
                    ok = res.status == 200 and len(res.body()) > 2
                else:
                    res = ctx.request.get(GRABCAD_PROBE, max_redirects=0)
                    ok = res.status != 401
            except Exception as e:  # noqa
                log.debug("login check: %s", e)
        if not ok:
            log.error("Timed out waiting for a logged-in %s session.", service)
            browser.close()
            return 1
        domain = "onshape.com" if service == "onshape" else "grabcad.com"
        cookies = [c for c in ctx.cookies() if domain in c.get("domain", "")]
        out.write_text(json.dumps(cookies, indent=1), encoding="utf-8")
        try:
            os.chmod(out, 0o600)
        except OSError:
            pass
        browser.close()
    print(f"Saved {service} session cookies to {out} (used automatically on later runs).")
    return 0


# ----------------------------------------------------------------------------
# index / status helpers
# ----------------------------------------------------------------------------
def summarize(robots: List[Dict]) -> Dict[str, Any]:
    by_year: Dict[str, Dict[str, int]] = {}
    status: Dict[str, int] = {}
    for r in robots:
        d = by_year.setdefault(str(r["year"]), {"robots": 0, "onshape": 0, "grabcad": 0})
        d["robots"] += 1
        for t in r["source_types"]:
            d[t] += 1
        st = r.get("status", "pending")
        status[st] = status.get(st, 0) + 1
    return {
        "robots_with_onshape_or_grabcad": len(robots),
        "robots_with_onshape": sum("onshape" in r["source_types"] for r in robots),
        "robots_with_grabcad": sum("grabcad" in r["source_types"] for r in robots),
        "robots_with_both": sum(len(r["source_types"]) == 2 for r in robots),
        "onshape_links": sum(s["type"] == "onshape" for r in robots for s in r["cad_sources"]),
        "grabcad_links": sum(s["type"] == "grabcad" for r in robots for s in r["cad_sources"]),
        "unsupported_links": sum(bool(s.get("unsupported")) for r in robots for s in r["cad_sources"]),
        "robot_status_counts": status,
        "by_year": dict(sorted(by_year.items(), reverse=True)),
    }


def robot_status(srcs: List[Dict]) -> str:
    st = [s.get("status", "pending") for s in srcs]
    if not st or all(s == "pending" for s in st):
        return "pending"
    if all(s == "dry_run" for s in st):
        return "dry_run"
    if all(s == "downloaded" for s in st):
        return "downloaded"
    if any(s in ("downloaded", "partial") for s in st):
        return "partial"
    if all(s.startswith("skipped") for s in st):
        return st[0]
    if "error" in st:
        return "error"
    return st[0] if len(set(st)) == 1 else "incomplete"


def collect_files(r: Dict) -> List[str]:
    files = []
    for s in r["cad_sources"]:
        for x in s.get("exports", []) + s.get("files", []):
            ex = x.get("extracted") or {}
            if ex.get("step_files"):
                files += [p["path"] for p in ex["step_files"]]
            elif x.get("status") in ("downloaded", "exists") and not x.get("is_zip"):
                files.append(x["path"])
    return files


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
class OnshapeBlocked(Exception):
    """Onshape's WAF answered with a challenge; stop and back off instead of failing robots."""


EXIT_BLOCKED = 75


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Download STEP files for the FRC robots listed on the Spectrum 3847 FRC CAD Collection "
                    "(Onshape + GrabCAD) and write JSON metadata. Requires: pip install requests beautifulsoup4 "
                    "(optional: playwright, for --login). Onshape exports need a logged-in Onshape session "
                    "(free account; --login onshape or ONSHAPE_COOKIE). GrabCAD downloads need a GrabCAD session "
                    "(--login grabcad or GRABCAD_COOKIE) or the opt-in --grabcad-direct. See the script docstring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-o", "--out", default="./frc_cad", help="output directory")
    ap.add_argument("--year", action="append", help="only these years (repeatable or comma list, e.g. 2024,2025)")
    ap.add_argument("--team", action="append", help="only these team numbers (repeatable or comma list)")
    ap.add_argument("--source", choices=["onshape", "grabcad", "all"], default="all")
    ap.add_argument("--limit", type=int, help="process at most N robots (after filters)")
    ap.add_argument("--skip-done", action="store_true", help="skip Onshape sources already downloaded or export-denied in a previous run (no API calls)")
    ap.add_argument("--dry-run", action="store_true", help="only build and write robots.json; download nothing")
    ap.add_argument("--resolve", action="store_true",
                    help="with --dry-run: also look up Onshape docs/elements and GrabCAD file lists (no login needed)")
    ap.add_argument("--refresh", action="store_true", help="re-query Onshape/GrabCAD metadata even if cached")
    ap.add_argument("--data-source", choices=["auto", "live", "snapshot"], default="auto",
                    help="live = the page's Google Apps Script feed, snapshot = JSON embedded in the page HTML")
    ap.add_argument("--page-url", default=PAGE_URL)
    ap.add_argument("--delay", type=float, default=1.0, help="min seconds between requests to the same host")
    ap.add_argument("--robot-delay", type=float, default=0.0,
                    help="seconds to wait after each robot that made Onshape export calls (be polite)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    ap.add_argument("--max-file-mb", type=float, default=0, help="skip single files larger than this (0 = no limit)")
    ap.add_argument("--compress-over-mb", type=float, default=0,
                    help="zip downloaded GrabCAD STEP files larger than this (one file per zip, testzip-verified; "
                         "split into ~95 MB .zip.001/.002 parts if still too big) and delete the raw file (0 = off)")
    g = ap.add_argument_group("Onshape")
    g.add_argument("--onshape-cookie-file", help="file with Onshape session cookies (default: env ONSHAPE_COOKIE, "
                                                 "then ~/.frc_cad/onshape_cookies.json saved by --login onshape)")
    g.add_argument("--onshape-base", default=ONSHAPE_BASE)
    g.add_argument("--onshape-all-assemblies", action="store_true", help="export every assembly, not just the main one")
    g.add_argument("--onshape-copy-fallback", action="store_true",
                   help="if an export is denied (403), copy the doc into your account and export the copy")
    g.add_argument("--onshape-max-calls", type=int, default=0,
                   help="stop authenticated Onshape calls after N (0 = unlimited; useful with API keys)")
    g.add_argument("--onshape-max-wait", type=float, default=1200, help="max seconds to wait for one translation")
    g = ap.add_argument_group("GrabCAD")
    g.add_argument("--grabcad-cookie-file", help="file with GrabCAD session cookies (default: env GRABCAD_COOKIE, "
                                                 "then ~/.frc_cad/grabcad_cookies.json saved by --login grabcad)")
    g.add_argument("--grabcad-direct", action="store_true",
                   help="opt-in: fetch .../cads/files/<cadid>/original.<ext> directly without a login session")
    g.add_argument("--grabcad-zips", choices=["auto", "always", "never"], default="auto",
                   help="download .zip files: auto = only when the model has no loose STEP file")
    g.add_argument("--keep-zips", action="store_true", help="keep zip archives after extracting their STEP files")
    g = ap.add_argument_group("Utilities")
    g.add_argument("--login", choices=["onshape", "grabcad"],
                   help="open a browser, sign in, save session cookies to --state-dir (needs playwright)")
    g.add_argument("--login-timeout", type=int, default=600, help="seconds to wait for you to sign in")
    g.add_argument("--repair-placeholders", metavar="DIR",
                   help="replace small JSON {\"url\": ...} placeholder .step/.stp files under DIR with the real files")
    g.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR), help="where --login stores session cookies")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)
    if args.resolve and not args.dry_run:
        args.resolve = False  # normal runs always resolve

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "frc_cad_downloader.log", encoding="utf-8")],
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    state_dir = Path(args.state_dir).expanduser()

    if args.login:
        return login_helper(args.login, state_dir, args.login_timeout)
    http = Http(delay=args.delay, retries=args.retries, timeout=args.timeout)
    if args.repair_placeholders:
        return repair_placeholders(http, Path(args.repair_placeholders).expanduser())

    log.info("Fetching %s", args.page_url)
    data = fetch_entries(http, args.data_source, args.page_url)
    robots = build_index(data["entries"])
    log.info("Page entries: %d (from %s). Entries with Onshape/GrabCAD CAD: %d",
             len(data["entries"]), data["data_source"], len(robots))

    # resume: merge cached per-robot metadata from earlier runs
    for r in robots:
        mp = out / r["folder"] / "metadata.json"
        if not mp.is_file():
            continue
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa
            log.warning("could not read %s: %s", mp, e)
            continue
        old_srcs = {s.get("url"): s for s in old.get("cad_sources", [])}
        for i, s in enumerate(r["cad_sources"]):
            if s["url"] in old_srcs:
                r["cad_sources"][i] = {**old_srcs[s["url"]],
                                       **{k: v for k, v in s.items() if v is not None and k != "status"}}
        for k in ("status", "updated_at", "local_files"):
            if old.get(k):
                r[k] = old[k]
        if not r.get("robot_name") and old.get("robot_name"):
            r["robot_name"], r["robot_name_source"] = old["robot_name"], old.get("robot_name_source")

    years = {y.strip() for v in (args.year or []) for y in v.split(",") if y.strip()}
    teams = {t.strip() for v in (args.team or []) for t in v.split(",") if t.strip()}
    selected = [r for r in robots
                if (not years or str(r["year"]) in years)
                and (not teams or str(r["team_number"]) in teams)
                and (args.source == "all" or args.source in r["source_types"])]
    if args.limit:
        selected = selected[: args.limit]
    log.info("Selected %d robots (year=%s team=%s source=%s limit=%s)%s", len(selected),
             ",".join(sorted(years)) or "all", ",".join(sorted(teams)) or "all", args.source, args.limit,
             " [dry run]" if args.dry_run else "")

    osh: Optional[Onshape] = None
    gc: Optional[GrabCAD] = None
    auth_info = {"onshape": "not used", "grabcad": "not used"}
    work = (not args.dry_run) or args.resolve
    if work and args.source in ("all", "onshape") and any("onshape" in r["source_types"] for r in selected):
        oc, osrc = load_cookies("ONSHAPE_COOKIE", args.onshape_cookie_file, state_dir / "onshape_cookies.json")
        osh = Onshape(http, args.onshape_base, oc, osrc, os.environ.get("ONSHAPE_ACCESS_KEY"),
                      os.environ.get("ONSHAPE_SECRET_KEY"), args.onshape_max_wait)
        if args.dry_run:
            auth_info["onshape"] = "dry run: anonymous metadata reads only"
        else:
            ok, msg = osh.check_auth()
            auth_info["onshape"] = msg
            (log.info if ok else log.warning)("Onshape: %s", msg)
    if work and args.source in ("all", "grabcad") and any("grabcad" in r["source_types"] for r in selected):
        gco, gsrc = load_cookies("GRABCAD_COOKIE", args.grabcad_cookie_file, state_dir / "grabcad_cookies.json")
        gc = GrabCAD(http, gco, gsrc, args.grabcad_direct)
        auth_info["grabcad"] = (f"session cookie ({gsrc})" if gco else
                                "direct file URLs (--grabcad-direct)" if args.grabcad_direct else
                                "no login: file lists only, downloads skipped")
        log.info("GrabCAD: %s", auth_info["grabcad"])

    def write_index(final: bool) -> None:
        write_json_atomic(out / "robots.json", {
            "generated_at": now_iso(),
            "tool": f"frc_cad_downloader.py {__version__}",
            "source_page": args.page_url,
            "data_source": data["data_source"],
            "data_endpoint": data["endpoint"],
            "total_page_entries": len(data["entries"]),
            "embedded_snapshot_entries": data["snapshot_count"],
            "page_entries_without_onshape_or_grabcad": len(data["entries"]) - len(robots),
            "summary": summarize(robots),
            "last_run": {
                "finished": final, "dry_run": args.dry_run, "resolve": args.resolve,
                "filters": {"year": sorted(years), "team": sorted(teams), "source": args.source, "limit": args.limit},
                "selected_robot_ids": [r["id"] for r in selected],
                "auth": auth_info,
                "onshape_authenticated_calls": osh.auth_calls if osh else 0,
                "http_requests_by_host": http.calls,
            },
            "robots": robots,
        })

    for n, r in enumerate(selected, 1):
        calls_before = osh.auth_calls if osh else 0
        robot_dir = out / r["folder"]
        log.info("[%d/%d] %s team %s %s (%s)", n, len(selected), r["year"], r["team_number"], r["team_name"],
                 "+".join(r["source_types"]))
        for s in r["cad_sources"]:
            if args.source != "all" and s["type"] != args.source:
                continue
            if not work:
                s["status"] = "dry_run"
                continue
            if (getattr(args, "skip_done", False) and s["type"] == "onshape"
                    and (s.get("status") == "downloaded"
                         or (s.get("status") == "error" and "export denied" in str(s.get("error", ""))))):
                log.info("  onshape: already handled earlier (%s), skipping", s.get("status"))
                continue
            try:
                if s["type"] == "onshape":
                    process_onshape(osh, r, s, robot_dir, args)
                else:
                    process_grabcad(gc, r, s, robot_dir, args)
            except KeyboardInterrupt:
                raise
            except OnshapeBlocked as e:
                log.warning("STOPPING: %s. Nothing marked failed; re-run later to resume.", e)
                if not args.dry_run:
                    write_index(final=False)
                return EXIT_BLOCKED
            except Exception as e:  # noqa
                log.exception("unexpected error")
                s.update(status="error", error=f"{type(e).__name__}: {e}")
            log.info("  %s %s -> %s%s", s["type"], s.get("document_name") or s.get("model_name") or s["url"],
                     s["status"], f": {s['error'][:250]}" if s.get("error") else "")
        r["status"] = robot_status(r["cad_sources"])
        r["updated_at"] = now_iso()
        r["local_files"] = collect_files(r)
        r["local_dir"] = str(robot_dir)
        if not args.dry_run:
            write_json_atomic(robot_dir / "metadata.json", r)
            write_index(final=False)
        if args.robot_delay and osh and osh.auth_calls > calls_before and n < len(selected):
            time.sleep(args.robot_delay)
    write_index(final=True)
    sm = summarize(robots)
    log.info("Wrote %s", out / "robots.json")
    log.info("Index: %d robots with Onshape/GrabCAD CAD (Onshape: %d, GrabCAD: %d). Status counts: %s",
             sm["robots_with_onshape_or_grabcad"], sm["robots_with_onshape"], sm["robots_with_grabcad"],
             sm["robot_status_counts"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted - re-run the same command to resume.\n")
        sys.exit(130)
