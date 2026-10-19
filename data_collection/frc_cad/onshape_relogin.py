#!/usr/bin/env python3
"""Sign in to Onshape with ONSHAPE_PASSWORD and write fresh session cookies for the Mac runner.
Output: /workspace/frc-cad/onshape_cookies_fresh.json (JSON cookie list). Exit 0 on a verified session."""
import os, sys, json, requests
EMAIL = os.environ.get("ONSHAPE_EMAIL", "")  # set ONSHAPE_EMAIL in your env (scrubbed)
OUT = "/workspace/frc-cad/onshape_cookies_fresh.json"
pw = os.environ.get("ONSHAPE_PASSWORD")
if not pw: print("ONSHAPE_PASSWORD missing"); sys.exit(2)
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
                  "Accept": "application/json;charset=UTF-8; qs=0.09", "Content-Type": "application/json"})
r = s.post("https://cad.onshape.com/api/users/session", json={"email": EMAIL, "password": pw}, timeout=30)
if r.headers.get("x-amzn-waf-action") or r.status_code != 200:
    print("login failed", r.status_code, r.headers.get("x-amzn-waf-action")); sys.exit(1)
r2 = s.get("https://cad.onshape.com/api/users/sessioninfo", timeout=30)
if r2.status_code != 200 or not r2.text:
    print("sessioninfo failed", r2.status_code); sys.exit(1)
cookies = [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path} for c in s.cookies]
with open(OUT, "w") as f: json.dump(cookies, f)
os.chmod(OUT, 0o600)
print("OK session for", r2.json().get("name"), "->", OUT)
