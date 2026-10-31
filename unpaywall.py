"""
unpaywall.py
------------
Free, legal open-access full-text resolution through the Unpaywall REST API.

    GET https://api.unpaywall.org/v2/{doi}?email={your_email}

Unpaywall is free for academic/developer use but REQUIRES a real contact email
(their "polite use" rule). The web UI asks each user for their own email; the CLI
reads UNPAYWALL_EMAIL. No email -> resolution is skipped, nothing breaks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from http_client import HttpClient, HttpError

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


def valid_email(email: Optional[str]) -> bool:
    e = (email or "").strip().lower()
    return bool(_EMAIL_RE.match(e)) and not e.endswith(("@example.com", "@example.org", "@test.com"))


class UnpaywallClient:
    def __init__(self, email: Optional[str], http: Optional[HttpClient] = None):
        self.email = (email or "").strip()
        self.http = http or HttpClient(cache_dir="cache/unpaywall", timeout=20.0, retries=1)

    @property
    def enabled(self) -> bool:
        return valid_email(self.email)

    def resolve(self, doi: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the best legal OA location for a DOI, or None."""
        if not doi or not self.enabled:
            return None
        doi = doi.strip().lower()
        try:
            data = self.http.get_json(f"{UNPAYWALL_BASE}/{quote(doi, safe='/')}", params={"email": self.email})
        except HttpError as exc:
            if "404" not in str(exc):
                print(f"  [Unpaywall] lookup failed for {doi}: {str(exc)[:120]}")
            return None
        if not isinstance(data, dict):
            return None

        best = data.get("best_oa_location") or {}
        locations: List[Dict[str, Any]] = [loc for loc in (data.get("oa_locations") or []) if isinstance(loc, dict)]
        pdf_urls = []
        for loc in [best] + locations:
            u = loc.get("url_for_pdf")
            if u and u not in pdf_urls:
                pdf_urls.append(u)

        return {
            "doi": doi,
            "is_oa": bool(data.get("is_oa")),
            "oa_status": data.get("oa_status"),            # gold / green / hybrid / bronze / closed
            "best_pdf_url": pdf_urls[0] if pdf_urls else None,
            "pdf_urls": pdf_urls[:4],
            "landing_url": best.get("url_for_landing_page") or best.get("url"),
            "host_type": best.get("host_type"),            # publisher / repository
            "version": best.get("version"),                # publishedVersion / acceptedVersion / submittedVersion
            "license": best.get("license"),
            "journal_is_oa": data.get("journal_is_oa"),
        }

    def download_pdf(self, info: Dict[str, Any], out_dir: str | Path) -> Optional[Path]:
        """Try each OA PDF URL until one returns a real PDF. Returns the saved path."""
        if not info:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", info["doi"])[:120]
        out = out_dir / f"{safe}.pdf"
        if out.exists() and out.stat().st_size > 1000:
            return out
        for url in info.get("pdf_urls") or []:
            try:
                data = self.http.get_bytes(url, max_bytes=40 * 1024 * 1024)
            except HttpError:
                continue
            if data[:5] == b"%PDF-":
                out.write_bytes(data)
                return out
        return None
