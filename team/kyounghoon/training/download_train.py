"""Download the organizer's public Train archive without extra dependencies.

Uses the file ID observed in the user-provided organizer Drive folder. It never
uses browser cookies, credentials, private Drive APIs, or a competition endpoint.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import http.cookiejar
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import zipfile

FILE_ID = "17wU8GTvSlZcMcTsYY6r0lsIhKBAcaAei"
EXPECTED_BYTES = 3893903540
SOURCE_FOLDER = "https://drive.google.com/drive/folders/1f7QFkEk1GqIB3Cv9VfothqaJiPXUuZUg"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.inside = True
            self.action = attrs.get("action")
        elif self.inside and tag == "input" and attrs.get("type") == "hidden":
            if attrs.get("name") in {"id", "export", "confirm", "uuid"}:
                self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


def open_public_download():
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": FILE_ID})
    response = opener.open(url, timeout=60)
    if "text/html" in response.headers.get("Content-Type", ""):
        page = response.read(1_000_001)
        response.close()
        if len(page) > 1_000_000:
            raise RuntimeError("Unexpectedly large Drive notice; download was not started.")
        form = DownloadForm()
        form.feed(page.decode("utf-8", "replace"))
        if (form.action != "https://drive.usercontent.google.com/download"
                or form.fields.get("id") != FILE_ID
                or form.fields.get("export") != "download"
                or form.fields.get("confirm") != "t"):
            raise RuntimeError("Public Drive download unavailable (access/quota/login/verification). "
                               "No authentication or verification bypass is attempted. "
                               "Use an existing mounted Drive copy via TRAIN_ZIP_OVERRIDE.")
        response = opener.open(form.action + "?" + urllib.parse.urlencode(form.fields), timeout=60)
    host = urllib.parse.urlsplit(response.geturl()).hostname or ""
    if not (host in {"drive.google.com", "drive.usercontent.google.com"}
            or host.endswith(".googleusercontent.com")):
        response.close()
        raise RuntimeError("Unexpected download host; stopped.")
    if "text/html" in response.headers.get("Content-Type", ""):
        response.close()
        raise RuntimeError("Drive returned a notice instead of a ZIP (possibly download quota). "
                           "Use an existing mounted Drive copy via TRAIN_ZIP_OVERRIDE.")
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/content/inthon-data/inthon2026_train.zip")
    parser.add_argument("--probe", action="store_true",
                        help="Read only ZIP signature and response headers; do not save the archive.")
    args = parser.parse_args()
    target = Path(args.output)
    if not args.probe and target.exists():
        if target.stat().st_size == EXPECTED_BYTES and zipfile.is_zipfile(target):
            print("Using existing ZIP; the contract step will verify its content identity:", target)
            return
        raise RuntimeError("Existing target is not the expected archive; refusing to overwrite it.")
    if not args.probe:
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < EXPECTED_BYTES + 1024**3:
            raise RuntimeError("At least the ZIP size plus 1 GiB of local free disk is required.")
    with open_public_download() as response:
        prefix = response.read(4)
        if prefix != b"PK\x03\x04":
            raise RuntimeError("Download response is not a ZIP archive.")
        print("Organizer ZIP response verified; source folder:", SOURCE_FOLDER, flush=True)
        if args.probe:
            print("Probe only: ZIP signature matched; full download was not performed.")
            return
        partial = target.with_name(target.name + ".part")
        count = len(prefix)
        checkpoint = 0
        with partial.open("wb") as destination:
            destination.write(prefix)
            while chunk := response.read(8 * 1024**2):
                destination.write(chunk)
                count += len(chunk)
                if count > EXPECTED_BYTES:
                    raise RuntimeError("Archive size exceeds the fixed source version.")
                if count - checkpoint >= 256 * 1024**2:
                    print(f"Downloaded {count / 1024**3:.2f} / {EXPECTED_BYTES / 1024**3:.2f} GiB", flush=True)
                    checkpoint = count
        if count != EXPECTED_BYTES or not zipfile.is_zipfile(partial):
            raise RuntimeError("Incomplete or changed source archive; final ZIP was not replaced.")
        partial.replace(target)
        print("Download complete:", target, "bytes:", count, flush=True)


if __name__ == "__main__":
    main()
