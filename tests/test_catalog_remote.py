"""Checks every weight file's URL and size against HuggingFace. Needs network.

Run:  python3 tests/test_catalog_remote.py

Separate from `test_catalog.py` because it is the only test that needs the
internet, and it is worth the wait for a reason local-upscaler learned the hard
way: `fetch` refuses any file that arrives at the wrong length, so a byte count
that is wrong in the catalog makes that model **permanently un-downloadable**,
and no offline test can tell. A moved file, a repo renamed, or an upstream
re-upload all look identical from here until someone tries to use it.

Uses the HuggingFace API rather than a HEAD per file: one request per repo
returns every file's size, which is both faster and kinder to the host.
"""

import json
import urllib.error
import urllib.request
from functools import lru_cache

from _harness import check, run                                    # noqa: E402
from local_edit.engine import binary, catalog as cat               # noqa: E402

TIMEOUT = 60


@lru_cache(maxsize=None)
def listing(repo):
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "local-edit-tests"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    return {f["rfilename"]: f.get("size") or 0
            for f in data.get("siblings", [])}


def test_every_weight_file_exists_at_the_declared_size():
    print("\nweight files")
    for f in sorted(cat.all_files(), key=lambda x: x.repo):
        try:
            files = listing(f.repo)
        except (urllib.error.URLError, OSError, ValueError) as e:
            check(False, f"{f.repo} is reachable ({e})")
            continue
        actual = files.get(f.path)
        if actual is None:
            check(False, f"{f.repo}::{f.path} still exists "
                         f"(the catalog's size is {f.size:,})")
            continue
        check(actual == f.size,
              f"{f.filename()} is {f.size:,} bytes"
              + ("" if actual == f.size else f" — upstream now says {actual:,}"))


def test_engine_releases_exist():
    print("\nengine releases")
    for name, release in binary.RELEASES.items():
        req = urllib.request.Request(release.url(), method="HEAD",
                                     headers={"User-Agent": "local-edit-tests"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                length = int(r.headers.get("Content-Length") or 0)
                final = r.url
        except (urllib.error.URLError, OSError, ValueError) as e:
            check(False, f"the {name} release is reachable ({e})")
            continue
        # GitHub redirects release assets to a CDN, which is fine; what matters
        # is that something of the right size is at the end of it.
        check(length == release.size,
              f"the {name} release is {release.size:,} bytes"
              + ("" if length == release.size
                 else f" — got {length:,} from {final}"))


if __name__ == "__main__":
    raise SystemExit(run(test_every_weight_file_exists_at_the_declared_size,
                         test_engine_releases_exist))
