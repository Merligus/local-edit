"""Downloading recipe weights and the engine binaries.

Nothing is bundled with this repository. The engine is a 46 MB prebuilt archive
and one recipe is 3.5-34 GB, so both are fetched on demand into the models
directory and the user pays only for the tiers they pick.

The rules are local-upscaler's, with one addition the scale forces:

* **Verify before installing.** A download that finishes is not a download that
  succeeded — a captive portal, a rate limit or a moved URL all return HTTP 200
  with an HTML body, and writing that to `flux-2-klein-4b-Q4_0.gguf` produces a
  file that fails minutes later inside ggml with an unreadable error. Weight
  files are checked against the exact byte counts in `catalog`, the engine
  archive against a SHA-256. A mismatch raises here, where the message can be
  useful.

* **Install atomically.** Everything lands in a `.part` beside the destination
  and is `os.replace`d into place only once it verifies, so an interrupted
  fetch can never leave a half-written file that looks present to `have`.

* **Resume.** New here, and not a nicety. The upscaler's largest single file was
  64 MB, where restarting costs seconds; `flux2-dev-q4` is a 19.8 GB file, and a
  dropped connection at 90% is half an hour thrown away. A surviving `.part` is
  continued with a `Range` request. The server's agreement is checked — a 200
  where 206 was asked for means the range was ignored, and appending to the
  `.part` would silently corrupt it — so a server that cannot resume restarts
  cleanly instead.

`urllib` rather than `requests`, following local-upscaler: the dependency policy
is system packages only, no venv (see docs/COMPATIBILITY.md), and the standard
library is entirely adequate for a GET.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from .. import paths
from . import binary
from .catalog import File, Recipe

#: `progress(done_bytes, total_bytes, label)`. `total` 0 when unknown.
ProgressCb = Callable[[int, int, str], None]
#: Returns True to abort an in-flight download.
CancelCb = Callable[[], bool]

_CHUNK = 1024 * 1024
_TIMEOUT = 60
_AGENT = "local-edit"

#: Executables inside the engine archive. Everything else in it is a shared
#: library that has to land beside them; see `binary.child_env`.
ENGINE_EXECUTABLES = (binary.CLI_NAME, binary.SERVER_NAME)


class FetchError(Exception):
    """A download failed, or arrived corrupt."""


class Cancelled(Exception):
    """The user cancelled an in-flight download."""


def _open(url: str, offset: int = 0):
    headers = {"User-Agent": _AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=_TIMEOUT)


def _download(url: str, dest: Path, expect_bytes: int | None = None,
              sha256: str | None = None, progress: ProgressCb | None = None,
              is_cancelled: CancelCb | None = None, label: str = "",
              base: int = 0, grand_total: int = 0) -> None:
    """Fetch `url` to `dest`, resuming and verifying.

    `base`/`grand_total` let a multi-file fetch report one continuous bar across
    all of its files rather than restarting at zero for each.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    # A checksummed download cannot resume: the digest has to cover every byte,
    # and the bytes already on disk were not hashed. Only the 46 MB engine
    # archive is checksummed, so this costs nothing worth having.
    offset = 0
    if sha256 is None and part.exists():
        try:
            offset = part.stat().st_size
        except OSError:
            offset = 0
        if expect_bytes is not None and offset >= expect_bytes:
            offset = 0                        # already-complete or overshot
    if offset == 0:
        part.unlink(missing_ok=True)

    digest = hashlib.sha256()
    got = offset
    try:
        resp = _open(url, offset)
        resumed = offset > 0 and resp.status == 206
        if offset and not resumed:
            # The server ignored the Range header and is sending the whole file
            # from byte zero. Appending would interleave two copies.
            got, offset = 0, 0
            part.unlink(missing_ok=True)
        with resp, part.open("ab" if resumed else "wb") as f:
            declared = int(resp.headers.get("Content-Length") or 0)
            total = grand_total or expect_bytes or (declared + offset)
            while True:
                if is_cancelled is not None and is_cancelled():
                    raise Cancelled(label or url)
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                if sha256 is not None:
                    digest.update(chunk)
                got += len(chunk)
                if progress is not None:
                    progress(base + got, total, label)
    except urllib.error.HTTPError as e:
        raise FetchError(f"could not download {dest.name}: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"could not download {dest.name}: {e.reason}") from e
    except OSError as e:
        raise FetchError(f"could not write {dest}: {e}") from e
    except Cancelled:
        raise                                  # keep the .part so it can resume
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    if expect_bytes is not None and got != expect_bytes:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"{dest.name}: expected {expect_bytes:,} bytes, got {got:,}. "
            f"The download was truncated, or the URL now serves something else.")
    if sha256 is not None and digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise FetchError(f"{dest.name}: checksum mismatch — the file is not "
                         f"what it should be.")
    os.replace(part, dest)


# ----------------------------------------------------------------- weights
def file_path(f: File, models_dir: Path) -> Path:
    return models_dir / f.subdir() / f.filename()


def recipe_paths(recipe: Recipe, models_dir: Path) -> list[Path]:
    return [file_path(f, models_dir) for f in recipe.files]


def missing(recipe: Recipe, models_dir: Path) -> list[File]:
    """Which of `recipe`'s files are absent or the wrong size.

    Size is checked, not just existence, so a file truncated by a full disk is
    re-downloaded rather than handed to ggml. A stale `.part` does not count as
    present — it has no bearing on whether the real file is there.
    """
    out = []
    for f in recipe.files:
        try:
            if file_path(f, models_dir).stat().st_size != f.size:
                out.append(f)
        except OSError:
            out.append(f)
    return out


def have(recipe: Recipe, models_dir: Path) -> bool:
    return not missing(recipe, models_dir)


def download_size(recipe: Recipe, models_dir: Path) -> int:
    """Bytes still to fetch, counting a resumable `.part` as already done."""
    total = 0
    for f in missing(recipe, models_dir):
        path = file_path(f, models_dir)
        part = path.with_name(path.name + ".part")
        try:
            done = part.stat().st_size
        except OSError:
            done = 0
        total += max(0, f.size - min(done, f.size))
    return total


def fetch_recipe(recipe: Recipe, models_dir: Path,
                 progress: ProgressCb | None = None,
                 is_cancelled: CancelCb | None = None) -> None:
    """Download whatever `recipe` is missing. A no-op when it is complete."""
    todo = missing(recipe, models_dir)
    if not todo:
        return
    total = sum(f.size for f in todo)
    base = 0
    for f in todo:
        _download(f.url(), file_path(f, models_dir), expect_bytes=f.size,
                  progress=progress, is_cancelled=is_cancelled,
                  label=f"Downloading {f.filename()}",
                  base=base, grand_total=total)
        base += f.size


# ------------------------------------------------------------------ engine
def fetch_engine(backend: str = binary.DEFAULT_BACKEND,
                 progress: ProgressCb | None = None,
                 is_cancelled: CancelCb | None = None) -> Path:
    """Download and unpack a pinned upstream release.

    Unlike local-upscaler, which extracted one self-contained executable and
    threw the rest away, **everything** in this archive is installed: `sd-cli`
    and `sd-server` are dynamically linked against the `libggml-*.so` files
    beside them and will not start without them.

    Members are filtered by basename rather than trusted as paths. A zip entry
    is free to contain `../` and a naive `extractall` would happily write
    outside the destination; there is no reason to accept a path from an archive
    when every file wanted here is a flat basename.
    """
    release = binary.RELEASES.get(backend)
    if release is None:
        raise FetchError(f"no prebuilt {backend} engine for Linux; build from "
                         f"source and set the engine path in Advanced")

    dest_dir = binary.managed_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = paths.cache_dir() / release.asset
    _download(release.url(), archive, expect_bytes=release.size,
              sha256=release.sha256, progress=progress,
              is_cancelled=is_cancelled,
              label=f"Downloading the {backend} engine")

    try:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name or name.startswith("."):
                    continue
                target = dest_dir / name
                part = target.with_name(target.name + ".part")
                with z.open(info) as src, part.open("wb") as f:
                    while chunk := src.read(_CHUNK):
                        f.write(chunk)
                if name in ENGINE_EXECUTABLES:
                    part.chmod(0o755)
                os.replace(part, target)
    except (zipfile.BadZipFile, OSError) as e:
        raise FetchError(f"could not unpack {archive.name}: {e}") from e
    finally:
        archive.unlink(missing_ok=True)        # nothing left to give

    exe = dest_dir / binary.CLI_NAME
    if not exe.is_file():
        raise FetchError(f"{release.asset} did not contain {binary.CLI_NAME}")
    return exe
