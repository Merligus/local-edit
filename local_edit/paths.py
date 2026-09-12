"""XDG locations and atomic JSON persistence.

Adapted from local-upscaler's `paths.py`, which took it from soundboard, and for
the same reason: every write goes through `write_json`, which writes a temporary
file in the same directory and then `os.replace`s it into place. `os.replace` is
atomic on POSIX, so a crash or a full disk mid-write leaves the previous file
intact rather than a truncated one. Readers tolerate a missing or corrupt file by
returning a default, so a bad settings file degrades to "defaults" instead of a
crash at startup.

Two differences from the upscaler are worth naming:

* `models_dir()` takes an **override**. The upscaler's models are 1-64 MB and
  live wherever XDG says; here one recipe is 5-30 GB, and the development machine
  has 33 GB free on `/` against 908 GB on a second disk. A user who cannot fit
  the catalog in `~` must be able to point the app at a volume that can hold it,
  so the location is a setting rather than a constant. The upscaler's hard
  requirement that the path contain the literal component `models` does **not**
  apply — that was an ncnn quirk; sd.cpp takes explicit file paths.

* `engine_dir()` holds a *directory* of files, not one executable. The
  stable-diffusion.cpp release ships `sd-cli`, `sd-server` and around twenty
  shared objects that must sit beside them; see `engine.binary.child_env`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

APP = "local-edit"


def _xdg(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return (Path(v) if v else default) / APP


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config")


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", Path.home() / ".local/share")


def cache_dir() -> Path:
    return _xdg("XDG_CACHE_HOME", Path.home() / ".cache")


def default_models_dir() -> Path:
    return data_dir() / "models"


def models_dir(override: str | None = None) -> Path:
    """Where recipe weights live.

    `override` is `settings.models_dir`; empty means "use the XDG default".
    Expanded here rather than at the call sites so a settings file containing
    `~/big-disk/models` works the same as an absolute path.
    """
    if override:
        return Path(override).expanduser()
    return default_models_dir()


def engine_dir() -> Path:
    """Where `--fetch-engine` unpacks sd-cli, sd-server and their libraries."""
    return data_dir() / "engine"


def work_dir() -> Path:
    """Scratch root: reference images staged for the CLI, filmstrip frames.

    Under the *cache* root, not data. Everything here is reproducible and is
    safe to lose at any time.
    """
    return cache_dir() / "work"


def desktop_file() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(base) / "applications" / f"{APP}.desktop"


def settings_file() -> Path:
    return config_dir() / "settings.json"


def write_json(path: Path, payload: dict) -> None:
    """Atomically replace `path` with `payload`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)      # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: dict | None = None) -> dict:
    """Read JSON, returning `default` for anything unreadable."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else (default or {})
    except (OSError, json.JSONDecodeError, ValueError):
        return default or {}


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """A free path like `<stem><suffix>`, `<stem>-2<suffix>`, ..."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate
