"""A log file on disk, so a failure can be read after it happens.

The app used to keep everything in memory: the parser held a 60-line tail, the
error dialog showed part of it, and pressing OK threw the rest away. That is
fine while someone is watching the screen and useless afterwards — a crash, a
dialog dismissed too quickly, or an intermittent failure left nothing to read.
Worse, the engine's own diagnosis of *why* it stopped is often fifty lines above
the four ERROR lines that reach the dialog, and those fifty lines were exactly
the ones being dropped.

So every run now writes to `~/.cache/local-edit/logs/local-edit.log`:

* what the app decided — chosen recipe, hardware probe, verdict, memory flags;
* the **exact argv** the engine was started with;
* every non-progress line the engine printed, prefixed `engine:`;
* every failure, with a traceback for the unexpected ones.

Progress lines are excluded on purpose. `Parser` tells them apart already, and a
four-step run at 1024 emits hundreds of `\\r`-rewritten bars that would bury the
one line that matters.

Under the *cache* root rather than data or state: it is diagnostic output, it is
rotated, and losing it costs nothing.

Rotation is size-based at 4 MB with three backups, which is around a thousand
runs. Time-based rotation would be wrong here — the interesting unit is "the
last few runs", not "today".
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from . import paths

LOG_NAME = "local-edit.log"
#: Rotate at this size, keeping `BACKUPS` older files beside it.
MAX_BYTES = 4_000_000
BACKUPS = 3

#: `LOCAL_EDIT_LOG=debug` turns on the noisy detail without a code change.
LEVEL_ENV = "LOCAL_EDIT_LOG"

_ROOT = "local_edit"
_configured: Path | None = None

# Attached at import, before anything can log. Without it a record emitted
# before `setup` — or in a headless test that never calls it — reaches
# `logging.lastResort`, which prints WARNING and above to stderr. That turned
# `_explain` into a duplicate error dump on the terminal, and would make the
# offline tests print engine failures they are deliberately provoking.
logging.getLogger(_ROOT).addHandler(logging.NullHandler())
logging.getLogger(_ROOT).propagate = False


def log_dir() -> Path:
    return paths.cache_dir() / "logs"


def log_file() -> Path:
    return log_dir() / LOG_NAME


def _level_from_env(default: int) -> int:
    """`LOCAL_EDIT_LOG=debug` and friends, ignoring anything unrecognised."""
    name = os.environ.get(LEVEL_ENV, "").strip().upper()
    value = logging.getLevelNamesMapping().get(name)
    return value if isinstance(value, int) else default


def setup(*, console: bool = False, level: int = logging.INFO) -> Path | None:
    """Start writing to the log file. Safe to call more than once.

    Returns the path, or `None` if the file could not be opened — a read-only
    or full home directory must not stop the app from running, which is why
    every failure here is swallowed rather than raised.
    """
    global _configured
    if _configured is not None:
        return _configured

    log = logging.getLogger(_ROOT)
    log.setLevel(_level_from_env(level))
    log.propagate = False

    path = log_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)-5s %(message)s"))
        log.addHandler(stream)

    _configured = path
    return path


def get(name: str) -> logging.Logger:
    """A logger under the app's root.

    Callable before `setup`: with no handler attached the records are simply
    dropped, so `engine` and `runner` can log unconditionally and the headless
    tests that never call `setup` stay silent.
    """
    return logging.getLogger(f"{_ROOT}.{name}")


#: One logger for everything the child process prints, so the engine's own
#: words are distinguishable from the app's at a glance.
_engine_log = logging.getLogger(f"{_ROOT}.engine.out")


def engine_line(line: str) -> None:
    """Record one line of engine output.

    Level comes from sd.cpp's own tag, so `LOCAL_EDIT_LOG=warning` leaves an
    error-only file — its DEBUG output alone is thousands of lines per run.
    """
    text = line.strip()
    if not text:
        return
    if text.startswith("[ERROR"):
        _engine_log.error("%s", text)
    elif text.startswith("[WARN"):
        _engine_log.warning("%s", text)
    elif text.startswith("[DEBUG") or text.startswith("[VERBOSE"):
        _engine_log.debug("%s", text)
    else:
        _engine_log.info("%s", text)


def tail(lines: int = 200, path: Path | None = None) -> str:
    """The last `lines` of the log, for `--log` and for a failure dialog."""
    target = path or log_file()
    try:
        with target.open(encoding="utf-8", errors="replace") as f:
            kept = f.readlines()[-lines:]
    except OSError:
        return ""
    return "".join(kept).rstrip()


def install_excepthook() -> None:
    """Send otherwise-unhandled exceptions to the log before they reach stderr.

    Qt swallows exceptions raised inside a slot — the app keeps running and the
    only trace is a line on a terminal nobody is watching, which is how a dead
    Generate button once went unexplained. Logging first means the traceback
    survives.
    """
    previous = sys.excepthook

    def hook(kind, value, tb) -> None:
        get("crash").error("unhandled %s", kind.__name__,
                           exc_info=(kind, value, tb))
        previous(kind, value, tb)

    sys.excepthook = hook


def session_header(**fields: object) -> None:
    """Mark the start of a session, with whatever context the caller has."""
    log = get("session")
    log.info("---- local-edit started ----")
    for key, value in fields.items():
        log.info("  %s: %s", key, value)
