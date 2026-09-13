"""Turning sd.cpp's console output into a determinate progress bar.

local-upscaler faced the same problem and solved it the hard way: its engine
printed nothing at all during a run, so the image had to be cut into tiles and
the per-file `done` lines counted. Here the engine is more forthcoming — it
prints a real step counter — but the *reason* the parsing lives in its own
module is the same one, and there is a second reason on top of it.

**The server's job API does not report progress.** `GET /sdcpp/v1/jobs/{id}`
returns `queued` / `generating` / `completed` and a `queue_position`, and
nothing between 0% and 100%. A four-step Klein run is a couple of minutes and a
Kontext run is the best part of an hour; an indeterminate bar for that long is
not an interface. So progress comes from the process's own output, which the app
can read because it *owns* the server process, and the same parser serves the
one-shot CLI path. One parser, two engines.

**The output is not line-oriented.** `print_progress_line` in sd.cpp is:

    printf("\\r%s %i/%i - %s\\033[K%s", bar, step, steps, speed, lf)

— a carriage return, the bar, the counter, an erase-to-end-of-line, and a
newline only when the pass finishes. A reader that splits on `\\n` therefore
sees nothing for the entire run and then one enormous line at the end. `feed`
splits on `\\r` as well, and strips the CSI sequences, which is most of why this
is a module rather than one regex at a call site.

A run also contains **several** counted passes — text encoding, sampling, and
VAE decode each drive the same bar from 1 to N. Only one of them is worth
showing, so `Parser` tracks which pass it is in from the log lines and reports
the sampling pass as the progress, leaving the others to set the stage text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import log as applog

#: `  |=========>              | 7/20 - 1.23s/it`
#: Anchored on the ` - <rate>` tail so an ordinary `3/4` in a log message — a
#: file path, a shard count, a tensor shape — cannot fake a step.
_STEP_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\s+-\s+([\d.]+)\s*(s/it|it/s)\b")

#: `  |####      | 149/298 - 312.81MB/s` — the *weight loading* bar, which uses
#: the same widget with a throughput unit instead of a rate. Worth parsing
#: separately rather than ignoring: loading Klein is half a minute and loading a
#: disk-streamed recipe is minutes, and an app that shows nothing for that long
#: looks hung. Reported as tensor counts, which is what the engine counts.
_LOAD_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\s+-\s+([\d.]+)\s*[KMG]?B/s\b")

#: ANSI CSI sequences. sd.cpp emits `\033[K` on every progress line, and `--color`
#: adds SGR colour runs around the log tags.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

#: sd.cpp's log prefix: `[INFO ] stable-diffusion.cpp:1234 - message`
_LOG_RE = re.compile(r"^\[(DEBUG|INFO|WARN|ERROR)\s*\]\s*(?:[\w.\-]+:\d+\s*-\s*)?(.*)$")

STAGE_LOAD = "load"
STAGE_ENCODE = "encode"
STAGE_SAMPLE = "sample"
STAGE_DECODE = "decode"

#: Substrings that mark the start of a stage, most specific first. Matched
#: against the lowercased message body of a log line, and taken verbatim from
#: a real run rather than from the docs:
#:
#:     model_loader.cpp - loading 149/149 tensors from flux-2-klein-4b-Q4_0.gguf
#:     conditioner.hpp  - computing condition graph completed, taking 19137 ms
#:     image.cpp        - get_learned_condition completed, taking 19.14s
#:     image.cpp        - generating image: 1/1 - seed 42
#:     image.cpp        - decode_first_stage completed
#:
#: Deliberately narrow. An earlier draft matched the bare word "latent", which
#: appears in setup lines long before decoding — and because stages only move
#: forward, one early match would have pinned the UI on "Saving…" for the whole
#: run.
_STAGE_MARKERS = (
    ("decode_first_stage", STAGE_DECODE),
    ("decoding image", STAGE_DECODE),
    ("generating image", STAGE_SAMPLE),
    ("sampling completed", STAGE_SAMPLE),
    ("get_learned_condition", STAGE_ENCODE),
    ("computing condition", STAGE_ENCODE),
    ("loading tensors", STAGE_LOAD),
    ("loading ", STAGE_LOAD),
)

#: How many of the engine's own lines to keep for a failure message. Its
#: capability dump alone is over a hundred lines, so the tail is what matters.
LOG_TAIL = 60


@dataclass
class Tick:
    """One parsed progress update."""

    stage: str
    step: int
    steps: int
    #: Seconds per step, normalised however the engine phrased it.
    sec_per_step: float


@dataclass
class Parser:
    """Feed it raw engine output; it calls back on every step and stage change."""

    on_tick: Callable[[Tick], None] | None = None
    #: Weight-loading progress, as (tensors_done, tensors_total).
    on_load: Callable[[int, int], None] | None = None
    on_stage: Callable[[str, str], None] | None = None
    #: Retained engine output, for explaining a failure.
    log: list[str] = field(default_factory=list)

    stage: str = STAGE_LOAD
    _pending: str = ""

    def feed(self, chunk: str) -> None:
        """Consume a chunk of output. Boundaries need not align with lines."""
        if not chunk:
            return
        text = self._pending + chunk
        # Split on either terminator; see the module docstring on why `\r`
        # matters more than `\n` here.
        parts = re.split(r"[\r\n]", text)
        self._pending = parts.pop() if not text.endswith(("\r", "\n")) else ""
        for part in parts:
            self.feed_line(part)

    def flush(self) -> None:
        """Process whatever is left when the stream ends without a terminator."""
        if self._pending:
            self.feed_line(self._pending)
            self._pending = ""

    def feed_line(self, raw: str) -> None:
        line = _ANSI_RE.sub("", raw).strip()
        if not line:
            return

        m = _LOAD_RE.search(line)
        if m is not None:
            if self.on_load is not None:
                self.on_load(int(m.group(1)), int(m.group(2)))
            return

        m = _STEP_RE.search(line)
        if m is not None:
            step, steps = int(m.group(1)), int(m.group(2))
            rate = float(m.group(3))
            if m.group(4) == "it/s":
                rate = 1.0 / rate if rate else 0.0
            if steps > 0 and self.on_tick is not None:
                self.on_tick(Tick(self.stage, min(step, steps), steps, rate))
            return

        self._note(line)

    def _note(self, line: str) -> None:
        """Record a log line, write it to disk, and let it move the stage.

        The in-memory `log` is a bounded tail — it exists to explain the failure
        that is happening *now* — so the engine's reasoning scrolls out of it
        within seconds. `applog.engine_line` is the durable copy, and it is
        called from here rather than from the two readers so that both the
        server and the one-shot CLI get it without either having to remember.
        Progress bars never reach this method, which is what keeps the file
        readable.
        """
        applog.engine_line(line)
        if len(self.log) >= LOG_TAIL:
            del self.log[0]
        self.log.append(line)

        m = _LOG_RE.match(line)
        body = (m.group(2) if m else line).lower()
        for needle, stage in _STAGE_MARKERS:
            if needle in body:
                self._set_stage(stage, line)
                return

    def _set_stage(self, stage: str, line: str) -> None:
        # Stages only ever move forward within one generation. Without this, a
        # stray "loading" in a debug line during sampling would send the UI back
        # to "Loading model…" with the bar half full.
        order = (STAGE_LOAD, STAGE_ENCODE, STAGE_SAMPLE, STAGE_DECODE)
        try:
            if order.index(stage) <= order.index(self.stage):
                return
        except ValueError:
            return
        self.stage = stage
        if self.on_stage is not None:
            self.on_stage(stage, line)

    def reset(self) -> None:
        """Start a new generation on the same (long-lived) server process."""
        self.stage = STAGE_LOAD
        self._pending = ""

    def tail(self, n: int = 6) -> str:
        return "\n".join(self.log[-n:])
