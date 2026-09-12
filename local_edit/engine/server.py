"""Owning a warm `sd-server`, and reading its output for progress.

**Why a server at all.** local-upscaler starts its engine once per run and that
is right for it — a Real-ESRGAN model is 33 MB and loads in about four seconds.
Here the smallest recipe is 5.3 GB and the largest 34 GB, and loading is tens of
seconds at best. Editing a prompt is inherently iterative: you say "make them
ride a motorcycle", look at it, and say "…in the rain". Paying the model load on
every one of those turns the app into a batch system. So the weights are loaded
once into a child process that stays alive between edits, and a recipe change —
which sd.cpp cannot do in place, the model being fixed at startup — is the only
thing that restarts it.

**Why the output is read at all.** The job API reports `queued` / `generating` /
`completed` and nothing in between, so an honest progress bar has to come from
somewhere else. The app owns the process, so it can read what the process
prints. That is the same manoeuvre local-upscaler makes for the same reason,
and the two awkward parts are worth naming:

* The stream is **not line-oriented**. Progress is written with `\\r` and no
  newline until a pass ends, so `for line in proc.stdout` blocks for the entire
  run and then yields one enormous line. The reader here uses `read1` on the raw
  buffer and hands whatever arrives to `progress.Parser`, which does the
  splitting.
* stdout and stderr are **merged** with `stderr=STDOUT`. Progress goes to
  stdout and logs to stderr, and the parser needs both in the order they were
  written — a stage marker that arrives after the steps it should precede would
  send the UI backwards.

**Lifecycle.** Idle servers are shut down after `idle_timeout`, because a
resident Klein holds about 2.5 GB and this machine has 3.4 GiB of VRAM in total;
leaving it parked would make every other GPU application on the desktop stutter
for as long as the app is open.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import binary, client, progress
from .catalog import Recipe

#: How long to wait for `capabilities` to answer, on top of the recipe's own
#: load prior. Generous because a disk-streamed recipe reads gigabytes before
#: it will answer anything.
READY_GRACE_S = 180.0
#: Gap between readiness polls while the model loads.
READY_POLL_S = 0.5
#: Gap between job polls. The job takes minutes; a tighter loop buys nothing and
#: the real progress signal is the output tap, not this.
JOB_POLL_S = 0.4
#: Give the process this long to exit on its own before SIGKILL.
STOP_GRACE_S = 10.0


class ServerError(Exception):
    """The engine process could not be started, or died."""


def free_port() -> int:
    """An unused loopback port.

    Asking the OS and closing beats picking a constant: 1234 is sd.cpp's default
    and is exactly the port another copy of this app, or a hand-started server,
    would already be on.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EngineServer:
    """One `sd-server` process holding one recipe's weights."""

    def __init__(self, idle_timeout: float = 600.0) -> None:
        self.idle_timeout = idle_timeout
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._client: client.Client | None = None
        self._key: tuple = ()
        self._last_used = 0.0
        self._lock = threading.RLock()
        self._parser = progress.Parser()
        #: Set when the reader thread sees the process end.
        self._ended = threading.Event()

    # -- state ------------------------------------------------------------
    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def client_(self) -> client.Client | None:
        return self._client

    def key_for(self, recipe: Recipe, memory: tuple[str, ...],
                backend: str, threads: int, max_vram_gb: float) -> tuple:
        """Everything that, if changed, requires a restart.

        The memory flags are in the key because they are startup flags: a
        recipe that was graded `FITS` at 512 and `OFFLOAD` at 1024 needs a
        different `--offload-to-cpu` than the running process was given.
        """
        return (recipe.id, memory, backend, threads, round(max_vram_gb, 2))

    def serves(self, key: tuple) -> bool:
        return self.running and self._key == key

    def idle_expired(self) -> bool:
        return (self.running and self.idle_timeout > 0
                and self._last_used > 0
                and time.monotonic() - self._last_used > self.idle_timeout)

    def log_tail(self, n: int = 8) -> str:
        return self._parser.tail(n)

    def set_load_tap(self, on_load: Callable[[int, int], None] | None) -> None:
        """Route weight-loading progress to a caller, or stop routing it.

        Separate from `generate`'s `on_tick` because the two happen at different
        times and mean different things: loading runs before the first edit (and
        again mid-run when weights stream from disk), while ticks are sampling
        steps. A caller that conflated them would show a bar jumping between two
        unrelated scales.
        """
        self._parser.on_load = on_load

    # -- lifecycle --------------------------------------------------------
    def ensure(self, recipe: Recipe, models_dir: Path, *,
               memory: tuple[str, ...] = (), backend: str = binary.DEFAULT_BACKEND,
               threads: int = 0, max_vram_gb: float = 0.0,
               engine_path: str | None = None,
               on_stage: Callable[[str, str], None] | None = None,
               is_cancelled: Callable[[], bool] | None = None) -> client.Client:
        """Return a client for a server holding `recipe`, starting one if needed."""
        with self._lock:
            key = self.key_for(recipe, memory, backend, threads, max_vram_gb)
            if self.serves(key):
                self._last_used = time.monotonic()
                assert self._client is not None
                return self._client
            self.stop()
            return self._start(recipe, models_dir, key, memory=memory,
                               backend=backend, threads=threads,
                               max_vram_gb=max_vram_gb, engine_path=engine_path,
                               on_stage=on_stage, is_cancelled=is_cancelled)

    def _start(self, recipe: Recipe, models_dir: Path, key: tuple, *,
               memory: tuple[str, ...], backend: str, threads: int,
               max_vram_gb: float, engine_path: str | None,
               on_stage: Callable[[str, str], None] | None,
               is_cancelled: Callable[[], bool] | None) -> client.Client:
        exe = binary.find_server(engine_path)
        if exe is None:
            raise ServerError(
                f"{binary.SERVER_NAME} was not found. Fetch a copy with:\n"
                f"    python3 -m local_edit --fetch-engine")

        port = free_port()
        argv = binary.build_server_argv(
            exe, recipe, models_dir, port=port, memory=memory, backend=backend,
            threads=threads, max_vram_gb=max_vram_gb)

        self._parser = progress.Parser(on_stage=on_stage)
        self._ended.clear()
        try:
            self._proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=binary.child_env(exe))
        except OSError as e:
            raise ServerError(f"could not start {exe.name}: {e}") from e

        self._reader = threading.Thread(target=self._pump, name="sd-server-out",
                                        daemon=True)
        self._reader.start()
        self._client = client.Client("127.0.0.1", port)
        self._key = key
        self._await_ready(recipe, is_cancelled)
        self._last_used = time.monotonic()
        return self._client

    def _await_ready(self, recipe: Recipe,
                     is_cancelled: Callable[[], bool] | None) -> None:
        """Block until `capabilities` answers, the process dies, or we give up."""
        assert self._client is not None
        deadline = time.monotonic() + recipe.load_s * 4 + READY_GRACE_S
        while time.monotonic() < deadline:
            if is_cancelled is not None and is_cancelled():
                self.stop()
                raise ServerError("cancelled while the model was loading")
            if not self.running:
                tail = self.log_tail()
                self.stop()
                raise ServerError(
                    "the engine exited while loading the model."
                    + (f"\n{tail}" if tail else ""))
            if self._client.alive(timeout=2.0):
                return
            time.sleep(READY_POLL_S)
        tail = self.log_tail()
        self.stop()
        raise ServerError(
            f"the engine did not finish loading {recipe.label} in time."
            + (f"\n{tail}" if tail else ""))

    def _pump(self) -> None:
        """Feed the child's merged output to the parser until it ends.

        `read1` returns whatever is available without waiting for a newline,
        which is what makes a `\\r`-terminated progress bar visible at all.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                chunk = proc.stdout.read1(8192)
                if not chunk:
                    break
                self._parser.feed(chunk.decode("utf-8", "replace"))
        except (OSError, ValueError):
            pass
        finally:
            self._parser.flush()
            self._ended.set()

    def stop(self) -> None:
        """Shut the process down. Safe to call when nothing is running."""
        with self._lock:
            proc, self._proc = self._proc, None
            self._client = None
            self._key = ()
            self._last_used = 0.0
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                # A server mid-generation ignores SIGTERM until the current ggml
                # graph finishes, which for a disk-streamed recipe is minutes.
                # The user asked for the VRAM back; take it.
                proc.kill()
                try:
                    proc.wait(timeout=STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
            except OSError:
                pass
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        reader, self._reader = self._reader, None
        if reader is not None and reader.is_alive():
            reader.join(timeout=2.0)

    def stop_if_idle(self) -> bool:
        if self.idle_expired():
            self.stop()
            return True
        return False

    # -- running one edit -------------------------------------------------
    def generate(self, api: client.Client, request: client.ImgGenRequest, *,
                 on_tick: Callable[[progress.Tick], None] | None = None,
                 is_cancelled: Callable[[], bool] | None = None) -> list[bytes]:
        """Submit one edit and wait for it, reporting progress from the tap."""
        self._parser.reset()
        self._parser.on_tick = on_tick
        try:
            job_id = api.submit(request)
            while True:
                if is_cancelled is not None and is_cancelled():
                    api.cancel(job_id)
                    raise client.ApiError("cancelled")
                if not self.running:
                    raise ServerError(
                        "the engine stopped during the edit — it most likely ran "
                        "out of memory.\n" + self.log_tail())
                job = api.job(job_id)
                status = str(job.get("status") or "")
                if status == client.STATUS_COMPLETED:
                    self._last_used = time.monotonic()
                    return client.images_from_job(job)
                if status == client.STATUS_FAILED:
                    raise client.ApiError(client.job_error(job))
                if status == client.STATUS_CANCELLED:
                    raise client.ApiError("cancelled")
                time.sleep(JOB_POLL_S)
        finally:
            self._parser.on_tick = None
            self._last_used = time.monotonic()
