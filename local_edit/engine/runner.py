"""Running one edit, start to finish, with honest progress.

Deliberately free of Qt, like local-upscaler's runner: the UI wraps this in a
worker thread and forwards its callbacks to signals, and keeping the
orchestration plain Python means `--bench` and the tests can drive it with no
display.

The shape of a run:

    fetch missing weights  ->  compose the prompt from the role slots  ->
    stage the images  ->  hand them to a warm server (or a one-shot CLI)  ->
    decode the PNG that comes back

**The server is not owned here.** `EngineServer` is passed in, because its whole
value is outliving any one run — a `Runner` per edit that started its own server
would reload gigabytes of weights on every prompt tweak, which is the exact cost
the server exists to avoid. `MainWindow` owns one and hands it to each run.

**The seed is chosen here, not by the engine.** sd.cpp accepts -1 for "random"
and prints what it picked, but parsing it back out of the log to tell the user
what to reuse is fragile for no reason. Drawing it in Python means the result
always knows its own seed, and "run that again with one word changed" actually
reproduces.
"""

from __future__ import annotations

import io
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .. import paths
from . import binary, client, fetch, hardware, progress, prompt, server
from .catalog import Recipe

#: `on_stage(key, human_text)`.
StageCb = Callable[[str, str], None]
#: `on_progress(done, total)` within the current stage. `total` 0 means unknown.
ProgressCb = Callable[[int, int], None]

STAGE_DOWNLOAD = "download"
STAGE_LOAD = "load"
STAGE_ENCODE = "encode"
STAGE_GENERATE = "generate"
STAGE_DECODE = "decode"

#: `progress.Parser` stage -> this module's, which adds a download stage in
#: front of everything the engine knows about.
_ENGINE_STAGES = {
    progress.STAGE_LOAD: STAGE_LOAD,
    progress.STAGE_ENCODE: STAGE_ENCODE,
    progress.STAGE_SAMPLE: STAGE_GENERATE,
    progress.STAGE_DECODE: STAGE_DECODE,
}

#: Substrings that identify a failure worth explaining rather than dumping.
#: Same idea as local-upscaler's `_HINTS`, with this engine's vocabulary.
_HINTS = (
    # Verbatim from a real out-of-VRAM run at 1024x1024 on a 4 GB card:
    #   ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
    #   alloc_tensor_range: failed to allocate Vulkan0 buffer of size 56623104
    #   flux segment 15/27 (flux.single_blocks.8) failed during execution
    ("erroroutofdevicememory", "The GPU ran out of memory at this size. Lower "
                               "the output size in Advanced — the model list "
                               "says what each one needs."),
    ("failed to allocate", "The GPU ran out of memory at this size. Lower the "
                           "output size in Advanced, or pick a lighter model."),
    ("out of memory", "Out of memory. Lower the output size in Advanced, or pick "
                      "a lighter model — the model list says which ones fit."),
    ("vk_error_out_of_device_memory", "The GPU ran out of memory. Lower the output "
                                      "size in Advanced, or pick a lighter model."),
    ("vkallocatememory", "The GPU ran out of memory. Lower the output size in "
                         "Advanced, or pick a lighter model."),
    ("cannot allocate memory", "The machine ran out of RAM. Close something, or "
                               "pick a model the list says fits."),
    ("no vulkan device", "No Vulkan device was found. Switch the backend to CPU "
                         "in Advanced, or check that your GPU driver is installed."),
    ("ggml_vulkan:", "The Vulkan backend failed to start. Switch the backend to "
                     "CPU in Advanced."),
    ("failed to load model", "The engine could not read a weight file. Re-fetch "
                             "it with --fetch-models."),
    ("unsupported", "The engine does not support this combination of model files. "
                    "Re-fetch the recipe with --fetch-models."),
)


class EditError(Exception):
    """The engine could not complete the edit."""


class Cancelled(Exception):
    """The user stopped the run."""


@dataclass
class Job:
    """Everything one edit needs to know."""

    recipe: Recipe
    #: The image being edited. None makes this a pure text-to-image run.
    source: Path | None = None
    references: tuple[prompt.Reference, ...] = ()
    #: What the user typed.
    instruction: str = ""
    #: Set to send this verbatim instead of composing from the role slots.
    prompt_override: str = ""
    negative_prompt: str = ""

    width: int = 1024
    height: int = 1024
    #: -1 draws a fresh one; `Result.seed` always reports what was used.
    seed: int = -1
    steps: int | None = None
    cfg_scale: float | None = None
    strength: float | None = None
    ip_adapter_strength: float = 1.0

    models_dir: Path = field(default_factory=paths.default_models_dir)
    #: From `hardware.Verdict.flags`.
    memory: tuple[str, ...] = ()
    backend: str = binary.DEFAULT_BACKEND
    threads: int = 0
    max_vram_gb: float = 0.0
    engine_path: str | None = None
    extra_args: tuple[str, ...] = ()

    def effective_steps(self) -> int:
        return self.steps if self.steps is not None else self.recipe.steps

    def all_references(self) -> list[prompt.Reference]:
        """References as the model will see them, source first.

        For an edit model the image being edited *is* reference 1 — that is how
        Kontext, FLUX.2 and Qwen-Image-Edit were trained — so it is prepended
        here and the user's own references start at 2. `prompt.compose` is
        handed this same list, which is what keeps the numbers in the prompt and
        the order of `--ref-image` in step with each other.
        """
        refs = list(self.references)
        if self.source is not None and self.recipe.source_as_ref:
            refs.insert(0, prompt.Reference(self.source, prompt.ROLE_PLAIN))
        return refs


@dataclass
class Result:
    image: Image.Image
    #: What was actually sent, after role composition and any trimming.
    prompt: str
    seed: int
    steps: int
    elapsed: float
    #: Measured seconds per sampling step, the figure calibration stores.
    sec_per_step: float
    size: tuple[int, int]
    #: Non-fatal things the user should know — a trimmed reference list, a
    #: missing PhotoMaker trigger word.
    notes: tuple[str, ...] = ()
    log_tail: str = field(default="", repr=False)


def _explain(log: str, returncode: int | None = None) -> str:
    low = log.lower()
    for needle, hint in _HINTS:
        if needle in low:
            return hint
    tail = [ln for ln in log.strip().splitlines() if ln.strip()][-4:]
    head = ("The engine failed." if returncode is None
            else f"The engine exited with code {returncode}.")
    return head + ("\n" + "\n".join(tail) if tail else "")


def round_to(value: int, multiple: int) -> int:
    """Nearest multiple, never zero.

    Latent patching means an off-grid size is silently rounded by the engine,
    which then returns an image that is not the size the UI promised. Rounding
    here keeps the promise and the result in step.
    """
    if multiple <= 1:
        return max(1, value)
    return max(multiple, int(round(value / multiple)) * multiple)


def plan_size(recipe: Recipe, source_size: tuple[int, int] | None,
              longest: int | None) -> tuple[int, int]:
    """Output size, either matching the source's aspect or square.

    `longest` caps the long edge; None means "use the source's own size".
    """
    if source_size and source_size[0] > 0 and source_size[1] > 0:
        w, h = source_size
        if longest:
            scale = longest / max(w, h)
            w, h = w * scale, h * scale
        return (round_to(int(w), recipe.size_multiple),
                round_to(int(h), recipe.size_multiple))
    side = round_to(longest or recipe.default_size, recipe.size_multiple)
    return side, side


#: How much one reference image adds to the cost of a sampling step, as a
#: fraction of a bare step. **Measured**, not guessed: FLUX.2 Klein 4B at
#: 512x512 on the development machine samples at 22.3 s/step with no
#: references and 55.4 s/step with two — a factor of 2.48, or about 0.75 per
#: reference.
#:
#: The mechanism is that these models concatenate each reference's latent
#: tokens onto the sequence the transformer attends over, so a reference is not
#: a cheap side input; it is more image to process. Leaving this out made the
#: estimate wrong by more than a factor of two for the app's central use case,
#: which is the one case where nobody would forgive it.
REF_STEP_COST = 0.75

#: Exponent relating sampling time to pixel count. **Measured**, and it is
#: below 1.0, which is the opposite of what the theory suggests.
#:
#: FLUX.2 Klein 4B on a GTX 1050 Ti: 22.3 s/step at 512x512, 76.0 s/step at
#: 1024x1024. Four times the pixels for 3.4 times the time — an exponent of
#: 0.88. Attention is quadratic in tokens, so this should have been *above* 1;
#: the reason it is not is that 512x512 does not keep a GPU busy. At 1024 tokens
#: the kernels are too small to fill the card and most of the step is launch
#: overhead and idle shader cores, so the extra pixels are close to free until
#: the quadratic term catches up. A bigger card would move this back above 1.
#:
#: This is exactly the kind of number that has no business being guessed, and it
#: was guessed at 1.4 before it was measured — which made the estimate for a
#: 1024px run nearly twice the truth.
PIXEL_EXPONENT = 0.9


def step_seconds(recipe: Recipe, width: int, height: int,
                 references: int = 0, sec_per_step: float | None = None,
                 grade: str = hardware.FITS) -> float:
    """Seconds for one sampling step, measured if known and modelled if not."""
    ref_factor = 1.0 + REF_STEP_COST * max(0, references)
    if sec_per_step and sec_per_step > 0:
        # A measurement already carries this size and grade; only the reference
        # count can differ from the run it came from.
        return sec_per_step * ref_factor
    megapixels = max(0.05, (width * height) / 1e6)
    return (recipe.sec_per_mpx_step * megapixels ** PIXEL_EXPONENT
            * hardware.speed_penalty(grade) * ref_factor)


def estimate_seconds(recipe: Recipe, width: int, height: int, steps: int,
                     sec_per_step: float | None = None,
                     grade: str = hardware.FITS,
                     include_load: bool = True,
                     references: int = 0) -> float:
    """How long a run should take, for the pre-run estimate and the ETA."""
    rate = step_seconds(recipe, width, height, references, sec_per_step, grade)
    return (recipe.load_s if include_load else 0.0) + rate * steps


def throughput(engine_rate: float, elapsed: float, steps_elapsed: int) -> float:
    """Seconds per sampling step, preferring the engine's own figure.

    Every progress line the engine prints carries a rate — `4/4 - 22.34s/it` —
    which `progress.Parser` already extracts. That number is measured inside the
    sampler loop, so it excludes weight loading, text encoding and VAE decode
    for free. Nothing this app can time from outside will beat it, and two
    attempts at timing it from outside both came out wrong:

    * Dividing total elapsed by steps counts the fixed costs as sampling. On
      this machine that is 11 s of weight loading and 19 s of text encoding
      before the first step — nearly double the real rate for a four-step Klein
      run, and it would then predict nearly double for a 24-step Kontext run
      where the same costs are amortised six times over.
    * Timing between the first and last tick is better but still wrong, because
      the clock can only start when the first tick *arrives* — so step 1 falls
      outside the window — and because the engine sometimes emits the last two
      ticks together. A real run whose ticks landed at 136.7, 193.1, 249.9 and
      249.9 gives 28.3 s/step by one reckoning and 37.7 by the other, against an
      engine-reported 56.

    So the wall-clock path is only a fallback for a run that produced no
    parseable rate at all.
    """
    if engine_rate and engine_rate > 0:
        return engine_rate
    return max(0.01, elapsed) / max(1, steps_elapsed)


class Runner:
    """Runs one `Job`. Not reusable; make a new one per edit."""

    def __init__(self, job: Job, engine: server.EngineServer,
                 on_stage: StageCb | None = None,
                 on_progress: ProgressCb | None = None) -> None:
        self.job = job
        self.engine = engine
        self._on_stage = on_stage
        self._on_progress = on_progress
        self._cancelled = False
        self._proc: subprocess.Popen | None = None
        self._notes: list[str] = []
        self._sampling_started = 0.0
        self._sampling_elapsed = 0.0
        self._steps_seen = 0
        #: The step number the clock started on, and the last one seen. Their
        #: difference is how many steps the measured interval actually covers.
        #: Only used when the engine reported no rate of its own.
        self._first_step = 0
        self._last_step = 0
        #: The engine's own seconds-per-step, from the last progress line.
        self._engine_rate = 0.0
        self._loading = False

    # -- control ----------------------------------------------------------
    def cancel(self) -> None:
        """Ask the run to stop. Safe to call from the GUI thread mid-run."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _check(self) -> None:
        if self._cancelled:
            raise Cancelled()

    def _stage(self, key: str, text: str) -> None:
        if self._on_stage is not None:
            self._on_stage(key, text)

    def _progress(self, done: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total)

    # -- callbacks from the engine tap ------------------------------------
    def _engine_stage(self, stage: str, _line: str) -> None:
        key = _ENGINE_STAGES.get(stage)
        if key is None:
            return
        if key == STAGE_GENERATE:
            # Not the start of sampling — see `_engine_load`. The engine logs
            # this before it loads the diffusion weights, so the clock that
            # `throughput` divides by must start at the first actual step.
            if self._loading:
                return
            self._sampling_started = time.monotonic()
        elif key == STAGE_DECODE and self._sampling_started:
            self._sampling_elapsed = time.monotonic() - self._sampling_started
        self._stage(key, self._stage_text(key))

    def _stage_text(self, key: str) -> str:
        label = self.job.recipe.label
        return {
            STAGE_LOAD: f"Loading {label}…",
            STAGE_ENCODE: "Reading the prompt…",
            STAGE_GENERATE: f"Generating with {label}…",
            STAGE_DECODE: "Finishing the image…",
        }.get(key, "Working…")

    def _engine_tick(self, tick: progress.Tick) -> None:
        if not self._sampling_started:
            self._sampling_started = time.monotonic()
            self._first_step = tick.step
            self._loading = False
            self._stage(STAGE_GENERATE, self._stage_text(STAGE_GENERATE))
        self._steps_seen = tick.steps
        self._last_step = max(self._last_step, tick.step)
        if tick.sec_per_step > 0:
            self._engine_rate = tick.sec_per_step
        self._progress(tick.step, tick.steps)

    def _engine_load(self, done: int, total: int) -> None:
        """Weight-loading progress, which is *not* a sampling step.

        `sd-server` answers `capabilities` about a second after it starts and
        loads nothing until the first request — so on a warm server the first
        edit still spends a minute or two reading weights, and it does it
        *after* the engine has already logged "generating image". Trusting the
        log line alone would leave the UI saying "Generating…" beside a frozen
        bar for the whole load, which reads as a hang.

        So the load bar overrides the stage text until a real step arrives.
        After that it is ignored: a disk-streamed recipe reloads segments
        mid-run and reuses the same bar, and letting that hijack the step bar
        would look like going backwards.
        """
        if self._sampling_started:
            return
        if not self._loading:
            self._loading = True
            self._stage(STAGE_LOAD, self._stage_text(STAGE_LOAD))
        self._progress(done, total)

    # -- the run ----------------------------------------------------------
    def run(self) -> Result:
        job = self.job
        self._fetch_weights()
        self._check()

        composed, refs = self.compose()
        seed = job.seed if job.seed >= 0 else random.randrange(0, 2 ** 31 - 1)
        started = time.monotonic()

        self._stage(STAGE_LOAD, self._stage_text(STAGE_LOAD))
        if job.recipe.engine == "cli":
            image = self._run_cli(composed, refs, seed)
        else:
            image = self._run_server(composed, refs, seed)
        self._check()

        elapsed = time.monotonic() - started
        sampling = self._sampling_elapsed or (
            time.monotonic() - self._sampling_started if self._sampling_started
            else 0.0)
        # Steps *inside* the timed window — see `throughput`. Falls back to the
        # full count when there were not two ticks to measure between, which is
        # the best available answer for a one-step run.
        measured_steps = max(1, self._last_step - self._first_step)
        return Result(
            image=image, prompt=composed, seed=seed,
            steps=self._steps_seen or job.effective_steps(),
            elapsed=elapsed,
            sec_per_step=throughput(self._engine_rate, sampling, measured_steps),
            size=image.size, notes=tuple(self._notes),
            log_tail=self.engine.log_tail())

    # -- steps ------------------------------------------------------------
    def _fetch_weights(self) -> None:
        job = self.job
        if fetch.have(job.recipe, job.models_dir):
            return
        pending = fetch.download_size(job.recipe, job.models_dir)
        self._stage(STAGE_DOWNLOAD,
                    f"Downloading {job.recipe.label} — "
                    f"{pending / 1e9:.1f} GB…")
        try:
            fetch.fetch_recipe(
                job.recipe, job.models_dir,
                progress=lambda d, t, _l: self._progress(d, t),
                is_cancelled=lambda: self._cancelled)
        except fetch.Cancelled as e:
            raise Cancelled() from e
        except fetch.FetchError as e:
            raise EditError(str(e)) from e

    def compose(self) -> tuple[str, list[prompt.Reference]]:
        """The prompt to send and the references to send with it.

        Public because the setup page shows the user exactly this before they
        press Generate — the composed prompt is not allowed to be a surprise.
        """
        job = self.job
        recipe = job.recipe
        refs, note = prompt.apply_limits(job.all_references(), recipe.max_refs,
                                         recipe.ref_images, recipe.label)
        if note:
            self._note(note)

        if job.prompt_override.strip():
            return job.prompt_override.strip(), refs

        if recipe.uses_photomaker():
            text, warning = prompt.photomaker_prompt(job.instruction)
            if warning:
                self._note(warning)
            return text, refs
        return prompt.compose(refs, job.instruction, recipe.family), refs

    def _note(self, text: str) -> None:
        if text and text not in self._notes:
            self._notes.append(text)

    # -- the server path --------------------------------------------------
    def _run_server(self, composed: str, refs: Sequence[prompt.Reference],
                    seed: int) -> Image.Image:
        job = self.job
        recipe = job.recipe
        try:
            api = self.engine.ensure(
                recipe, job.models_dir, memory=job.memory, backend=job.backend,
                threads=job.threads, max_vram_gb=job.max_vram_gb,
                engine_path=job.engine_path, on_stage=self._engine_stage,
                is_cancelled=lambda: self._cancelled)
        except server.ServerError as e:
            if self._cancelled:
                raise Cancelled() from e
            raise EditError(_explain(f"{e}\n{self.engine.log_tail()}")) from e
        self._check()

        init_b64 = ""
        ref_b64: list[str] = []
        try:
            if recipe.ref_images:
                ref_b64 = [client.encode_image(r.path) for r in refs]
            if job.source is not None and not recipe.source_as_ref:
                init_b64 = client.encode_image(job.source)
            # A recipe with `ref_images=False` still has one slot worth using:
            # IP-Adapter's single conditioning image. The first reference is the
            # one the user put first, which is the one they meant.
            ip_b64 = ("" if recipe.ref_images or not recipe.uses_ip_adapter()
                      or not refs else client.encode_image(refs[0].path))
        except client.ApiError as e:
            raise EditError(str(e)) from e

        request = client.request_from_recipe(
            recipe, composed, width=job.width, height=job.height, seed=seed,
            steps=job.steps, cfg_scale=job.cfg_scale, strength=job.strength,
            negative_prompt=job.negative_prompt, init_image=init_b64,
            ref_images=ref_b64, ip_adapter_image=ip_b64,
            ip_adapter_strength=job.ip_adapter_strength)

        try:
            images = self.engine.generate(
                api, request, on_tick=self._engine_tick,
                on_load=self._engine_load, on_stage=self._engine_stage,
                is_cancelled=lambda: self._cancelled)
        except client.ApiError as e:
            if self._cancelled or "cancelled" in str(e).lower():
                raise Cancelled() from e
            raise EditError(_explain(f"{e}\n{self.engine.log_tail()}")) from e
        except server.ServerError as e:
            raise EditError(_explain(f"{e}\n{self.engine.log_tail()}")) from e

        if not images:
            raise EditError("The engine finished but returned no image.\n"
                            + self.engine.log_tail())
        return _decode(images[0])

    # -- the one-shot path ------------------------------------------------
    def _run_cli(self, composed: str, refs: Sequence[prompt.Reference],
                 seed: int) -> Image.Image:
        """`sd-cli` for recipes the server API cannot express.

        Today that is PhotoMaker, whose identity images are given as a
        *directory* (`--pm-id-images-dir`) that `sd-server` fixes at startup —
        so a face set cannot travel in a request. The directory is staged here
        from whichever references the user tagged Face, which is a better fit
        than it sounds: PhotoMaker genuinely improves with several photos of the
        same person, so a role that collects them is the right interface.
        """
        job = self.job
        exe = binary.find_cli(job.engine_path)
        if exe is None:
            raise EditError(
                f"{binary.CLI_NAME} was not found. Fetch a copy with:\n"
                f"    python3 -m local_edit --fetch-engine")

        work = paths.work_dir() / f"run-{int(time.time() * 1000)}"
        out_path = work / "out.png"
        try:
            work.mkdir(parents=True, exist_ok=True)
            pm_dir = self._stage_photomaker_dir(work, refs) \
                if job.recipe.uses_photomaker() else None
            ip_image = (refs[0].path if refs and job.recipe.uses_ip_adapter()
                        and not job.recipe.ref_images else None)
            argv = binary.build_cli_argv(
                exe, job.recipe, job.models_dir, prompt=composed, output=out_path,
                init_image=(job.source if job.source is not None
                            and not job.recipe.source_as_ref else None),
                ref_images=tuple(r.path for r in refs)
                if job.recipe.ref_images else (),
                ip_adapter_image=ip_image, pm_id_dir=pm_dir, memory=job.memory,
                backend=job.backend, steps=job.steps, cfg_scale=job.cfg_scale,
                seed=seed, width=job.width, height=job.height,
                strength=job.strength, negative_prompt=job.negative_prompt,
                threads=job.threads, max_vram_gb=job.max_vram_gb)
            argv.extend(job.extra_args)
            self._spawn_cli(argv, exe)
            try:
                with Image.open(out_path) as im:
                    im.load()
                    return im.convert("RGBA" if "A" in im.mode else "RGB")
            except (OSError, ValueError) as e:
                # `sd-cli` exits 0 even when generation failed — a real run that
                # ran out of VRAM at segment 15 of 27 logged four ERROR lines,
                # wrote nothing, and returned success. So the exit code is not
                # the signal; the missing file is, and the reason is in the log.
                raise EditError(_explain(self._cli_log)) from e
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _stage_photomaker_dir(self, work: Path,
                              refs: Sequence[prompt.Reference]) -> Path:
        """Copy the Face references into a directory of their own."""
        faces = [r for r in refs if r.role == prompt.ROLE_FACE] or list(refs)
        pm_dir = work / "id_images"
        pm_dir.mkdir(parents=True, exist_ok=True)
        for index, ref in enumerate(faces, start=1):
            try:
                shutil.copy2(ref.path, pm_dir / f"{index:02d}{ref.path.suffix}")
            except OSError as e:
                raise EditError(f"could not stage {ref.path.name}: {e}") from e
        if not faces:
            self._note("PhotoMaker needs at least one reference photo of the "
                       "person, tagged Face.")
        return pm_dir

    def _spawn_cli(self, argv: list[str], exe: Path) -> None:
        """Run `sd-cli` to completion, parsing its output for progress."""
        parser = progress.Parser(on_tick=self._engine_tick,
                                 on_load=self._engine_load,
                                 on_stage=self._engine_stage)
        try:
            self._proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=binary.child_env(exe))
        except OSError as e:
            raise EditError(f"could not start {exe.name}: {e}") from e

        assert self._proc.stdout is not None
        try:
            while True:
                chunk = self._proc.stdout.read1(8192)
                if not chunk:
                    break
                parser.feed(chunk.decode("utf-8", "replace"))
        except (OSError, ValueError):
            pass
        parser.flush()
        code = self._proc.wait()
        self._proc = None
        # The whole retained log, not a short tail: `_explain` matches on
        # substrings that may be many lines above the last thing printed.
        self._cli_log = "\n".join(parser.log)

        if self._cancelled:
            raise Cancelled()
        if code != 0:
            raise EditError(_explain("\n".join(parser.log), code))

    #: Set by `_spawn_cli`, read when the output file turns out unreadable.
    _cli_log: str = ""


def _decode(data: bytes) -> Image.Image:
    """A PNG from the engine, as a PIL image detached from its buffer.

    `Image.open` on a `BytesIO` is lazy and keeps the buffer alive; `load()`
    forces the decode so the bytes — which for a 1024x1024 PNG is most of a
    megabyte — can be freed with the rest of the request.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGBA" if "A" in im.mode else "RGB")
    except (OSError, ValueError) as e:
        raise EditError("The engine returned something that is not an "
                        "image.") from e
