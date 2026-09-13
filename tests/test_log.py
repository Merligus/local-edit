"""Tests for the on-disk log.

Run:  python3 tests/test_log.py

The log exists because a failure that has already happened has to stay
readable. Three properties are worth holding onto, and each was a real gap:

* **Engine output reaches the file.** The parser used to keep a 60-line tail in
  memory and nothing else, so a message box dismissed too quickly took the
  engine's own diagnosis with it.
* **Progress bars do not.** A run emits hundreds of `\\r`-rewritten bars. If
  they landed in the file, the one line that explains a failure would be buried.
* **The failure has an actionable hint.** `flux segment 2/27 ... failed during
  weight preparation` reached the user as the bare words "The engine failed."
"""

import logging
import tempfile
from pathlib import Path

from _harness import check, run                                    # noqa: E402
from local_edit import log as applog                               # noqa: E402
from local_edit.engine import progress as pr                       # noqa: E402
from local_edit.engine import runner as rn                         # noqa: E402


def capture(fn):
    """Run `fn` with the app's loggers writing to a temporary file."""
    root = logging.getLogger("local_edit")
    saved_handlers, saved_level = list(root.handlers), root.level
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.log"
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        root.handlers = [handler]
        root.setLevel(logging.DEBUG)
        root.propagate = False
        try:
            fn()
        finally:
            handler.close()
            root.handlers, root.level = saved_handlers, saved_level
        return path.read_text(encoding="utf-8")


def test_engine_lines_are_recorded():
    print("\nengine output lands in the file, progress bars do not")
    # Verbatim from a real run: two log lines with a rewriting progress bar
    # between them, exactly as the child writes it.
    stream = ("[INFO   ] image.cpp:595  - latent 1 decoded, taking 9.71s\n"
              "  |==>      | 1/4 - 34.40s/it\x1b[K\r"
              "  |=====>   | 2/4 - 34.40s/it\x1b[K\r"
              "[ERROR  ] ggml_runner.cpp:868 - flux segment 2/27 "
              "(flux.double_blocks.0) failed during weight preparation\n")

    text = capture(lambda: pr.Parser().feed(stream))
    check("latent 1 decoded" in text, "an INFO line is written")
    check("failed during weight preparation" in text, "an ERROR line is written")
    check("ERROR" in text.split("weight preparation")[0].splitlines()[-1],
          "the engine's own [ERROR] tag sets the record's level")
    check("34.40s/it" not in text, "progress bars are not written")
    check("\x1b" not in text, "ANSI escapes are stripped before writing")


def test_blank_lines_are_dropped():
    print("\nempty output is not logged")
    text = capture(lambda: [applog.engine_line(""), applog.engine_line("   "),
                            applog.engine_line("real")])
    check("real" in text, "a real line survives")
    check(len([ln for ln in text.splitlines() if ln.strip()]) == 1,
          "blank and whitespace-only lines are dropped")


def test_weight_preparation_has_a_hint():
    print("\nthe segment failure explains itself")
    # The four lines that reached the user's screen, and nothing else — the
    # allocator's own message had already scrolled out of the retained tail,
    # which is why this needle has to stand on its own.
    log = ("[ERROR  ] ggml_runner.cpp:868 - flux segment 2/27 "
           "(flux.double_blocks.0) failed during weight preparation\n"
           "[ERROR  ] diffusion_engine.cpp:2372 - diffusion model compute failed\n"
           "[ERROR  ] diffusion_engine.cpp:2483 - Diffusion model sampling failed\n"
           "[ERROR  ] image.cpp:879 - sampling for image 1/1 failed after 2.97s\n")
    message = rn._explain(log)
    check(message != "The engine failed.", "it is not the generic fallback")
    check("memory" in message.lower(), "it names memory as the problem")
    check("GPU" in message, "it says to check what else is using the GPU")

    later = rn._explain("flux segment 15/27 (flux.single_blocks.8) "
                        "failed during execution")
    check("memory" in later.lower(), "the execution-phase variant is covered too")


def test_hints_still_win_over_the_dump():
    print("\nthe more specific allocator message is preferred")
    both = ("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n"
            "flux segment 15/27 failed during execution\n")
    check("output size" in rn._explain(both),
          "an explicit out-of-VRAM message is matched first")


def test_setup_is_idempotent_and_survives_a_bad_home():
    print("\nsetup never stops the app from running")
    first = applog.setup()
    second = applog.setup()
    check(first == second, "calling setup twice returns the same path")
    check(applog.log_file().name == applog.LOG_NAME, "the file has a stable name")
    check(applog.tail(5, Path("/nonexistent/nope.log")) == "",
          "tailing a missing file returns empty rather than raising")


if __name__ == "__main__":
    raise SystemExit(run(
        test_engine_lines_are_recorded,
        test_blank_lines_are_dropped,
        test_weight_preparation_has_a_hint,
        test_hints_still_win_over_the_dump,
        test_setup_is_idempotent_and_survives_a_bad_home,
    ))
