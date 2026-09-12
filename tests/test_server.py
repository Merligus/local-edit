"""Tests for the warm-engine lifecycle, without starting one.

Run:  python3 tests/test_server.py

`EngineServer` outliving any one edit is the whole point of it — weights here
are 5 to 34 GB — and it is also where the subtle bugs live, because the object
the parser reports *to* does not outlive the edit even though the parser does.
"""

from _harness import check, run                                    # noqa: E402
from local_edit.engine import catalog as cat                       # noqa: E402
from local_edit.engine import server                               # noqa: E402

KLEIN = cat.get("flux2-klein-4b-q4")
KONTEXT = cat.get("kontext-q3")


def test_restart_key():
    print("\nwhat forces a restart")
    eng = server.EngineServer()
    base = eng.key_for(KLEIN, ("--offload-to-cpu",), "vulkan", 0, 0.0)
    check(eng.key_for(KLEIN, ("--offload-to-cpu",), "vulkan", 0, 0.0) == base,
          "the same everything is the same server")
    check(eng.key_for(KONTEXT, ("--offload-to-cpu",), "vulkan", 0, 0.0) != base,
          "a different model restarts — sd.cpp cannot switch one in place")
    check(eng.key_for(KLEIN, (), "vulkan", 0, 0.0) != base,
          "different memory flags restart, because they are startup flags: a "
          "recipe graded OFFLOAD at 512 and STREAM at 1024 needs a different "
          "process, not a different request")
    check(eng.key_for(KLEIN, ("--offload-to-cpu",), "cpu", 0, 0.0) != base,
          "a different backend restarts")


def test_stop_is_idempotent():
    print("\nstopping is always safe")
    eng = server.EngineServer()
    check(not eng.running, "a fresh server is not running")
    eng.stop()
    eng.stop()
    check(not eng.running, "stopping nothing twice is fine")
    check(not eng.idle_expired(), "and nothing idle has expired")
    check(not eng.stop_if_idle(), "so the housekeeper has nothing to do")


def test_ports_are_not_guessed():
    print("\nthe port is asked for, not assumed")
    a, b = server.free_port(), server.free_port()
    check(a > 1024 and b > 1024, "both are unprivileged")
    check(a != 1234 and b != 1234,
          "and neither is sd.cpp's default, which is exactly the port a "
          "hand-started server would already be on")


def test_taps_are_released():
    print("\ncallbacks do not outlive the run that installed them")
    # The bug this guards: the parser outlives an edit, the UI worker that owns
    # the callbacks does not. A callback left bound fires on the *next* edit
    # into a deleted Qt object, PySide6 raises inside the reader thread, and
    # progress is dead for the rest of the session.
    eng = server.EngineServer()

    def exploding(*_args):
        raise RuntimeError("Internal C++ object already deleted")

    eng.set_taps(on_tick=exploding, on_load=exploding, on_stage=exploding)
    eng.set_taps()                       # what generate()'s finally does
    seen = []
    eng.set_taps(on_tick=seen.append)
    eng._parser.feed("  |=| 2/4 - 1.00s/it\r")
    check([(t.step, t.steps) for t in seen] == [(2, 4)],
          "the second run's ticks reach the second run, not the first")

    eng.set_taps()
    eng._parser.feed("  |=| 3/4 - 1.00s/it\r")
    check(True, "output arriving with no run in flight is ignored, not an error")


def test_idle_timeout():
    print("\nidle servers give the VRAM back")
    eng = server.EngineServer(idle_timeout=0)
    check(not eng.idle_expired(), "a timeout of 0 means never retire")
    eng = server.EngineServer(idle_timeout=600)
    check(not eng.idle_expired(),
          "and a server that was never started has nothing to retire either")


if __name__ == "__main__":
    raise SystemExit(run(test_restart_key, test_stop_is_idempotent,
                         test_ports_are_not_guessed, test_taps_are_released,
                         test_idle_timeout))
