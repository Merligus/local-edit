"""Tests for the engine-output parser.

Run:  python3 tests/test_progress.py

Every fixture below is copied verbatim from a real sd.cpp run on the development
machine. That matters more than usual here: this parser is the only source of
progress the app has — the server's job API reports no step count at all — and
the output it reads is neither line-oriented nor single-purpose.
"""

from _harness import check, run                                    # noqa: E402
from local_edit.engine import progress as pr                       # noqa: E402


def collect(chunks):
    ticks, loads, stages = [], [], []
    p = pr.Parser(on_tick=ticks.append,
                  on_load=lambda d, t: loads.append((d, t)),
                  on_stage=lambda s, line: stages.append(s))
    for chunk in chunks:
        p.feed(chunk)
    p.flush()
    return p, ticks, loads, stages


def test_carriage_returns():
    print("\nthe stream is \\r-separated, not \\n-separated")
    # sd.cpp writes "\r<bar> n/m - rate\033[K" and emits no newline until the
    # pass ends. A reader that split on \n would see nothing for the whole run.
    stream = ("  |==>      | 1/4 - 34.40s/it\x1b[K\r"
              "  |=====>   | 2/4 - 21.50s/it\x1b[K\r"
              "  |=========| 4/4 - 22.34s/it\x1b[K\n")
    _, ticks, _, _ = collect([stream])
    check([(t.step, t.steps) for t in ticks] == [(1, 4), (2, 4), (4, 4)],
          "all three updates are seen despite there being one newline")
    check(abs(ticks[-1].sec_per_step - 22.34) < 1e-9, "the rate is parsed")


def test_split_across_reads():
    print("\nchunk boundaries do not have to align with anything")
    _, ticks, _, _ = collect(["  |==>  | 1/4 - 3", "4.40s/it\x1b[K\r"])
    check([(t.step, t.steps) for t in ticks] == [(1, 4)],
          "an update split mid-number is reassembled")


def test_ansi_is_stripped():
    print("\nANSI sequences are removed")
    _, ticks, _, _ = collect(["\x1b[32m  |=| 2/2 - 1.00s/it\x1b[K\x1b[0m\r"])
    check(len(ticks) == 1, "colour codes do not prevent a match")


def test_units_are_normalised():
    print("\nit/s and s/it both become seconds per step")
    _, ticks, _, _ = collect(["  |=| 4/4 - 2.50it/s\x1b[K\n"])
    check(abs(ticks[0].sec_per_step - 0.4) < 1e-9,
          "2.5 it/s is 0.4 s per step, not 2.5")


def test_load_bar_is_not_a_step_bar():
    print("\nthe weight-loading bar is a different bar")
    # Same widget, different unit. Counting these as sampling steps would show
    # "298 steps" for a four-step model.
    _, ticks, loads, _ = collect([
        "  |####      | 27/149 - 252.50MB/s\x1b[K\r",
        "  |##########| 149/149 - 312.81MB/s\x1b[K\n"])
    check(not ticks, "MB/s updates are not sampling steps")
    check(loads == [(27, 149), (149, 149)], "they are reported as loading")


def test_log_ratios_do_not_tick():
    print("\na bare ratio in a log line is not progress")
    _, ticks, loads, _ = collect([
        "[INFO   ] model_loader.cpp:1035 - loading 149/149 tensors from a.gguf\n",
        "[INFO   ] main.cpp:575  - 1/1 images saved\n"])
    check(not ticks and not loads,
          "'149/149 tensors' and '1/1 images saved' have no rate and do not count")


def test_stages_from_real_output():
    print("\nstages follow the engine's real log lines")
    _, _, _, stages = collect([
        "[INFO   ] model_loader.cpp:1035 - loading 149/149 tensors from a.gguf\n",
        "[VERBOSE] conditioner.hpp:2933 - computing condition graph completed, "
        "taking 19137 ms\n",
        "[INFO   ] image.cpp:504  - get_learned_condition completed, taking 19.14s\n",
        "[INFO   ] image.cpp:841  - generating image: 1/1 - seed 42\n",
        "[INFO   ] image.cpp:599  - decode_first_stage completed, taking 9.66s\n"])
    check(stages == [pr.STAGE_ENCODE, pr.STAGE_SAMPLE, pr.STAGE_DECODE],
          "encode, then sample, then decode (load is the starting state, so it "
          "is not a transition)")


def test_stages_only_move_forward():
    print("\nstages never go backwards")
    # A disk-streamed recipe reloads segments mid-run and logs "loading" again.
    p, _, _, stages = collect([
        "[INFO   ] image.cpp:841  - generating image: 1/1 - seed 42\n",
        "[VERBOSE] model_loader.cpp:1035 - loading 8/8 tensors from a.gguf\n"])
    check(p.stage == pr.STAGE_SAMPLE,
          "a mid-run reload does not send the UI back to 'Loading model…'")


def test_log_is_bounded_and_kept():
    print("\nthe log is kept for failures, but bounded")
    p, _, _, _ = collect([f"[INFO   ] x.cpp:1 - line {i}\n" for i in range(500)])
    check(len(p.log) == pr.LOG_TAIL,
          f"only the last {pr.LOG_TAIL} lines are held — the engine's capability "
          f"dump alone is over a hundred")
    check("line 499" in p.tail(1), "and the most recent one is the tail")


def test_reset_between_generations():
    print("\na warm server runs more than one generation")
    p, _, _, _ = collect(["[INFO   ] image.cpp:599  - decode_first_stage done\n"])
    check(p.stage == pr.STAGE_DECODE, "the first edit ends in decode")
    p.reset()
    check(p.stage == pr.STAGE_LOAD,
          "the next one starts from the beginning, or its stages could never "
          "advance again")


if __name__ == "__main__":
    raise SystemExit(run(
        test_carriage_returns, test_split_across_reads, test_ansi_is_stripped,
        test_units_are_normalised, test_load_bar_is_not_a_step_bar,
        test_log_ratios_do_not_tick, test_stages_from_real_output,
        test_stages_only_move_forward, test_log_is_bounded_and_kept,
        test_reset_between_generations))
