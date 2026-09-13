"""Tests for one edit's orchestration, without running an engine.

Run:  python3 tests/test_runner.py

The stage machine is what this mostly guards, because the engine's output does
not arrive in the order its log lines suggest and the honest reading of it is
not the obvious one.
"""

from pathlib import Path

from _harness import check, run                                    # noqa: E402
from local_edit.engine import catalog as cat                       # noqa: E402
from local_edit.engine import hardware as hw                       # noqa: E402
from local_edit.engine import progress, prompt as P, runner, server  # noqa: E402

KLEIN = cat.get("flux2-klein-4b-q4")


def replay(lines, job=None):
    """Drive a Runner's callbacks from engine output, with no engine."""
    job = job or runner.Job(recipe=KLEIN, instruction="x")
    r = runner.Runner(job, server.EngineServer())
    stages, bars = [], []
    r._on_stage = lambda k, _t: stages.append(k)
    r._on_progress = lambda d, t: bars.append((d, t))
    parser = progress.Parser(on_tick=r._engine_tick, on_load=r._engine_load,
                             on_stage=r._engine_stage)
    for line in lines:
        parser.feed_line(line)
    return r, stages, bars


def test_generating_is_not_sampling():
    print("\n'generating image' is not the start of sampling")
    # The trap, in the order a real run produces it: the engine logs that it is
    # generating, then spends two minutes loading the diffusion weights, and
    # only then takes a step. Believing the log line leaves the UI claiming to
    # generate beside a frozen bar for the whole load.
    _, stages, bars = replay([
        "[INFO] image.cpp:841 - generating image: 1/1 - seed 42",
        "  |####  | 27/149 - 252.50MB/s",
        "  |######| 149/149 - 312.81MB/s",
        "  |==>   | 1/4 - 41.20s/it",
        "  |=====>| 4/4 - 40.10s/it",
        "[INFO] image.cpp:599 - decode_first_stage completed",
    ])
    check(stages == [runner.STAGE_LOAD, runner.STAGE_GENERATE,
                     runner.STAGE_DECODE],
          f"stages are load -> generate -> decode, not generate first ({stages})")
    check((27, 149) in bars,
          "the loading bar is still shown, even with no load bar before the "
          "'generating' line — an earlier version suppressed it and only "
          "worked because the text encoder happened to print one first")
    check(bars.index((27, 149)) < bars.index((1, 4)),
          "and it is shown before the step bar, not mixed into it")


def test_stage_text_is_not_repeated():
    print("\na stage is announced once")
    r, stages, _ = replay([
        "  |#  | 10/149 - 100.00MB/s",
        "  |## | 20/149 - 100.00MB/s",
        "  |###| 149/149 - 100.00MB/s",
    ])
    check(stages == [runner.STAGE_LOAD],
          f"three load ticks announce one stage, not three ({stages})")


def test_vae_tiles_are_not_sampling_steps():
    print("\na tiled VAE decode drives the same bar and must not be measured")
    # Verbatim from a real CUDA run: four sampling steps at ~4.4 s, then nine
    # VAE tiles at ~1.05 s, all through the same `n/m - Xs/it` widget. Counting
    # the tiles put the median at 1.05 — a fourfold overstatement, and it went
    # into calibration, where it would have promised every later run a speed the
    # card cannot reach.
    job = runner.Job(recipe=KLEIN, instruction="x", steps=4)
    r = runner.Runner(job, server.EngineServer())
    bars = []
    r._on_progress = lambda d, t: bars.append((d, t))
    r._on_stage = lambda k, txt: None
    parser = progress.Parser(on_tick=r._engine_tick, on_load=r._engine_load,
                             on_stage=r._engine_stage)
    for line in ("  |=| 1/4 - 5.68s/it", "  |=| 2/4 - 4.42s/it",
                 "  |=| 3/4 - 4.40s/it", "  |=| 4/4 - 4.39s/it",
                 "  |=| 1/9 - 1.30s/it", "  |=| 5/9 - 1.05s/it",
                 "  |=| 9/9 - 1.05s/it"):
        parser.feed_line(line)
    check(r._engine_rates == [5.68, 4.42, 4.40, 4.39],
          "only the four ticks whose denominator matches the requested step "
          "count are measured")
    check(abs(runner.throughput(r._engine_rates, 0, 0) - 4.41) < 0.01,
          f"giving 4.41 s/step, not the 1.05 the tiles would have produced")
    check((1, 9) in bars and (9, 9) in bars,
          "the tiles still drive the progress bar — they are real work and the "
          "user should see it")


def test_engine_rates_are_all_captured():
    print("\nevery rate the engine reports is kept, not just the last")
    r, _, _ = replay(["  |=| 1/4 - 41.20s/it", "  |=| 2/4 - 4.40s/it",
                      "  |=| 3/4 - 4.40s/it", "  |=| 4/4 - 1.05s/it"])
    check(r._engine_rates == [41.20, 4.40, 4.40, 1.05],
          "all four are collected")
    check(runner.throughput(r._engine_rates, 0, 0) == 4.40,
          "and the median discards both the warm-up high and the bogus final "
          "reading — a real CUDA run emitted its last two ticks in the same "
          "instant and reported 1.05 against a true 4.4")


def test_source_placement():
    print("\nthe source reaches the model the way its family expects")
    refs = (P.Reference(Path("face.jpg"), P.ROLE_FACE),)
    edit = runner.Job(recipe=KLEIN, source=Path("photo.png"), references=refs)
    check([r.path.name for r in edit.all_references()]
          == ["photo.png", "face.jpg"],
          "an edit model sees the image being edited as reference 1")

    import dataclasses
    generative = dataclasses.replace(KLEIN, id="gen", source_as_ref=False)
    gen = runner.Job(recipe=generative, source=Path("photo.png"), references=refs)
    check([r.path.name for r in gen.all_references()] == ["face.jpg"],
          "a generative model does not — its source is an init image, and "
          "counting it as a reference would shift every number in the prompt")

    none = runner.Job(recipe=KLEIN, references=refs)
    check([r.path.name for r in none.all_references()] == ["face.jpg"],
          "with no source, the user's first reference really is image 1")


def test_prompt_numbering_matches_reference_order():
    print("\nthe prompt's numbers match the order the references are sent")
    job = runner.Job(
        recipe=KLEIN, source=Path("photo.png"),
        references=(P.Reference(Path("face.jpg"), P.ROLE_FACE),
                    P.Reference(Path("coat.png"), P.ROLE_CLOTHING)),
        instruction="make them ride a motorcycle")
    r = runner.Runner(job, server.EngineServer())
    composed, refs = r.compose()
    check([x.path.name for x in refs] == ["photo.png", "face.jpg", "coat.png"],
          "the references go in list order, source first")
    check("person in image 2" in composed and "clothing from image 3" in composed,
          "and the prompt numbers them from that same list — the source is 1, "
          "so the user's own start at 2")


def test_sizes_are_rounded_to_the_latent_grid():
    print("\noutput sizes land on the grid the model actually uses")
    for w, h in ((1920, 1080), (1000, 1000), (33, 17)):
        rw, rh = runner.plan_size(KLEIN, (w, h), 1024)
        check(rw % KLEIN.size_multiple == 0 and rh % KLEIN.size_multiple == 0,
              f"{w}x{h} -> {rw}x{rh}, both multiples of {KLEIN.size_multiple}")
    check(runner.plan_size(KLEIN, (1920, 1080), 1024) == (1024, 576),
          "the long edge is what the cap applies to, and the aspect is kept")
    check(runner.plan_size(KLEIN, None, None)[0] == KLEIN.default_size,
          "with no source and no cap, the recipe's own size is used")


def test_failures_are_explained_not_dumped():
    print("\nfailures get an explanation, not a log dump")
    oom = ("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n"
           "[ERROR] main.cpp:961 - generate failed")
    check("output size" in runner._explain(oom).lower(),
          "an out-of-memory failure says what to change")
    check("Vulkan" in runner._explain("ggml_vulkan: no vulkan device found"),
          "a missing device says so")
    mystery = "\n".join(f"line {i}" for i in range(50))
    out = runner._explain(mystery, 1)
    check("code 1" in out and len(out.splitlines()) <= 5,
          "an unrecognised failure shows the exit code and a short tail, not "
          "fifty lines of it")


if __name__ == "__main__":
    raise SystemExit(run(
        test_generating_is_not_sampling, test_stage_text_is_not_repeated,
        test_vae_tiles_are_not_sampling_steps,
        test_engine_rates_are_all_captured, test_source_placement,
        test_prompt_numbering_matches_reference_order,
        test_sizes_are_rounded_to_the_latent_grid,
        test_failures_are_explained_not_dumped))
