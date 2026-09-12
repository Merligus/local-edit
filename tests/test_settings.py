"""Tests for settings and calibration.

Run:  python3 tests/test_settings.py

The calibration keying is what this mostly guards. A rate measured at 512x512
with weights in RAM says nothing about 1024x1024 with weights streaming from
disk — on the development machine those differ by more than tenfold — so a key
that conflated them would learn one number and then be confidently wrong
everywhere else.
"""

from _harness import check, run                                    # noqa: E402
from local_edit import settings as st                              # noqa: E402
from local_edit.engine import hardware as hw                       # noqa: E402
from local_edit.engine import runner                               # noqa: E402
from local_edit.engine import catalog as cat                       # noqa: E402


def test_round_trip():
    print("\nsettings survive a round trip")
    s = st.Settings()
    check(st.Settings.from_dict(s.to_dict()).to_dict() == s.to_dict(),
          "defaults round trip unchanged")
    s.size = 1024
    s.steps = 12
    s.memory_mode = st.MEMORY_DISK
    check(st.Settings.from_dict(s.to_dict()).size == 1024, "size round trips")
    check(st.Settings.from_dict(s.to_dict()).steps == 12, "steps round trip")


def test_unknown_keys_are_preserved():
    print("\na newer version's settings are not destroyed by an older one")
    d = st.Settings().to_dict()
    d["something_from_the_future"] = {"a": 1}
    check(st.Settings.from_dict(d).to_dict()["something_from_the_future"]
          == {"a": 1}, "unknown keys survive")


def test_garbage_is_clamped_not_rejected():
    print("\nhand-edited nonsense degrades to defaults")
    s = st.Settings.from_dict({"recipe_id": "no-such-model", "steps": 9999,
                               "size": "banana", "seed": -50,
                               "memory_mode": "wishful", "backend": "quantum",
                               "idle_timeout": -1})
    check(s.recipe_id == cat.DEFAULT_RECIPE_ID, "an unknown model falls back")
    check(s.steps == st.MAX_STEPS, "an absurd step count is clamped")
    check(s.size is None, "a non-integer size becomes 'match the source'")
    check(s.seed == -1, "a negative seed becomes 'random'")
    check(s.memory_mode == st.MEMORY_AUTO, "an unknown memory mode falls back")
    check(s.backend == st.BACKEND_AUTO, "so does an unknown backend")
    check(s.idle_timeout == 0, "a negative timeout is clamped, not rejected")
    check(st.Settings.from_dict("not a dict").recipe_id == cat.DEFAULT_RECIPE_ID,
          "a settings file that is not even an object still opens the app")


def test_calibration_keys():
    print("\ncalibration distinguishes what genuinely differs")
    c = st.Calibration()
    c.record("klein", 512, 512, 22.34, hw.OFFLOAD)
    check(c.get("klein", 512, 512, hw.OFFLOAD) == 22.34, "a measurement reads back")
    check(c.get("klein", 512, 576, hw.OFFLOAD) == 22.34,
          "a nearby size shares the bucket — those cost the same and there is "
          "no reason to make the user measure both")
    check(c.get("klein", 1024, 1024, hw.OFFLOAD) is None,
          "four times the pixels does NOT share it — measured at 3.4x the cost")
    check(c.get("klein", 512, 512, hw.STREAM) is None,
          "nor does a different grade — streaming is about 3x slower")
    check(c.get("other", 512, 512, hw.OFFLOAD) is None,
          "nor a different model")


def test_calibration_blends():
    print("\na measurement is blended, not replaced")
    c = st.Calibration()
    c.record("klein", 512, 512, 20.0, hw.FITS)
    c.record("klein", 512, 512, 30.0, hw.FITS)
    v = c.get("klein", 512, 512, hw.FITS)
    check(20.0 < v < 30.0, f"the second run moves it partway ({v:.1f}), so one "
                           f"run that shared the GPU does not poison it")
    c.record("klein", 512, 512, 0, hw.FITS)
    check(c.get("klein", 512, 512, hw.FITS) == v, "a zero is ignored")
    check(st.Calibration.from_dict({"a": "banana", "b": -1, "c": 3.0}).rates
          == {"c": 3.0}, "unparseable rates are dropped on load")


def test_memory_mode_overrides_the_verdict():
    print("\nthe memory override wins over the automatic grade")
    s = st.Settings()
    v = hw.Verdict("klein", hw.OFFLOAD, 1.0, 0.0, "", ("--offload-to-cpu",))
    check(s.memory_flags(v) == ("--offload-to-cpu",),
          "auto follows the verdict")
    s.memory_mode = st.MEMORY_VRAM
    check(s.memory_flags(v) == (), "'keep on the GPU' passes no memory flags")
    s.memory_mode = st.MEMORY_DISK
    check("--params-backend" in s.memory_flags(v), "'stream from disk' does")


def test_estimates_use_the_measurement():
    print("\nan estimate prefers a measurement to a prior")
    klein = cat.get("flux2-klein-4b-q4")
    prior = runner.estimate_seconds(klein, 512, 512, 4, grade=hw.OFFLOAD)
    measured = runner.estimate_seconds(klein, 512, 512, 4, sec_per_step=60.0,
                                       grade=hw.OFFLOAD)
    check(measured > prior, "a slower measured rate produces a longer estimate")
    with_refs = runner.estimate_seconds(klein, 512, 512, 4, grade=hw.OFFLOAD,
                                        references=2)
    check(with_refs > prior * 2,
          "two references more than double the sampling cost — measured at "
          "22.3 s/step bare against 55.4 s/step with two")


def test_mpx_buckets():
    print("\nsize buckets are coarse but monotonic")
    seen = [st.mpx_bucket(s, s) for s in (256, 512, 768, 1024, 1536, 2048)]
    check(len(set(seen)) >= 4, f"distinct sizes mostly get distinct buckets: {seen}")
    check(st.mpx_bucket(512, 512) == st.mpx_bucket(576, 448),
          "equal-area sizes share a bucket")


if __name__ == "__main__":
    raise SystemExit(run(
        test_round_trip, test_unknown_keys_are_preserved,
        test_garbage_is_clamped_not_rejected, test_calibration_keys,
        test_calibration_blends, test_memory_mode_overrides_the_verdict,
        test_estimates_use_the_measurement, test_mpx_buckets))
