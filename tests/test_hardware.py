"""Tests for the hardware grading.

Run:  python3 tests/test_hardware.py

Graded against synthetic machines, so the result does not depend on what else is
using the GPU while the tests run — which is itself a bug this file exists to
prevent: an early draft judged "can this card ever fit the working set" against
*live free* VRAM, and relabelled the entire catalog "needs a bigger GPU" the one
time it was run during a benchmark.

The anchor cases are real measurements on a GTX 1050 Ti, and they moved once
flash attention was enabled on Vulkan — which is the point of the numbers being
measured rather than assumed:

    size    without FA   with FA
     512      491 MB      329 MB
    1024     2763 MB     1120 MB   <- 2763 MB aborted; 1120 MB finishes
"""

from _harness import check, run                                    # noqa: E402
from local_edit.engine import catalog as cat                       # noqa: E402
from local_edit.engine import hardware as hw                       # noqa: E402

KLEIN = cat.get("flux2-klein-4b-q4")

#: The development machine, idle.
PASCAL = hw.Hardware(vram_gb=3.66, vram_total_gb=4.29, ram_gb=8.1,
                     ram_total_gb=12.5, disk_gb=32.0,
                     gpu_name="GTX 1050 Ti", fp16=False, swap_is_zram=True)
BIG = hw.Hardware(vram_gb=23.0, vram_total_gb=24.0, ram_gb=60.0,
                  ram_total_gb=64.0, disk_gb=900.0, gpu_name="RTX 4090",
                  fp16=True)


def test_matches_the_real_measurements():
    print("\nthe working-set model reproduces what sd.cpp reported")
    # Every figure is the engine's own `flux compute buffer size`, with flash
    # attention on — which it always is now. The reference rows matter more
    # than the size rows: an edit in this app always has at least one, and the
    # model had no term for them at all until they were measured.
    MEASURED = (
        (384, 0, 0.239), (512, 0, 0.329), (768, 0, 0.598), (1024, 0, 1.120),
        (512, 1, 0.547), (512, 2, 0.836), (512, 3, 1.066),
        (768, 1, 1.180), (768, 2, 1.697), (768, 3, 2.213),
    )
    # And one non-square, because every real edit is one: a 3200x4012 photo
    # capped for a three-reference run.
    check(abs(hw.working_gb(KLEIN, 384, 512, PASCAL, 3) - 0.847) / 0.847 < 0.15,
          "384x512 with three references: measured 0.847 GB")
    for side, refs, measured in MEASURED:
        predicted = hw.working_gb(KLEIN, side, side, PASCAL, refs)
        check(abs(predicted - measured) / measured < 0.15,
              f"{side}x{side} with {refs} ref(s): predicted {predicted:.2f} GB "
              f"vs measured {measured:.3f} GB")
    # The direction of the error is not incidental. A model that under-predicts
    # grades a run FITS and then the engine dies at segment 2 of 27 with
    # nothing to show; one that over-predicts costs a smaller image.
    check(all(hw.working_gb(KLEIN, side, side, PASCAL, refs) >= measured
              for side, refs, measured in MEASURED),
          "and never predicts less than was actually needed")


def test_references_cost_what_they_were_measured_to_cost():
    print("\na reference is another frame of attention, not a free extra")
    bare = hw.working_gb(KLEIN, 512, 512, PASCAL, 0)
    one = hw.working_gb(KLEIN, 512, 512, PASCAL, 1)
    check(one > bare * 1.4,
          f"one reference at 512px is a large increase, not a rounding error "
          f"({bare:.2f} -> {one:.2f} GB)")
    # The engine resizes every reference to the output size and concatenates
    # its tokens, so N references at one size cost about what one does at N
    # times the area. This is the property the whole model rests on.
    two_refs = hw.working_gb(KLEIN, 512, 512, PASCAL, 2)
    triple_area = hw.working_gb(KLEIN, 887, 887, PASCAL, 0)   # 3 x 512x512
    check(abs(two_refs - triple_area) / triple_area < 0.05,
          f"two references cost about what three times the area does "
          f"({two_refs:.2f} against {triple_area:.2f} GB)")
    check(hw.working_gb(KLEIN, 512, 512, PASCAL, -1) == bare,
          "a negative count cannot make a run look cheaper than a bare one")


def test_predicts_the_real_outcomes():
    print("\nthe verdicts predict what actually happened")
    ok = hw.verdict(KLEIN, PASCAL, 512, 512)
    check(ok.grade == hw.FITS,
          "512x512 fits outright — its peak is 2.83 GB, not the 5.29 GB "
          "download, and an earlier version summed the two model files that "
          "are never resident at the same time")
    big = hw.verdict(KLEIN, PASCAL, 1024, 1024)
    check(big.runnable,
          "1024x1024 is runnable — it aborted out of device memory before "
          "flash attention was enabled, and finishes now")
    check(big.grade != hw.TOO_BIG,
          "and is no longer graded beyond the card, because it no longer is")


def test_the_offload_ceiling_matches_the_runs():
    print("\nan offloaded run cannot claim the whole card, and the line is "
          "where the runs put it")
    # The card as it actually was while these were run: an idle Plasma desktop
    # holding about 0.3 GB. PASCAL is a busier snapshot of the same card, and
    # using it here would compare the outcomes against a machine that was not
    # the one they happened on.
    measured_on = hw.Hardware(vram_gb=3.99, vram_total_gb=4.29, ram_gb=10.65,
                              ram_total_gb=12.47, disk_gb=25.0, fp16=False)
    # Eleven real runs on the development card, CUDA, flash attention, weights
    # offloaded, tightest VAE tile. `True` means it produced an image.
    RUNS = (
        (512, 512, 0, True),  (512, 512, 1, True),  (512, 512, 2, True),
        (512, 512, 3, True),  (768, 768, 1, True),  (1024, 1024, 0, True),
        (384, 512, 3, True),
        (512, 640, 3, False), (768, 768, 2, False),
        (640, 768, 3, False), (768, 768, 3, False),
    )
    for w, h, refs, ran in RUNS:
        v = hw.verdict(KLEIN, measured_on, w, h, references=refs)
        graded_ok = v.grade not in (hw.TOO_BIG, hw.NO_DISK)
        check(graded_ok == ran,
              f"{w}x{h} with {refs} ref(s): graded {v.grade!r} "
              f"({v.working_gb:.2f} GB), and it "
              f"{'ran' if ran else 'ran out of memory'}")
    # The pair that pins the constant. 1.27 GB runs and 1.40 GB does not, so
    # any share that puts both on the same side of the line is wrong.
    near_pass = hw.working_gb(KLEIN, 768, 768, PASCAL, 1)
    near_fail = hw.working_gb(KLEIN, 512, 640, PASCAL, 3)
    check(near_pass < near_fail,
          f"the boundary is narrow: {near_pass:.2f} GB runs, "
          f"{near_fail:.2f} GB does not")
    check(0.40 < hw.OFFLOAD_VRAM_SHARE < 0.50,
          f"and OFFLOAD_VRAM_SHARE ({hw.OFFLOAD_VRAM_SHARE}) lies between them")


def test_grades_are_ordered():
    print("\ngrades degrade in the right order as the card shrinks")
    big = hw.verdict(KLEIN, BIG, 1024, 1024)
    check(big.grade == hw.FITS, "a 24 GB card fits Klein outright")
    check(not big.flags, "and needs no memory flags")

    # Sized against `peak_weights_gb`, not the download size: Klein's peak is
    # 2.83 GB (the larger of its encoder and its transformer, plus the VAE),
    # against a 5.29 GB download.
    tight = hw.Hardware(vram_gb=2.0, vram_total_gb=4.0, ram_gb=16.0,
                        ram_total_gb=16.0, disk_gb=100.0, fp16=True)
    check(hw.verdict(KLEIN, tight, 512, 512).grade == hw.OFFLOAD,
          "weights that do not fit in VRAM but do fit in RAM offload")
    no_ram = hw.Hardware(vram_gb=2.0, vram_total_gb=4.0, ram_gb=2.0,
                         ram_total_gb=4.0, disk_gb=100.0, fp16=True)
    check(hw.verdict(KLEIN, no_ram, 512, 512).grade == hw.STREAM,
          "weights that fit in neither stream from disk")


def test_offload_flags():
    print("\neach grade carries the flags it implies")
    small = hw.Hardware(vram_gb=2.0, vram_total_gb=4.0, ram_gb=16.0,
                        ram_total_gb=16.0, disk_gb=100.0, fp16=True)
    check("--offload-to-cpu" in hw.verdict(KLEIN, small, 512, 512).flags,
          "OFFLOAD passes --offload-to-cpu")
    stream = hw.Hardware(vram_gb=2.0, vram_total_gb=4.0, ram_gb=2.0,
                         ram_total_gb=4.0, disk_gb=100.0, fp16=True)
    flags = hw.verdict(KLEIN, stream, 512, 512).flags
    check("--params-backend" in flags, "STREAM passes --params-backend")
    check(any("disk" in f for f in flags), "and points the diffusion model at disk")


def test_disk_is_the_only_hard_stop():
    print("\nonly a disk shortage blocks the button")
    tiny = hw.Hardware(vram_gb=3.5, vram_total_gb=4.0, ram_gb=8.0,
                       ram_total_gb=8.0, disk_gb=1.0, fp16=True)
    v = hw.verdict(KLEIN, tiny, 512, 512, pending_bytes=5_300_000_000)
    check(v.grade == hw.NO_DISK, "5.3 GB does not fit in 1 GB of free space")
    check(not v.runnable, "and that is the one thing the app refuses to start")
    check("1 GB left" in v.summary, "the message says how much room there is")


def test_no_fp16_doubles_the_working_set():
    print("\nfp16 changes the answer")
    pascal = hw.working_gb(KLEIN, 768, 768, PASCAL) - hw.WORK_FIXED_GB
    modern = hw.working_gb(KLEIN, 768, 768, BIG) - hw.WORK_FIXED_GB
    check(abs(pascal - 2 * modern) < 1e-6,
          "a card with no fp16 needs twice the *activation* working set — "
          "every activation is full precision")
    # The fixed term is buffers, not activations, so precision does not touch
    # it and it is subtracted above rather than doubled.
    check(hw.working_gb(KLEIN, 768, 768, PASCAL)
          - hw.working_gb(KLEIN, 768, 768, BIG) == pascal - modern,
          "and the fixed term is the same on both, being buffers rather than "
          "activations")


def test_busy_gpu_does_not_condemn_the_card():
    print("\na busy GPU is a temporary state, not a verdict on the hardware")
    busy = hw.Hardware(vram_gb=0.2, vram_total_gb=4.29, ram_gb=8.1,
                       ram_total_gb=12.5, disk_gb=32.0, fp16=False)
    check(abs(busy.busy_gb - 4.09) < 0.01, "busy_gb reports what is held")
    v = hw.verdict(KLEIN, busy, 512, 512)
    check(v.grade != hw.TOO_BIG,
          "something else holding 95% of the card is not the card being small")
    check(v.grade == hw.BUSY and "in use elsewhere" in v.summary,
          "it says so, because closing that application is the fix")
    check(v.runnable, "and it is still startable")

    # The same card idle: the only thing holding VRAM is the compositor, which
    # nobody can close, so a size genuinely beyond the card is reported as the
    # card being small rather than as something else hogging it. 1024x1024 used
    # to be that size and is not any more, which is what flash attention bought.
    idle = hw.verdict(KLEIN, PASCAL, 2048, 2048)
    check(idle.grade == hw.TOO_BIG,
          "a normal desktop's few hundred MB is part of the card's real size")
    check(hw.verdict(KLEIN, PASCAL, 1024, 1024).grade != hw.TOO_BIG,
          "and 1024x1024 is no longer in that category")


def test_unknown_hardware_is_permissive():
    print("\nan unmeasurable machine is not a blocked one")
    v = hw.verdict(KLEIN, hw.Hardware(), 512, 512)
    check(v.grade == hw.UNKNOWN, "nothing measured means nothing claimed")
    check(v.runnable, "and the user can still press the button")


if __name__ == "__main__":
    raise SystemExit(run(
        test_matches_the_real_measurements,
        test_references_cost_what_they_were_measured_to_cost,
        test_predicts_the_real_outcomes,
        test_the_offload_ceiling_matches_the_runs,
        test_grades_are_ordered, test_offload_flags,
        test_disk_is_the_only_hard_stop, test_no_fp16_doubles_the_working_set,
        test_busy_gpu_does_not_condemn_the_card,
        test_unknown_hardware_is_permissive))
