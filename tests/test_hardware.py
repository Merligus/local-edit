"""Tests for the hardware grading.

Run:  python3 tests/test_hardware.py

Graded against synthetic machines, so the result does not depend on what else is
using the GPU while the tests run — which is itself a bug this file exists to
prevent: an early draft judged "can this card ever fit the working set" against
*live free* VRAM, and relabelled the entire catalog "needs a bigger GPU" the one
time it was run during a benchmark.

The two anchor cases are real measurements on a GTX 1050 Ti:
  512x512  -> 0.49 GB working set, ran fine
  1024x1024 -> 2.76 GB working set, died at segment 15/27 out of device memory
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
    for side, measured in ((512, 0.491), (1024, 2.763)):
        predicted = hw.working_gb(KLEIN, side, side, PASCAL)
        check(abs(predicted - measured) / measured < 0.05,
              f"{side}x{side}: predicted {predicted:.2f} GB vs measured "
              f"{measured:.3f} GB (within 5%)")


def test_predicts_the_real_outcomes():
    print("\nthe verdicts predict what actually happened")
    ok = hw.verdict(KLEIN, PASCAL, 512, 512)
    check(ok.grade == hw.OFFLOAD,
          "512x512 is graded runnable — and it ran")
    bad = hw.verdict(KLEIN, PASCAL, 1024, 1024)
    check(bad.grade == hw.TOO_BIG,
          "1024x1024 is graded too big — and it died out of device memory")
    check(bad.runnable,
          "but it is still startable: sd.cpp cuts the graph finer and the user "
          "is entitled to find out where their machine gives up")


def test_grades_are_ordered():
    print("\ngrades degrade in the right order as the card shrinks")
    big = hw.verdict(KLEIN, BIG, 1024, 1024)
    check(big.grade == hw.FITS, "a 24 GB card fits Klein outright")
    check(not big.flags, "and needs no memory flags")

    no_vram = hw.Hardware(vram_gb=3.5, vram_total_gb=4.0, ram_gb=16.0,
                          ram_total_gb=16.0, disk_gb=100.0, fp16=True)
    check(hw.verdict(KLEIN, no_vram, 512, 512).grade == hw.OFFLOAD,
          "weights that do not fit in VRAM but do fit in RAM offload")
    no_ram = hw.Hardware(vram_gb=3.5, vram_total_gb=4.0, ram_gb=2.0,
                         ram_total_gb=4.0, disk_gb=100.0, fp16=True)
    check(hw.verdict(KLEIN, no_ram, 512, 512).grade == hw.STREAM,
          "weights that fit in neither stream from disk")


def test_offload_flags():
    print("\neach grade carries the flags it implies")
    check("--offload-to-cpu" in hw.verdict(KLEIN, PASCAL, 512, 512).flags,
          "OFFLOAD passes --offload-to-cpu")
    stream = hw.Hardware(vram_gb=3.5, vram_total_gb=4.0, ram_gb=2.0,
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
    pascal = hw.working_gb(KLEIN, 768, 768, PASCAL)
    modern = hw.working_gb(KLEIN, 768, 768, BIG)
    check(abs(pascal - 2 * modern) < 1e-6,
          "a card with no fp16 needs twice the working set — every activation "
          "is full precision")


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
    # nobody can close, so 1024x1024 really is beyond this GPU.
    idle = hw.verdict(KLEIN, PASCAL, 1024, 1024)
    check(idle.grade == hw.TOO_BIG,
          "a normal desktop's few hundred MB is part of the card's real size")


def test_unknown_hardware_is_permissive():
    print("\nan unmeasurable machine is not a blocked one")
    v = hw.verdict(KLEIN, hw.Hardware(), 512, 512)
    check(v.grade == hw.UNKNOWN, "nothing measured means nothing claimed")
    check(v.runnable, "and the user can still press the button")


if __name__ == "__main__":
    raise SystemExit(run(
        test_matches_the_real_measurements, test_predicts_the_real_outcomes,
        test_grades_are_ordered, test_offload_flags,
        test_disk_is_the_only_hard_stop, test_no_fp16_doubles_the_working_set,
        test_busy_gpu_does_not_condemn_the_card,
        test_unknown_hardware_is_permissive))
