"""Formatting shared by the three pages.

local-upscaler kept these in `setup_page` and imported them from there into the
progress and result pages, which works but puts a dependency on the largest UI
module in the two smallest. They are pure functions of numbers; they live on
their own here.
"""

from __future__ import annotations


def human_bytes(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e3:.0f} kB"


def human_time(seconds: float) -> str:
    """A duration at a precision the number deserves.

    The ranges here are wider than the upscaler's because this app's are: a
    four-step Klein run is a couple of minutes and a disk-streamed Qwen run is
    hours, and "142 s" for one and "7431 s" for the other would both be
    technically correct and useless.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    return f"{hours:.1f} h" if hours < 10 else f"{hours:.0f} h"


def plural(n: int, singular: str, plural_form: str = "") -> str:
    return f"{n} {singular if n == 1 else (plural_form or singular + 's')}"
