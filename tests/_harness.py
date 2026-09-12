"""Shared scaffolding for the test scripts.

Same convention as local-upscaler and soundboard: standalone scripts, `PASS`/
`FAIL` per line, a non-zero exit when anything failed. No framework, so the
tests run on a machine with nothing installed but the app's own dependencies —
which for this project is the same machine that cannot install PyTorch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def run(*tests):
    for fn in tests:
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0
