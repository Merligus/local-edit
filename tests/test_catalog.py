"""Tests for the recipe catalog.

Run:  python3 tests/test_catalog.py

Pure data, so this runs with no network, no engine and no display. What it
guards is the class of mistake that only surfaces much later as a failed
download or an unreadable ggml error: a duplicated id, a file with no flag to
pass it under, two files that would overwrite each other on disk, or an edit
model quietly marked as taking an init image.
"""

from _harness import check, run                                    # noqa: E402
from local_edit.engine import catalog as cat                       # noqa: E402


def test_identity():
    print("\nidentity")
    ids = [r.id for r in cat.RECIPES]
    check(len(ids) == len(set(ids)), "recipe ids are unique")
    labels = [r.label for r in cat.RECIPES]
    check(len(labels) == len(set(labels)), "labels are unique")
    check(cat.exists(cat.DEFAULT_RECIPE_ID),
          f"the default recipe {cat.DEFAULT_RECIPE_ID!r} exists")
    check(cat.get("no-such-recipe").id == cat.DEFAULT_RECIPE_ID,
          "an unknown id falls back to the default instead of raising")


def test_fully_described():
    print("\nevery recipe is fully described")
    for r in cat.RECIPES:
        check(bool(r.label and r.blurb and r.author and r.licence
                   and r.url.startswith("https://")),
              f"{r.id} has a label, blurb, author, licence and URL")
        check(len(r.blurb) > 60,
              f"{r.id}'s blurb says something useful, not just a restated name")
        check(r.steps >= 1 and r.cfg_scale >= 0,
              f"{r.id} has usable sampling defaults")


def test_files_are_addressable():
    print("\nevery file can be passed to the engine and stored on disk")
    for r in cat.RECIPES:
        roles = [f.role for f in r.files]
        check(len(roles) == len(set(roles)),
              f"{r.id} gives each file a distinct role (one flag each)")
        for f in r.files:
            check(f.role in cat.ROLE_FLAGS, f"{r.id}: {f.role} maps to a flag")
            check(f.role in cat.ROLE_DIRS, f"{r.id}: {f.role} maps to a directory")
            check(f.size > 1_000_000,
                  f"{r.id}: {f.filename()} has a plausible byte count")
            check(f.url().startswith("https://huggingface.co/"),
                  f"{r.id}: {f.filename()} resolves to a HuggingFace URL")
        check(any(f.role in ("diffusion", "checkpoint") for f in r.files),
              f"{r.id} has something to denoise with")


def test_no_filename_collisions():
    print("\nno two distinct files share a destination")
    # Two repos' `model.safetensors` landing in the same directory would leave
    # whichever downloaded second silently standing in for the first.
    seen = {}
    for r in cat.RECIPES:
        for f in r.files:
            key = (f.subdir(), f.filename())
            other = seen.setdefault(key, f)
            check(other.repo == f.repo and other.path == f.path,
                  f"{f.subdir()}/{f.filename()} means one file, not two "
                  f"({other.repo} vs {f.repo})")


def test_derived_sizes():
    print("\nsizes are derived, never typed twice")
    for r in cat.RECIPES:
        check(r.weights_bytes() == sum(f.size for f in r.files),
              f"{r.id}'s download size is the sum of its files")
    check(0 < cat.get("flux2-klein-4b-q4").weights_gb() < 7,
          "the default recipe is small enough to be a sane default")


def test_reference_handling():
    print("\nreference handling matches what each family supports")
    for r in cat.RECIPES:
        check(r.max_refs >= 1, f"{r.id} accepts at least one reference")
        if not r.ref_images:
            check(not r.source_as_ref,
                  f"{r.id} takes no --ref-image, so its source must be an "
                  f"init image")
            check(r.uses_ip_adapter() or r.uses_photomaker(),
                  f"{r.id} has some other way to take a reference")
        if r.uses_ip_adapter():
            check(r.has("clip_vision"),
                  f"{r.id} ships the CLIP-vision encoder --ip-adapter requires")
    # No recipe uses ENGINE_CLI today — the PhotoMaker tier that needed it was
    # removed because the engine cannot load its weights (see catalog.py) — but
    # the distinction still has to mean something, because the tier returns as
    # soon as that regression is fixed.
    check(cat.ENGINE_CLI != cat.ENGINE_SERVER,
          "the two engine modes are distinct")
    check(all(r.engine in (cat.ENGINE_CLI, cat.ENGINE_SERVER)
              for r in cat.RECIPES),
          "and every recipe names one of them")


def test_licences_are_stated():
    print("\nlicensing is visible")
    for r in cat.RECIPES:
        check(bool(r.licence.strip()), f"{r.id} states a licence")
    nc = [r.id for r in cat.RECIPES if "Non-Commercial" in r.licence]
    check(len(nc) > 0,
          f"non-commercial models are labelled as such ({len(nc)} of them)")


def test_readme_agrees():
    print("\nthe README's model table matches the catalog")
    # Documentation that drifts from the data is worse than none: the table is
    # how someone decides what to spend 34 GB of disk on, and nothing else in
    # this repo would notice it going stale.
    import re
    from pathlib import Path
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    for r in cat.RECIPES:
        row = re.search(rf"^\|\s*\*{{0,2}}`{re.escape(r.id)}`\*{{0,2}}\s*\|(.+)$",
                        readme, re.M)
        if row is None:
            check(False, f"{r.id} appears in the README's model table")
            continue
        cells = [c.strip().strip("*") for c in row.group(1).split("|")]
        refs, steps, size = cells[1], cells[2], cells[3]
        check(refs == str(r.max_refs),
              f"{r.id}: README says {refs} references, catalog says {r.max_refs}")
        check(steps == str(r.steps),
              f"{r.id}: README says {steps} steps, catalog says {r.steps}")
        stated = float(size.replace("GB", "").strip())
        check(abs(stated - r.weights_gb()) < 0.1,
              f"{r.id}: README says {stated} GB, catalog sums to "
              f"{r.weights_gb():.1f} GB")


if __name__ == "__main__":
    raise SystemExit(run(
        test_identity, test_fully_described, test_files_are_addressable,
        test_no_filename_collisions, test_derived_sizes,
        test_reference_handling, test_licences_are_stated,
        test_readme_agrees))
