"""Tests for role composition — how reference slots become a prompt.

Run:  python3 tests/test_prompt.py

The numbering is the part worth guarding. `--increase-ref-index` assigns indices
in the order references are passed, so if the list order and the prompt's
numbers ever disagree, "the jacket in image 2" silently points at the wrong
picture and the result looks like a bad model rather than a bug.
"""

from pathlib import Path

from _harness import check, run                                    # noqa: E402
from local_edit.engine import prompt as P                          # noqa: E402

FACE = P.Reference(Path("face.jpg"), P.ROLE_FACE)
COAT = P.Reference(Path("jacket.png"), P.ROLE_CLOTHING)
CITY = P.Reference(Path("paris.jpg"), P.ROLE_SCENE)


def test_numbering_follows_order():
    print("\nnumbering follows list order")
    text = P.compose([FACE, COAT, CITY], "make them ride a motorcycle")
    check("person in image 1" in text, "the first reference is image 1")
    check("clothing from image 2" in text, "the second is image 2")
    check("location shown in image 3" in text, "the third is image 3")

    reordered = P.compose([CITY, COAT, FACE], "make them ride a motorcycle")
    check("location shown in image 1" in reordered,
          "reordering renumbers — the scene becomes image 1")
    check("person in image 3" in reordered,
          "reordering renumbers — the face becomes image 3")
    check(reordered != text, "a reorder really does change what is sent")


def test_instruction_goes_last():
    print("\nthe instruction goes last")
    text = P.compose([FACE], "make it rainy")
    check(text.rstrip().endswith("make it rainy."),
          "what the user typed is the tail of the prompt")
    check(text.index("image 1") < text.index("make it rainy"),
          "the references are established before the thing to do with them")


def test_punctuation():
    print("\npunctuation")
    check(P.compose([], "make it rainy") == "make it rainy.",
          "a bare instruction gains a full stop")
    check(P.compose([], "make it rainy?") == "make it rainy?",
          "an instruction that has punctuation keeps it")
    check(P.compose([], "") == "", "no instruction and no roles is empty")
    check(P.compose([P.Reference(Path("x.png"), P.ROLE_PLAIN)], "") == "",
          "a plain reference contributes no wording of its own")


def test_families_differ():
    print("\nfamilies get the phrasing they were trained on")
    flux = P.compose([FACE], "x", "flux2")
    qwen = P.compose([FACE], "x", "qwen-edit")
    check("image 1" in flux, "FLUX.2 hears 'image 1'")
    check("picture 1" in qwen, "Qwen-Image-Edit hears 'picture 1'")
    check(P.compose([FACE], "x", "no-such-family") == flux,
          "an unknown family falls back to the default rather than losing the "
          "reference entirely")


def test_photomaker_trigger():
    print("\nPhotoMaker's trigger word")
    text, warn = P.photomaker_prompt("a woman riding a motorcycle")
    check(text == "a woman img riding a motorcycle",
          "the trigger is inserted after the class word")
    check(not warn, "a prompt that names the subject needs no warning")

    text, warn = P.photomaker_prompt("a man img on a bike")
    check(text == "a man img on a bike", "an existing trigger is left alone")
    check(not warn, "and needs no warning")

    text, warn = P.photomaker_prompt("make the scenery rainy")
    check(text == "make the scenery rainy",
          "a prompt with no subject word is NOT rewritten — the app does not "
          "guess whether the person in someone's photographs is a man")
    check("woman" in warn and "img" in warn,
          "instead it says what to add")

    _, warn = P.photomaker_prompt("")
    check(bool(warn), "an empty prompt is flagged too")


def test_limits_are_announced():
    print("\ntrimming is announced, never silent")
    kept, note = P.apply_limits([FACE, COAT, CITY], 1, False, "SD 1.5")
    check(len(kept) == 1, "an IP-Adapter recipe keeps one reference")
    check(kept[0] is FACE, "and it keeps the first one, which the user put first")
    check("face.jpg" in note and "2 more" in note,
          "the note names what survived and how many did not")

    kept, note = P.apply_limits([FACE, COAT, CITY], 3, True, "FLUX.2")
    check(len(kept) == 3 and not note,
          "a recipe that can use them all says nothing")

    kept, note = P.apply_limits([], 1, False, "SD 1.5")
    check(not kept and not note, "no references, no complaint")


def test_roles_are_complete():
    print("\nevery role is usable")
    for role in P.ROLES:
        check(role in P.ROLE_LABELS, f"{role} has a label for the dropdown")
        check(role in P.ROLE_HINTS, f"{role} has a tooltip")
        text = P.compose([P.Reference(Path("x.png"), role)], "do a thing")
        check(bool(text), f"{role} composes to something")


if __name__ == "__main__":
    raise SystemExit(run(
        test_numbering_follows_order, test_instruction_goes_last,
        test_punctuation, test_families_differ, test_photomaker_trigger,
        test_limits_are_announced, test_roles_are_complete))
