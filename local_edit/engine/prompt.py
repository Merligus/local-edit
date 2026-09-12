"""Turning reference images with roles into a prompt the model understands.

The models in this catalog take references as a numbered list and expect the
prompt to refer to them by number: *"the person in image 1, wearing the jacket
from image 2"*. That is a precise interface and a poor one to type, so the UI
gives each reference a **role** — Face, Clothing, Scene, Style, Object — and
this module writes the sentence that wires it up.

Two rules keep that from being magic:

* The composed prompt is **shown**, not hidden, and the user can take it over
  verbatim. A template that gets the phrasing wrong for one model is then a
  five-second fix rather than a mystery.
* Numbering follows **list order**, because that is what `--increase-ref-index`
  does. Reordering the list renumbers the prompt, which is why the reference
  rows are drag-reorderable and why `compose` takes a sequence rather than a
  mapping.

Phrasing is per family, because the families genuinely differ. Qwen-Image-Edit
was trained on "Picture 1 / Picture 2"; FLUX.2 and Kontext on "image 1".
PhotoMaker does not take numbered references at all — its identity images go to
a directory, and it instead requires a class word followed by the literal
trigger `img` somewhere in the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROLE_PLAIN = "plain"
ROLE_SUBJECT = "subject"
ROLE_FACE = "face"
ROLE_CLOTHING = "clothing"
ROLE_SCENE = "scene"
ROLE_STYLE = "style"
ROLE_OBJECT = "object"

#: Order is the order of the role dropdown.
ROLES = (ROLE_PLAIN, ROLE_SUBJECT, ROLE_FACE, ROLE_CLOTHING, ROLE_SCENE,
         ROLE_STYLE, ROLE_OBJECT)

ROLE_LABELS = {
    ROLE_PLAIN: "Reference",
    ROLE_SUBJECT: "Subject",
    ROLE_FACE: "Face",
    ROLE_CLOTHING: "Clothing",
    ROLE_SCENE: "Scene",
    ROLE_STYLE: "Style",
    ROLE_OBJECT: "Object",
}

ROLE_HINTS = {
    ROLE_PLAIN: "Passed through with no wording of its own — refer to it "
                "yourself as image N.",
    ROLE_SUBJECT: "Who or what the edit is about.",
    ROLE_FACE: "Keep this person's face and identity.",
    ROLE_CLOTHING: "Take the clothing from this image.",
    ROLE_SCENE: "Put the subject in this place.",
    ROLE_STYLE: "Match this image's look, palette and lighting.",
    ROLE_OBJECT: "Include this object.",
}

#: `{n}` is the 1-based reference index; `{noun}` is the family's word for an
#: image. A role missing from a family's table falls back to `_DEFAULT`.
_DEFAULT = {
    ROLE_SUBJECT: "The main subject is the person in {noun} {n}.",
    ROLE_FACE: "Keep the face and identity of the person in {noun} {n}.",
    ROLE_CLOTHING: "Dress the subject in the clothing from {noun} {n}.",
    ROLE_SCENE: "Place the subject in the location shown in {noun} {n}.",
    ROLE_STYLE: "Match the visual style, palette and lighting of {noun} {n}.",
    ROLE_OBJECT: "Include the object from {noun} {n}.",
    ROLE_PLAIN: "",
}

_QWEN = dict(_DEFAULT, **{
    ROLE_FACE: "Keep the face and identity of the person in {noun} {n} exactly.",
    ROLE_CLOTHING: "Put the clothing from {noun} {n} on the person.",
    ROLE_SCENE: "Use the scene in {noun} {n} as the background.",
})

#: Per-family templates and the noun each family was trained to hear.
_FAMILIES = {
    "flux2": (_DEFAULT, "image"),
    "kontext": (_DEFAULT, "image"),
    "qwen-edit": (_QWEN, "picture"),
}
_FALLBACK = (_DEFAULT, "image")

#: PhotoMaker's trigger. Upstream documents `man`, `woman`, `girl` and `boy` as
#: the class words it was trained on; the wider set is only used to find where
#: to insert the trigger in a prompt the user already wrote.
PM_TRIGGER = "img"
PM_CLASS_WORDS = ("man", "woman", "girl", "boy")
_PM_NOUNS = PM_CLASS_WORDS + ("person", "guy", "lady", "child", "kid",
                              "male", "female", "he", "she", "they")


@dataclass(frozen=True)
class Reference:
    """One reference image and what it is for."""

    path: Path
    role: str = ROLE_PLAIN

    def label(self) -> str:
        return ROLE_LABELS.get(self.role, ROLE_LABELS[ROLE_PLAIN])


def _templates(family: str) -> tuple[dict, str]:
    return _FAMILIES.get(family, _FALLBACK)


def compose(references: "list[Reference] | tuple[Reference, ...]",
            instruction: str, family: str = "flux2") -> str:
    """The prompt actually sent, from the role slots and what the user typed.

    The instruction goes **last**. Everything before it establishes what the
    references are; the instruction is the thing to do with them, and these
    models weight the tail of the prompt more heavily.
    """
    templates, noun = _templates(family)
    parts: list[str] = []
    for index, ref in enumerate(references, start=1):
        template = templates.get(ref.role, "")
        if template:
            parts.append(template.format(n=index, noun=noun))

    instruction = (instruction or "").strip()
    if instruction:
        parts.append(instruction if instruction[-1] in ".!?"
                     else instruction + ".")
    return " ".join(parts)


# ------------------------------------------------------------- photomaker
def photomaker_prompt(instruction: str) -> tuple[str, str]:
    """`(prompt, warning)` for PhotoMaker, which needs a class word plus `img`.

    Deliberately does not invent the class word when there is none. PhotoMaker
    wants "a man img on a motorcycle", and the app cannot know whether the
    person in someone's photographs is a man — guessing would be both wrong and
    presumptuous. So a prompt that already names one gets the trigger inserted,
    and a prompt that does not gets a warning the user can act on. The composed
    prompt is on screen either way, so the fix is visible.
    """
    text = (instruction or "").strip()
    if not text:
        return "", (f"PhotoMaker needs a prompt containing one of "
                    f"{', '.join(PM_CLASS_WORDS)} followed by "
                    f"'{PM_TRIGGER}' — for example 'a woman {PM_TRIGGER} "
                    f"riding a motorcycle'.")

    words = text.split()
    lowered = [w.strip(".,!?;:'\"").lower() for w in words]
    if PM_TRIGGER in lowered:
        return text, ""

    for i, word in enumerate(lowered):
        if word in _PM_NOUNS:
            words.insert(i + 1, PM_TRIGGER)
            return " ".join(words), ""

    return text, (f"PhotoMaker only uses the reference faces when the prompt "
                  f"names the subject — add one of {', '.join(PM_CLASS_WORDS)} "
                  f"followed by '{PM_TRIGGER}'.")


# ----------------------------------------------------------------- limits
def apply_limits(references: "list[Reference] | tuple[Reference, ...]",
                 max_refs: int, ref_images: bool,
                 recipe_label: str) -> tuple[list[Reference], str]:
    """Trim the list to what the recipe can use, and say so.

    Silence here would be the worst outcome: an IP-Adapter recipe takes exactly
    one reference image, so a user who lined up a face, a jacket and a street
    would get a picture that used the face and quietly ignored the rest, with
    nothing to suggest why.
    """
    refs = list(references)
    if not refs:
        return refs, ""
    if len(refs) <= max_refs:
        return refs, ""

    dropped = len(refs) - max_refs
    kept = refs[:max_refs]
    if not ref_images and max_refs == 1:
        why = (f"{recipe_label} takes a single reference image, so only "
               f"{kept[0].path.name} is used")
    else:
        why = (f"{recipe_label} uses at most {max_refs} reference"
               f"{'s' if max_refs != 1 else ''}")
    return kept, (f"{why} — {dropped} more "
                  f"{'is' if dropped == 1 else 'are'} ignored.")
