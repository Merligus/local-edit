"""Tests for what actually reaches the engine.

Run:  python3 tests/test_argv.py

Two shapes, one meaning: the CLI path builds an argv and the server path builds
a JSON body, and a difference between them is a bug that only shows up for
whichever recipe uses the other path. Both are checked here against the flags
`sd-cli --help` really prints.
"""

import json
from pathlib import Path

from _harness import check, run                                    # noqa: E402
from local_edit.engine import binary, catalog as cat               # noqa: E402
from local_edit.engine import client as cl                         # noqa: E402

EXE = Path("/opt/sd/sd-cli")
MODELS = Path("/models")
KLEIN = cat.get("flux2-klein-4b-q4")
KONTEXT = cat.get("kontext-q3")
SD15 = cat.get("sd15-ip")
PM = cat.get("sdxl-photomaker")


def argv_for(recipe, **kw):
    kw.setdefault("prompt", "make it rainy")
    kw.setdefault("output", Path("/tmp/out.png"))
    return binary.build_cli_argv(EXE, recipe, MODELS, **kw)


def pairs(argv):
    """Flags with their values, so order-independent assertions are possible."""
    out = {}
    for i, token in enumerate(argv):
        if token.startswith("-"):
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            out.setdefault(token, []).append("" if nxt.startswith("-") else nxt)
    return out


def test_model_flags_match_roles():
    print("\nevery file is passed under the flag its role names")
    for recipe in cat.RECIPES:
        argv = argv_for(recipe)
        got = pairs(argv)
        for f in recipe.files:
            check(f.flag() in got, f"{recipe.id}: {f.role} uses {f.flag()}")
            check(any(f.filename() in v for v in got[f.flag()]),
                  f"{recipe.id}: {f.flag()} points at {f.filename()}")


def test_references_repeat():
    print("\nmultiple references are multiple -r flags")
    argv = argv_for(KLEIN, ref_images=(Path("a.png"), Path("b.png"),
                                       Path("c.png")))
    check(argv.count("-r") == 3, "three references, three -r flags")
    check("--increase-ref-index" in argv,
          "and they are numbered, or the prompt's 'image 2' means nothing")

    single = argv_for(KLEIN, ref_images=(Path("a.png"),))
    check(single.count("-r") == 1, "one reference, one -r")
    check("--increase-ref-index" not in single,
          "numbering one reference is noise")

    check("-r" not in argv_for(KLEIN), "no references, no -r")


def test_edit_models_do_not_get_an_init_image():
    print("\nedit models take the image as a reference, not as an init image")
    # Passing an edit model --init-img produces a plausible picture that ignores
    # the instruction, which reads as a bad model rather than a wiring mistake.
    check(KLEIN.source_as_ref and KONTEXT.source_as_ref,
          "FLUX.2 and Kontext are edit models")
    check(not SD15.source_as_ref and not PM.source_as_ref,
          "SD1.5 and SDXL are generative, and re-noise an init image")
    argv = argv_for(SD15, init_image=Path("src.png"))
    check("-i" in argv and "--strength" in argv,
          "a generative model gets -i and a denoising strength")


def test_memory_flags_and_flash_attention():
    print("\nmemory flags, and flash attention only where it exists")
    argv = argv_for(KLEIN, memory=("--offload-to-cpu",), backend="vulkan")
    check("--offload-to-cpu" in argv, "the verdict's flags are passed through")
    check("--diffusion-fa" not in argv,
          "flash attention is NOT passed on Vulkan — ggml has no kernel for it, "
          "and this app's only GPU is Vulkan")
    check("--diffusion-fa" in argv_for(KLEIN, backend="cuda"),
          "but it is passed on CUDA, where it exists")
    check("--max-vram" in argv_for(KLEIN, max_vram_gb=3.0),
          "an explicit VRAM budget is passed when set")
    check("--max-vram" not in argv, "and omitted when not")


def test_sampling_params():
    print("\nsampling parameters come from the recipe unless overridden")
    got = pairs(argv_for(KLEIN))
    check(got["--steps"] == [str(KLEIN.steps)], "steps default to the recipe's")
    check(got["--sampling-method"] == [KLEIN.sampling_method], "so does the sampler")
    check(pairs(argv_for(KLEIN, steps=12))["--steps"] == ["12"],
          "an override wins")
    check("--guidance" in pairs(argv_for(KONTEXT)),
          "Kontext gets its distilled guidance")
    check("--guidance" not in pairs(argv_for(KLEIN)),
          "a model with no guidance input does not")
    check("--flow-shift" in pairs(argv_for(cat.get("qwen-edit-2509-q2"))),
          "Qwen-Image-Edit gets its flow shift")


def test_photomaker():
    print("\nPhotoMaker's identity directory")
    argv = argv_for(PM, pm_id_dir=Path("/tmp/ids"))
    check("--pm-id-images-dir" in argv, "the id directory is passed")
    check(PM.engine == cat.ENGINE_CLI,
          "which is why PhotoMaker is a CLI recipe: the flag is a directory "
          "path fixed at server startup, not something a request can carry")


def test_server_body_matches():
    print("\nthe server body says the same thing as the argv")
    req = cl.request_from_recipe(KLEIN, "make it rainy", width=768, height=768,
                                 seed=7, ref_images=["aaa", "bbb"])
    body = json.loads(json.dumps(req.to_json()))       # must be serialisable
    check(body["ref_images"] == ["aaa", "bbb"], "references travel as an array")
    check(body["increase_ref_index"] is True, "and are numbered, as in the argv")
    check(body["sample_params"]["sample_steps"] == KLEIN.steps,
          "steps match the recipe")
    check(body["sample_params"]["guidance"]["txt_cfg"] == KLEIN.cfg_scale,
          "guidance is nested where the server reads it")
    check(body["seed"] == 7 and body["width"] == 768, "size and seed are top level")
    check("init_image" not in body, "absent fields are omitted, not sent empty")

    kb = cl.request_from_recipe(KONTEXT, "x", width=512, height=512)
    check(kb.to_json()["sample_params"]["guidance"]["distilled_guidance"]
          == KONTEXT.guidance,
          "Kontext's distilled guidance reaches the server too")


def test_redaction():
    print("\nlogging a request does not dump megabytes of base64")
    req = cl.request_from_recipe(KLEIN, "x", width=512, height=512,
                                 init_image="A" * 10_000,
                                 ref_images=["B" * 10_000])
    red = json.dumps(req.redacted())
    check(len(red) < 500, "the redacted form is short enough to log")
    check("AAAA" not in red and "BBBB" not in red, "and holds no image data")


def test_server_argv_has_no_prompt():
    print("\nthe server's own argv carries no per-edit state")
    argv = binary.build_server_argv(EXE, KLEIN, MODELS, port=9999,
                                    memory=("--offload-to-cpu",))
    check("--listen-port" in argv and "9999" in argv, "it is told where to listen")
    check("-p" not in argv and "--prompt" not in argv,
          "but not what to generate — that is the point of a warm server")
    check("--offload-to-cpu" in argv,
          "memory flags are startup flags and must be here")


if __name__ == "__main__":
    raise SystemExit(run(
        test_model_flags_match_roles, test_references_repeat,
        test_edit_models_do_not_get_an_init_image,
        test_memory_flags_and_flash_attention, test_sampling_params,
        test_photomaker, test_server_body_matches, test_redaction,
        test_server_argv_has_no_prompt))
