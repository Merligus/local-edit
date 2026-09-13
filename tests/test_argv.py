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
    check(all(r.source_as_ref for r in cat.RECIPES),
          "every recipe in the catalog today is an edit model, so every source "
          "image travels as reference 1")
    # The generative wiring is still reachable and still tested, because the
    # recipes that need it are one catalog entry away from returning.
    import dataclasses
    gen = dataclasses.replace(KLEIN, id="gen", source_as_ref=False)
    argv = argv_for(gen, init_image=Path("src.png"))
    check("-i" in argv and "--strength" in argv,
          "a generative recipe would get -i and a denoising strength")
    check("-i" not in argv_for(KLEIN, ref_images=(Path("a.png"),)),
          "an edit model never gets an init image")


def test_memory_flags_and_flash_attention():
    print("\nmemory flags, and flash attention only where it exists")
    argv = argv_for(KLEIN, memory=("--offload-to-cpu",), backend="vulkan")
    check("--offload-to-cpu" in argv, "the verdict's flags are passed through")
    check("--diffusion-fa" in argv,
          "flash attention IS passed on Vulkan. Upstream's notes list the "
          "supported backends as CPU, CUDA/ROCm and Metal, so an earlier "
          "version excluded Vulkan on trust — measured, it is 6% faster and "
          "uses a third less memory there")
    check("--diffusion-fa" in argv_for(KLEIN, backend="cuda"),
          "and on CUDA")
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
    # No recipe uses this today — the engine cannot load PhotoMaker's weights,
    # see catalog.py — but the wiring is kept and kept tested, because the
    # failure is an engine regression and the tier returns when it is fixed.
    import dataclasses
    pm = dataclasses.replace(KLEIN, id="pm", engine=cat.ENGINE_CLI)
    argv = argv_for(pm, pm_id_dir=Path("/tmp/ids"))
    check("--pm-id-images-dir" in argv, "the id directory is passed")
    check(cat.ENGINE_CLI != cat.ENGINE_SERVER,
          "and a CLI recipe exists as a concept: --pm-id-images-dir is a "
          "directory fixed at server startup, not something a request can carry")


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


def test_cuda_vae_tile_size():
    print("\nthe CUDA VAE tile is chosen from the working set, and is real")
    def tile(working_gb, backend="cuda"):
        argv = binary.memory_args(("--offload-to-cpu",), backend,
                                  working_gb=working_gb)
        if "--vae-tile-size" not in argv:
            return None
        return argv[argv.index("--vae-tile-size") + 1]

    check(tile(0.62) == "8x8", "a roomy run gets the 8x8 tile")
    check(tile(1.14) == "4x4",
          "a tight one drops to 4x4 — measured: 512px with three references "
          "fails at 8x8 and finishes at 4x4")
    check(tile(0.0) == "8x8", "an unknown working set assumes the roomier tile")
    check(tile(5.0, "vulkan") is None,
          "and none of this applies to Vulkan, which decodes the same VAE on "
          "the same card with no tiling at all")

    # The value that was there before is the one value that does nothing: the
    # engine doubles it and clamps to the image, so 16x16 became one 32x32
    # latent tile covering a 512px frame. The 848 MB buffer that resulted is
    # why a reference edit could not run at any size on CUDA.
    check("16x16" not in binary.CUDA_VAE_ARGS,
          "16x16 is gone — it resolved to a single tile, which is not tiling")
    for name, args in (("CUDA_VAE_ARGS", binary.CUDA_VAE_ARGS),
                       ("CUDA_VAE_ARGS_TIGHT", binary.CUDA_VAE_ARGS_TIGHT)):
        check(args[0] == "--vae-tiling" and args[1] == "--vae-tile-size",
              f"{name} switches tiling on as well as sizing it")


def test_the_tile_matches_what_was_measured():
    print("\nand the boundary sits where the measurements put it")
    from local_edit.engine import hardware as hw
    machine = hw.Hardware(vram_gb=3.99, vram_total_gb=4.29, ram_gb=10.6,
                          ram_total_gb=12.5, disk_gb=25.0, fp16=False)
    # (size, references, the tile that was measured to work on this card)
    for px, refs, want in ((512, 1, "8x8"), (512, 2, "8x8"),
                           (512, 3, "4x4"), (768, 2, "4x4")):
        v = hw.verdict(KLEIN, machine, px, px, references=refs)
        argv = binary.build_server_argv(EXE, KLEIN, MODELS, port=1,
                                        memory=v.flags, backend="cuda",
                                        working_gb=v.working_gb)
        got = argv[argv.index("--vae-tile-size") + 1]
        check(got == want,
              f"{px}px with {refs} reference(s): {got} (measured: {want})")


if __name__ == "__main__":
    raise SystemExit(run(
        test_model_flags_match_roles, test_references_repeat,
        test_edit_models_do_not_get_an_init_image,
        test_memory_flags_and_flash_attention, test_sampling_params,
        test_photomaker, test_server_body_matches, test_redaction,
        test_server_argv_has_no_prompt, test_cuda_vae_tile_size,
        test_the_tile_matches_what_was_measured))
