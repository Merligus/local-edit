# TODO

Roughly in order of how much they would improve the app.

0. **Bring back the face/adapter tiers.** Two were written, downloaded and
   removed again — SD 1.5 + IP-Adapter, and SDXL + PhotoMaker. Both fail the
   same way, and it is an **engine regression, not a missing download**.

   Anything that carries a CLIP vision tower is refused by
   `master-859-7f410a3`. IP-Adapter asks for
   `cond_stage_model.transformer.vision_model.…`; PhotoMaker asks for
   `pmid.vision_model.…`; neither matches what the published weights contain,
   and all three candidate files (h94's image encoder, ComfyUI's
   `clip_vision_h`, PhotoMaker v1 and v2) are refused.

   The proof that it is a regression: **PhotoMaker v1 loads on
   `master-485-4ccce02` (January 2026)** —

       loading stacked ID embedding (PHOTOMAKER) model file from photomaker-v1
       Photomaker ID Stacking, taking 36485 ms
       PHOTOMAKER: start_merge_step: 0

   — and only then runs out of VRAM, because that build predates the segmented
   execution a 4 GB card needs for SDXL. So on this machine the feature is
   unavailable either way: the new engine cannot load it, and the old engine
   cannot run SDXL. On a 12 GB card the January build would work today, and
   `engine_path` in settings.json is how to point at one.

   Worth an upstream issue. For IP-Adapter the fix looks like one lookup:
   `name_conversion.cpp`'s `cond_model_name_map` is keyed on
   `transformer.vision_model.pre_layrnorm.*`, but by the time it runs the
   `clip_vision.` prefix has already become `cond_stage_model.transformer.`.

   Everything else for both tiers is still here and still tested: the
   `clip_vision` and `ip_adapter` roles, `--ip-adapter-image` and
   `--pm-id-images-dir` through both engines, the CLI execution mode, and the
   PhotoMaker trigger-word handling in `prompt.py`. Each tier is one catalog
   entry away from returning.

1. **Live preview during sampling.** The engine has `--preview tae` and
   `--preview-interval`, which write a cheap decode of the current latent every
   N steps. On a machine where a run is minutes, watching it converge is worth
   more than the progress bar — and it would let you cancel a bad seed at step 1
   instead of step 4.

2. **An inpainting mask.** `--mask` already exists in the engine. The UI would
   need a brush over the source image on the setup page. "Change only this part"
   is the most common edit there is.

3. **Batch: one prompt, several seeds.** `batch_count` is already in the request
   and the filmstrip already holds several results. The only reason it is not
   exposed is that four seeds at two minutes each is eight minutes, and the
   progress bar would need to say which one it is on.

4. **LoRAs.** The request carries a `lora` array and the engine has
   `--lora-model-dir`. Wants a small manager: they are per-family, and applying
   a FLUX LoRA to SD1.5 fails in a way that is hard to explain after the fact.

5. **Remember the last run's parameters per model.** Steps and guidance that
   suit Klein's four steps are wrong for Kontext's twenty-four, and switching
   back and forth currently means re-typing them.

6. **A persistent gallery.** The filmstrip dies with the session. Keeping each
   result with its prompt, seed and references would make "what did I do to get
   that one" answerable.

7. **`--bench` across sizes.** It currently measures 512² only, so the first run
   at 768 or 1024 is still an estimate. Sweeping the buckets `settings` uses
   would make every estimate measured from the start.

8. **ADetailer for faces.** The engine has `--ad-model` and an `adetailer` doc.
   Small faces are where these models are weakest and where a second pass at
   higher resolution helps most.

9. **Reference thumbnails that show the crop.** References are auto-resized by
   the engine; the row shows the original. For a tall portrait used as a scene
   reference, what the model sees and what the row shows differ.

10. **A CUDA build helper.** `docs/COMPATIBILITY.md` explains the cmake
    invocation; a `--build-engine` that runs it and pins the result would be
    friendlier than a copy-paste.
