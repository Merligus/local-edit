# TODO

Roughly in order of how much they would improve the app.

0. **Bring back an IP-Adapter tier.** There was one — SD 1.5 plus
   `ip-adapter-plus-face_sd15` — and it was removed because sd.cpp could not
   load a CLIP-ViT-H encoder for it on this build.

   Both published candidates (`h94/IP-Adapter::models/image_encoder/
   model.safetensors` and ComfyUI's `clip_vision_h.safetensors`) use
   HuggingFace's `vision_model.*` naming with the upstream `pre_layrnorm`
   typo. `src/name_conversion.cpp` has a `cond_model_name_map` that rewrites
   that spelling, but it is keyed on `transformer.vision_model.pre_layrnorm.*`
   and by the time it runs the `clip_vision.` prefix has already become
   `cond_stage_model.transformer.`, so it never matches:

       CLIP vision tensor 'cond_stage_model.transformer.vision_model.
       pre_layernorm.weight' not in model metadata
       model metadata validation failed

   Correcting the two names in the file makes it worse, not better — the file
   is then taken for an already-converted one and every tensor goes missing.
   So this is not a matter of finding the right download. Worth an upstream
   issue; the fix is one `find` on the unprefixed name. Everything else for the
   tier is still here: the `clip_vision` and `ip_adapter` roles, the
   `--ip-adapter-image` plumbing through both engines, and their tests. Adding
   the recipe back is one catalog entry.

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
