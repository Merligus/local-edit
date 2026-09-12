# TODO

Roughly in order of how much they would improve the app.

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
