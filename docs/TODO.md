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

0.5 **Try a 4-bit text encoder in diffusers (bitsandbytes), and re-run the
   engine comparison.** This is the one untested branch of "is sd.cpp actually
   the right engine here", and it is the branch most likely to change the
   answer.

   diffusers lost that comparison on this machine for one structural reason: it
   quantises the diffusion transformer through GGUF but has **no quantisation
   path for a transformers text encoder**, so 8 GB of fp16 Qwen3-4B has to live
   somewhere. On 4 GB of VRAM it cannot be the GPU; in 11 GB of RAM it thrashed
   into 9 GB of zram swap and never finished loading. sd.cpp runs the same
   encoder at 4-bit in 2.5 GB on the GPU, which is the whole of its advantage.

   `bitsandbytes` would close exactly that gap:

       from transformers import BitsAndBytesConfig
       q = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16)
       Flux2KleinPipeline.from_pretrained(REPO, transformer=gguf_transformer,
                                          quantization_config=q)

   The doubt is hardware: bitsandbytes documents 4-bit (NF4/FP4) as wanting
   compute capability 7.5 or newer, and this card is 6.1. 8-bit is supported
   further back. Either would bring the encoder under 4 GB. If it works,
   diffusers becomes viable here and brings LoRAs, ControlNets, inpainting and
   the adapters this engine cannot load (see item 0) — which would be a real
   reason to reconsider the engine rather than a theoretical one.

   **The pieces are no longer on disk** — the 15 GB venv and HuggingFace cache
   were deleted to reclaim space, so this starts from nothing:

       python3 -m venv venv
       venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
       venv/bin/pip install diffusers transformers accelerate bitsandbytes \
                            gguf numpy pillow sentencepiece protobuf
       # then snapshot_download black-forest-labs/FLUX.2-klein-4B, allow_patterns
       # ["model_index.json","scheduler/*","tokenizer/*","text_encoder/*","vae/*"]
       # (~8 GB; the 7.75 GB fp16 transformer is not needed — use the Q4_0 GGUF
       #  the app already has)

   Budget about 25 minutes of downloading and 20 GB. Set `TMPDIR` somewhere on
   the real disk: `/tmp` here is tmpfs and pip will fill it. The baseline to
   beat is sd.cpp at **32 s** on CUDA for 512x512, 4 steps, seed 42 — not the
   95 s Vulkan figure this item was originally written against.

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

## 0.6 — re-measure the reference cost on the new machine

`hardware.OFFLOAD_VRAM_SHARE = 0.45` and the `CUDA_VAE_ARGS` tile sizes were
calibrated against eleven runs on one 4 GB GTX 1050 Ti with the CUDA build. The
working-set model itself (total megapixels = output x (1 + references)) should
carry over, being a property of how the model concatenates reference tokens;
the two constants are allocator behaviour and probably will not.

Re-run the sweep and re-fit: output 512/768/1024, references 0..3, reading
`flux compute buffer size` from `-v` output. Vulkan needs the same treatment —
it decodes this VAE with no tiling at all on the same card, so `CUDA_VAE_ARGS`
may not be CUDA-specific so much as allocator-specific.
