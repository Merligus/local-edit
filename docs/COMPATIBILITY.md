# Compatibility

## Hard dependencies

| | Why |
|---|---|
| Python 3.10+ | `X \| None` annotations throughout, behind `from __future__ import annotations`. Developed and tested on 3.14 |
| PySide6 | the entire interface |
| Pillow | decoding what the engine returns, and validating what you open |
| A Vulkan loader | `vulkan-icd-loader`, plus your GPU's driver |

```fish
sudo pacman -S --needed pyside6 python-pillow vulkan-icd-loader
```

`vulkan-tools` is optional but recommended: `hardware.py` reads free VRAM from
`vulkaninfo`, and falls back to `nvidia-smi` without it.

**No pip, no venv.** System packages only, as in local-upscaler and soundboard.

## Why there is no PyTorch

**Not because it cannot run here.** That was the original claim in this file and
it was wrong twice over. The correction matters more than the conclusion, so it
goes first:

```
$ python3 -m venv venv && venv/bin/pip install torch \
      --index-url https://download.pytorch.org/whl/cu126
Successfully installed torch-2.14.0+cu126

python         : 3.14.7
torch          : 2.14.0+cu126
compiled archs : ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
device         : NVIDIA GeForce GTX 1050 Ti
capability     : sm_61
cuda available : True
fp32 4096x4096 matmul:  72.9 ms  ->  1.89 TFLOPS
fp16 4096x4096 matmul:  89.6 ms  ->  1.53 TFLOPS
```

The GTX 1050 Ti runs PyTorch at about 90% of its 2.1 TFLOPS spec sheet, on
Python 3.14, today. Two beliefs produced the wrong answer:

1. *"Python 3.14 has no PyTorch CUDA wheels."* True of the **default** index
   (cu128), which is what a bare `pip install torch` uses. The **cu126** index
   publishes `cp314` wheels up to 2.14.0.
2. *"PyTorch dropped Pascal after 2.6.0."* The removal was for the CUDA 12.8 and
   12.9 builds. cu126 kept `sm_50` and `sm_60`, and `sm_61` not appearing in the
   arch list is irrelevant: CUDA cubins are binary-compatible *within* a major
   compute capability, so an `sm_60` binary runs on any `sm_6x` device. No PTX
   JIT is involved — there are no `compute_*` entries in that list.

   (fp16 being *slower* than fp32 is real, and is Pascal GP107's 1:64
   half-precision rate. sd.cpp reports the same thing as `fp16: 0`.)

So the real reasons are trade-offs, and you may weigh them differently:

* **Size.** The engine is a 46 MB download. torch plus its CUDA runtime is about
  10 GB installed — a third of the free space on the development machine.
* **No virtualenv.** This project's dependency policy, inherited from
  local-upscaler and soundboard, is system packages only. PyTorch cannot be that
  here, because Arch's `python-pytorch-cuda` is built against a CUDA too new for
  this card; it would have to be a venv.
* **Weight streaming from disk.** `--params-backend diffusion=disk` reads a
  segment, computes it, and drops it. diffusers offers CPU offload but nothing
  that streams from disk, and that is what makes `qwen-edit-2509-q2` (13.4 GB)
  and `flux2-dev-q4` (34.5 GB) attemptable at all on 11 GB of RAM.
* **It is measured.** The numbers in the README come from this engine actually
  running. A diffusers port would be a rewrite whose performance on this card is
  unknown until someone benchmarks it.

None of that makes PyTorch impossible here, and if you want the HuggingFace
ecosystem — LoRAs, ControlNets, the pipelines — it is a legitimate choice. It is
just not the one this app made.

## Hardware

| | Minimum | Comfortable | Development machine |
|---|---|---|---|
| GPU | anything with Vulkan 1.1 | 8 GB VRAM | GTX 1050 Ti, 4 GB |
| RAM | 8 GB | 16 GB | 12 GB |
| Disk | 6 GB | 40 GB | 33 GB free |

A GPU is not required — `--backend cpu` works and is very slow.

`--list-models` grades the catalog against whatever you have and says which
models fit, which offload, which stream from disk, and which need a bigger card.

### Two things about this class of machine

**Pascal reports `fp16: 0` under Vulkan.** There is no usable half-precision
path, so every activation is full precision and the working set is about twice
what the same model needs on a card one generation newer. `hardware.py` carries
this through to the grading.

**`--diffusion-fa` is deliberately not passed on Vulkan.** Flash attention would
be the obvious fix for the above, and ggml's implementation covers CPU, CUDA/ROCm
and Metal — not Vulkan. Passing it here would be ignored at best.

**Swap may not be memory.** On this machine all 12 GB of swap is `zram` —
compressed RAM. Quantised model weights do not compress, so a recipe that
"fits in RAM plus swap" would thrash. `hardware.py` never counts swap, and says
so when it matters.

### Building the CUDA engine — worth 4x on Pascal

Vulkan works everywhere and is the default. **CUDA is four times faster on this
card**, because ggml's Vulkan path falls back to scalar code on Pascal while its
CUDA path has hand-tuned quantised-matmul kernels. Measured, FLUX.2 Klein 4B at
512x512:

| | Vulkan | CUDA |
|---|---|---|
| text encode | 19.1 s | 10.0 s |
| sampling | 18.3 s/step | **4.4 s/step** |
| VAE decode | 9.7 s | 10.1 s |
| whole run, warm engine | ~95 s | **~32 s** |

4.4 s/step is essentially the arithmetic floor for a 4 B model at this size on a
card measured at 1.89 TFLOPS, so there is little left after this.

Upstream publishes CUDA binaries for **Windows only**, so on Linux this means
building. On a current Arch/CachyOS box that takes more than the obvious
incantation, and every step below exists because the obvious thing failed:

1. **Do not use the distribution's `cuda` package.** It is 13.x, and CUDA 13
   removed Pascal entirely — `nvcc -arch=sm_61` is not valid there. Use
   NVIDIA's 12.x redistributables, which need no root:

   ```fish
   set R https://developer.download.nvidia.com/compute/cuda/redist
   # cuda_nvcc 12.9, plus cudart / cccl / libcublas 12.6 for headers and libs
   ```

2. **Do not use GCC 16.** No CUDA 12.x nvcc can parse its libstdc++
   (`char8_t is undefined`, `0.0bf16` literal). `-allow-unsupported-compiler`
   skips the version *check*, not the incompatibility. Install `gcc14` and pass
   `-ccbin /usr/bin/g++-14`. clang does not help — it uses the same headers.

3. **Patch six declarations.** glibc 2.42 declares `cospi`, `cospif`, `sinpi`,
   `sinpif`, `rsqrt` and `rsqrtf` `noexcept`; CUDA 12.x declares them without,
   and nvcc rejects the pair. Append `__THROW` to those six lines in
   `crt/math_functions.h`. CUDA 13 fixed this upstream, which is precisely why
   the distribution ships 13.

4. **Symlink `lib64` to `lib`.** The redistributables use `lib/`; nvcc looks in
   `lib64/` and otherwise cannot find `-lcudart_static`.

5. Configure and build:

   ```fish
   cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=61 -DCUDAToolkit_ROOT=$CUDA \
     -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DGGML_NATIVE=OFF \
     -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-14 -allow-unsupported-compiler"
   ```

   14 minutes on four cores.

6. **Copy the CUDA runtime libraries next to the binaries.**
   `binary.child_env` puts only the executable's own directory on
   `LD_LIBRARY_PATH`, so `libcudart.so.12`, `libcublas.so.12` and
   `libcublasLt.so.12` have to sit beside `sd-cli`.

Then point the app at it — `engine_path` to that directory and `backend` to
`cuda` in `~/.config/local-edit/settings.json`, or **Advanced → Backend → Cuda**.
The app supplies `--vae-tiling --vae-tile-size 16x16` automatically on CUDA:
without it the VAE decode aborts with `ggml_cuda_pool_vmm::alloc` *after*
sampling has finished, which is the most expensive possible moment to fail.

None of this is something `--fetch-engine` can do for you, which is why the
Vulkan build remains the default and this stays a documented recipe.

## Formats

Whatever Pillow and Qt both read — PNG, JPEG (including `.jfif`), WebP, AVIF,
TIFF, BMP, GIF. The list is derived at runtime from what each library reports
rather than hardcoded; local-upscaler's hand-written list omitted `.jfif`, which
is not an obscure format but ordinary JPEG.

Output is PNG, with JPEG and WebP available in the save dialog.

## Where things go

| | |
|---|---|
| `~/.local/share/local-edit/engine/` | `sd-cli`, `sd-server` and their libraries |
| `~/.local/share/local-edit/models/` | weights — **configurable**, and worth configuring |
| `~/.cache/local-edit/work/` | scratch: staged references, chained results |
| `~/.config/local-edit/settings.json` | preferences and measured speeds |

The models directory is a setting because one recipe here is 34 GB. Point it at
another volume by setting `models_dir` in `settings.json`.

## Known limits

* **PhotoMaker runs one-shot, not on the warm server.** Its identity images are
  given as a *directory* that `sd-server` fixes at startup, so a face set cannot
  travel in a request. That recipe reloads its weights each run.
* **One model at a time.** sd.cpp has no runtime model-switch endpoint, so
  changing model restarts the engine process.
* **No inpainting mask yet.** The engine supports `--mask`; the UI does not
  expose it. See `docs/TODO.md`.
