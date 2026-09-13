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

### If your GPU is newer

The Vulkan build works on everything and stays a reasonable default. CUDA is
faster on NVIDIA, but **upstream publishes CUDA binaries for Windows only**, so
on Linux that means building:

```fish
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = Ada, 86 = Ampere
cmake --build build --config Release -j
```

Then point the app at it: **Advanced → Backend → Cuda**, and put the build's
`bin` directory in `engine_path` in
`~/.config/local-edit/settings.json`. A directory or a binary both work.

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
