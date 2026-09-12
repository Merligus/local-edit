# Compatibility

## Hard dependencies

| | Why |
|---|---|
| Python 3.11+ | `X \| None` annotations, `match`-free but modern typing |
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

This is the central constraint of the project, not a stylistic preference.

The development machine has a **GTX 1050 Ti** — Pascal, compute capability
`sm_61`. Two independent blockers rule out diffusers, ComfyUI, and everything
else built on PyTorch:

1. **PyTorch dropped Pascal.** Support ended after 2.6.0; every build since
   targets `sm_70` and up. On this card `torch.cuda` is either an
   unsupported-architecture error or a silent fall back to the CPU.
2. **This system has only Python 3.14**, for which no PyTorch CUDA wheel exists.

Either one alone would be enough. Working around them would mean pinning
PyTorch 2.6 with CUDA 12.6 *and* installing a second Python — a stack frozen in
2025, on a machine whose Python moves with the distribution.

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) goes
through ggml and Vulkan instead. Vulkan does not care what CUDA has deprecated,
the binaries are 46 MB with no Python at all, and `sd-cli --list-devices` finds
this card and runs on it.

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
