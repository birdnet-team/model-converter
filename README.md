<p align="center">
  <img src="birdnet-logo-circle.png" width="250" alt="BirdNET Logo">
</p>

<h1 align="center">BirdNET+ Model Converter</h1>

<p align="center">
  <a href="https://github.com/birdnet-team/model-converter/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg" alt="PyTorch 2.9+"></a>
  <img src="https://img.shields.io/badge/model-V3.0--preview3-green.svg" alt="BirdNET+ V3.0-preview3">
  <img src="https://img.shields.io/badge/species-11%2C560-yellow.svg" alt="11,560 species">
</p>

<p align="center">
  Convert BirdNET V3.0 PyTorch audio classification models into TensorFlow/LiteRT, TorchScript, and ONNX formats.<br>
  Handles dynamic input lengths and batch sizes, including post-conversion fixes for Conv2dSame padding and TFLite static ops.
</p>

> **Status:** Experimental — under active development. Things will change and break.

## Overview

This repository contains scripts to convert BirdNET V3.0 PyTorch models (EfficientNetV2-S backbone, multi-head architecture, ~11K species) into deployment-ready formats:

| Format | Output (FP32) | FP16 | INT8 |
|---|---|---|---|
| TF SavedModel (preprocessing) | `export/tf/preprocessing/` | — | — |
| TF SavedModel (main) | `export/tf/main/` | — | — |
| TFLite — preprocessing | `export/tf/preprocessing/model.tflite` | — | — |
| TFLite — main (spectrogram in) | `export/tf/main/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` |
| TFLite — combined (audio in) | `export/tf/combined/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` |
| TorchScript | `export/torchscript/model.pt` | `model_fp16.pt` | — |
| ONNX | `export/onnx/model.onnx` | `model_fp16.onnx` | `model_int8.onnx` |

Additional utilities:

- **`fix_conv2dsame.py`** — Replaces baked-in static Conv2dSame padding in ONNX exports with dynamic padding subgraphs.
- **`fix_tflite_dynamic_input.py`** — Patches TFLite flatbuffers to fix static batch/temporal dimensions after `litert_torch` conversion.
- **`test_models.py`** — Compares converted model outputs (ONNX, LiteRT, TorchScript) against the PyTorch reference across different batch sizes and segment durations.

## Setup

Download or clone the repository:

```bash
git clone https://github.com/birdnet-team/model-converter.git
cd model-converter
```

> **Git LFS required:** Model checkpoints are stored with [Git LFS](https://git-lfs.com/). If `git lfs` is not installed, checkpoint files will be small pointer files instead of the actual weights. Install Git LFS and pull:
>
> ```bash
> sudo apt install git-lfs   # or: brew install git-lfs
> git lfs install
> git lfs pull
> ```

Create a virtual environment (choose one):

**Option A — conda:**

```bash
conda create --name model-converter python=3.12
conda activate model-converter
pip install -r requirements.txt
```

**Option B — venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Python 3.10 and 3.11 are also supported, but 3.12 is recommended.

## Usage

Convert the PyTorch model to all supported formats and precisions:

```bash
python convert_model.py
```

Converted models are written to `export/` by default. Use `-o` to change the output directory:

```bash
python convert_model.py -o output/my_export
```

Convert to specific formats only:

```bash
python convert_model.py -f onnx torchscript
python convert_model.py -f tf tflite
```

Export FP32 only (skip FP16 and INT8):

```bash
python convert_model.py -p fp32
```

Export only ONNX in FP16:

```bash
python convert_model.py -f onnx -p fp16
```

Use a custom audio file for INT8 calibration (TFLite):

```bash
python convert_model.py -p int8 --calibration-audio path/to/audio.wav
```

Use a different model directory:

```bash
python convert_model.py -m models/my_custom_model
```

List available models:

```bash
python convert_model.py --list-models
```

### All Options

```
usage: convert_model.py [-h] [-m MODEL_DIR] [-o OUTPUT_DIR] [-f FMT [FMT ...]]
                        [-p PREC [PREC ...]] [--calibration-audio FILE]
                        [--batch-size BATCH_SIZE]
                        [--segment-duration SEGMENT_DURATION]
                        [--list-models]

  -m, --model-dir          Path to model directory (default: models/01_BirdNET+_...)
  -o, --output-dir         Directory for converted model output (default: export/)
  -f, --formats FMT        Output format(s): tf, tflite, torchscript, onnx (default: all)
  -p, --precisions PREC    Precision variant(s): fp32, fp16, int8 (default: all)
      --calibration-audio  Audio file for INT8 calibration (default: example/soundscape.wav)
      --batch-size          Batch size baked into TFLite models (default: 1)
      --segment-duration    Min audio segment duration in seconds for tracing (default: 3.0)
      --list-models         List available models and exit
```

### Precision Variants

By default all precision variants are exported. Use `-p` to select specific ones:

| Precision | Method | Notes |
|---|---|---|
| `fp32` | Standard export | Full precision, used as reference |
| `fp16` | Weight conversion to float16 | ~2× smaller, minimal accuracy loss |
| `int8` | Dynamic quantization (ONNX, TorchScript) / representative-dataset quantization (TFLite) | **Currently unstable** — ~4× smaller but significant accuracy loss; not recommended at the moment |

For TFLite INT8, calibration data is generated from an audio file (default: `example/soundscape.wav`).
Pass `--calibration-audio` to use a different file.

### Conversion Pipeline

The full pipeline (`-f tf tflite torchscript onnx`) runs these steps:

1. **TF Preprocessing** — Exports the mel-spectrogram preprocessing as a standalone TF SavedModel and TFLite model.
2. **TF Main Model** — Converts the classifier via `litert_torch`, re-wraps with dynamic input signatures, and exports to TFLite.
3. **TF Combined** — Chains preprocessing + main model into a single end-to-end TFLite model (audio → predictions).
4. **TorchScript** — Exports the full model (with preprocessing) via `torch.jit.script`.
5. **ONNX** — Exports via `torch.onnx.export` with `dynamo=True`, then patches Resize and Conv2dSame ops for dynamic inputs.

The TFLite preprocessing and main models are exported **separately** in addition to the combined model.
This lets users plug in their own spectrogram pipeline or run the classifier independently —
for example, pre-computing spectrograms on a server and sending only the images to an edge device running the main model.

### Testing Converted Models

Compare outputs of all converted formats against the TorchScript reference:

```bash
python test_models.py
```

Use a custom audio file or export directory:

```bash
python test_models.py -a path/to/audio.wav
python test_models.py -e output/my_export
```

Test with different segment durations or batch sizes:

```bash
python test_models.py --durations 3.0 5.0
python test_models.py --batch-size 2
```

```
usage: test_models.py [-h] [-e EXPORT_DIR] [-a AUDIO] [--batch-size N]
                      [--durations S [S ...]] [--gpu]

  -e, --export-dir    Path to export directory (default: export/)
  -a, --audio         Audio file for test input (default: example/soundscape.wav)
      --batch-size    Batch size for test inputs (default: 1)
      --durations     Segment durations in seconds (default: 3.0)
      --gpu           Use GPU for TorchScript and ONNX inference
```

> **Interpreting results:** The test compares each format against TorchScript using a tight tolerance (`atol=0.025`).
> ONNX and LiteRT (combined) should pass at the default 3.0s duration.
> **FP16** variants typically show slightly larger differences but should still pass.
> **INT8** variants are **unstable** — dynamic quantization introduces severe numerical
> differences that change the top predicted species entirely. INT8 is included as an
> experimental option and is **not recommended** for production use.

## License

The source code is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Trained model weights are licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). See [TERMS_OF_USE.md](TERMS_OF_USE.md) for full terms.

## Funding

Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)