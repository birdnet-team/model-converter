<p align="center">
  <img src="birdnet-logo-circle.png" width="250" alt="BirdNET Logo">
</p>

<h1 align="center">BirdNET Model Converter</h1>

<p align="center">
  <a href="https://github.com/birdnet-team/model-converter/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  Convert BirdNET V3.0 PyTorch audio classification models into TensorFlow/LiteRT, TorchScript, and ONNX formats.<br>
  Handles dynamic input lengths and batch sizes, including post-conversion fixes for Conv2dSame padding and TFLite static ops.
</p>

> **Status:** Experimental — under active development. Things will change and break.

## Overview

This repository contains scripts to convert BirdNET V3.0 PyTorch models (EfficientNetV2-S backbone, multi-head architecture, ~11K species) into deployment-ready formats:

| Format | Script | Output |
|---|---|---|
| TensorFlow SavedModel | `convert_model.py` | `models/.../tf/` |
| TFLite / LiteRT | `convert_model.py` | `models/.../tf/**/*.tflite` |
| TorchScript | `convert_model.py` | `models/.../torchscript/model.pt` |
| ONNX | `convert_model.py` | `models/.../onnx/model.onnx` |

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

## Usage

Convert the PyTorch model to all supported formats:

```bash
python convert_model.py
```

This runs the full conversion pipeline:

1. **TF Preprocessing** — Exports the mel-spectrogram preprocessing as a standalone TF SavedModel and TFLite model.
2. **TF Main Model** — Converts the classifier via `litert_torch`, re-wraps with dynamic input signatures, and exports to TFLite.
3. **TF Combined** — Chains preprocessing + main model into a single end-to-end TFLite model.
4. **TorchScript** — Exports the full model (with preprocessing) via `torch.jit.script`.
5. **ONNX** — Exports via `torch.onnx.export` with `dynamo=True`, then patches Resize and Conv2dSame ops for dynamic inputs.

### Testing Converted Models

Compare outputs of all converted formats against the TorchScript reference:

```bash
python test_models.py
```

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