<p align="center">
  <img src="birdnet-logo-circle.png" width="250" alt="BirdNET Logo">
</p>

<h1 align="center">BirdNET+ Model Converter</h1>

<p align="center">
  <a href="MODEL_LICENSE.md"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Model license: Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg" alt="PyTorch 2.9+"></a>
  <img src="https://img.shields.io/badge/model-V3.0-green.svg" alt="BirdNET+ V3.0">
  <img src="https://img.shields.io/badge/species-11%2C709-yellow.svg" alt="11,709 animal species">
</p>

<p align="center">
  Convert BirdNET V3.0 PyTorch audio classification models into TensorFlow/LiteRT, TorchScript, and ONNX formats.<br>
  Handles dynamic input lengths and batch sizes, including post-conversion fixes and TFLite static ops.
</p>

## Overview

This repository contains scripts to convert BirdNET V3.0 PyTorch models (EfficientNetV2-S backbone, multi-head architecture, ~11K species) into deployment-ready formats:

| Format | Output (FP32) | FP16 | INT8 |
|---|---|---|---|
| TorchScript | `export/<model name>/torchscript/model.pt` | `model_fp16.pt` ¹ | — |
| ONNX | `export/<model name>/onnx/model.onnx` | `model_fp16.onnx` | `model_int8.onnx` ¹ |
| TFLite — preprocessing | `export/<model name>/tf/preprocessing/model.tflite` | — | — |
| TFLite — main (spectrogram in) | `export/<model name>/tf/main/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` ¹ |
| TFLite — combined (audio in) | `export/<model name>/tf/combined/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` ¹ |
| TF SavedModel — preprocessing | `export/<model name>/tf/preprocessing/` | — | — |
| TF SavedModel — main (spectrogram in) | `export/<model name>/tf/main/` ² | — | — |
| TF SavedModel — combined (audio in) | `export/<model name>/tf/combined/` ² | — | — |


¹ Requires `--experimental-mode` (see [Experimental Mode](#experimental-mode)).  
² Intermediate TF model currently not supporting dynamic inputs.


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

Converted models are written to `export/<model name>/` by default. Use `-o` to change the output directory; the model's folder is created inside it:

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

Use only Head 1 for predictions (reduces model size by pruning unused heads):

```bash
python convert_model.py --head-weights 1.0 0.0 0.0
```

Keep only classes marked `selected=1` in an Excel label file (`ix` column). If the sheet has no `birdnet_id` column, a `labels_pruned.xlsx` with those rows and `ix` rewritten to `0..n-1` is written to the export directory. If the model directory has a `labels_pruning.xlsx`, it is applied by default; pass `--class-pruning` to use a different file:

```bash
python convert_model.py -m 03_BirdNET+_V3.0-preview4_Global_12K_EffNetB0 --class-pruning models/03_BirdNET+_V3.0-preview4_Global_12K_EffNetB0/labels_selection_01_wabad_europe.xlsx
```

For BirdNET+ V3.0 (`02_BirdNET+_V3.0_Global_12K`), `labels_pruning.xlsx` keeps one class per BirdNET taxonomy id, leaves out classes with fewer than 10 training samples (`nParts`), and keeps only 23 non-animal sound classes, such as gun, chainsaw, engine and siren. That leaves 11,732 of 16,852 classes: 11,709 animal species and the 23 sound classes. The `pruning_note` column says why each other class was dropped.

When the label sheet has a `birdnet_id` column, as the V3.0 model's does, the export directory gets a `labels.csv` instead, with one row per model output: `ix`, `birdnet_id`, `scientific_name`, `common_name`, `taxon_group`, `inat_id`, `ebird_code` and every `common_name_<lang>` translation, taken from the BirdNET taxonomy (`BirdNET_Taxonomy_v0.5-Nov2026.csv` in the repository root; pass `--taxonomy` to use another version). The conversion stops before writing anything if a kept class has no `birdnet_id`, shares one with another kept class, or has one the taxonomy doesn't list.

Export every class instead (no `labels.csv` or `labels_pruned.xlsx` is written, and copies left by an earlier run are removed):

```bash
python convert_model.py --no-class-pruning
```

Enable full INT8 quantization for ONNX and TFLite (experimental):

```bash
python convert_model.py -p int8 --experimental-mode
```
Use a different model, by its directory name in `models/` or by a path (the export goes to a folder named after the directory):

```bash
python convert_model.py -m my_custom_model
python convert_model.py -m /path/to/my_custom_model
```

List available models:

```bash
python convert_model.py --list-models
```

### All Options

```
usage: convert_model.py [-h] [-m MODEL_NAME] [-o OUTPUT_DIR] [-f FMT [FMT ...]]
                        [-p PREC [PREC ...]]
                        [--batch-size BATCH_SIZE]
                        [--segment-duration SEGMENT_DURATION]
                        [--head-weights W W W]
                        [--class-pruning XLSX | --no-class-pruning]
                        [--taxonomy CSV]
                        [--experimental-mode]
                        [--list-models]

-m, --model-name         Model directory: a name in models/ or a path (default: 02_BirdNET+_V3.0_Global_12K)
-o, --output-dir         Directory that receives one folder per model (default: export/)
-f, --formats FMT        Output format(s): tf, tflite, torchscript, onnx (default: all)
-p, --precisions PREC    Precision variant(s): fp32, fp16, int8 (default: all)
--batch-size             Batch size baked into TFLite models (default: 1)
--segment-duration       Min audio segment duration in seconds for tracing (default: 3.0)
--head-weights W W W     Weights for the three classifier heads (default: from model definition)
--class-pruning XLSX     Keep classes whose selected column is 1 (default: labels_pruning.xlsx in the model directory if present, else all classes)
--no-class-pruning       Export every class, ignoring labels_pruning.xlsx
--taxonomy CSV           BirdNET taxonomy for labels.csv (default: BirdNET_Taxonomy_v0.5-Nov2026.csv)
--experimental-mode      Enable experimental features like full INT8
--list-models            List available models and exit
```

### Head Weights and Pruning

The BirdNET V3.0 model uses a **multi-head architecture** with three classifier heads whose outputs are combined as a weighted sum of logits before applying sigmoid:

```
predictions = sigmoid(w0 * head1_logits + w1 * head2_logits + w2 * head3_logits)
```

The default weights are `[0.4, 0.35, 0.25]` as defined in the model definition. You can override them at export time with `--head-weights`:

```bash
# Use default weights [0.4, 0.35, 0.25]
python convert_model.py

# Use only Head 1
python convert_model.py --head-weights 1.0 0.0 0.0

# Use Heads 1 and 2 equally
python convert_model.py --head-weights 0.5 0.5 0.0

# Custom blend
python convert_model.py --head-weights 0.6 0.3 0.1
```


**Automatic pruning:** When a head weight is set to `0.0`, the corresponding head's layers (fully connected layers, attention blocks) are replaced with tiny 1×1 dummy layers after loading the checkpoint. This significantly reduces the exported model size:

| Configuration | Approximate model size |
|---|---|
| All heads `[0.4, 0.35, 0.25]` | ~520 MB |
| Head 1 only `[1.0, 0.0, 0.0]` | ~155 MB |

This is useful for deploying smaller models on edge devices when only a single head's predictions are needed.

### Experimental Mode

Some features are gated behind `--experimental-mode` because they are unstable or produce models with significant accuracy degradation:

```bash
python convert_model.py --experimental-mode
```

Features requiring experimental mode:

| Feature | Why experimental |
|---|---|
| **ONNX INT8** (dynamic quantization) | Severe numerical differences; changes top predicted species entirely |
| **TFLite INT8** (dynamic range quantization) | Same quantization instability; not recommended for production |
| **TorchScript FP16** (mixed-precision) | Requires careful BN folding and stage-5 fp32 fallback; may produce NaN on untested models |

Without `--experimental-mode`, these variants are skipped with an informational message:

```
⚠️  Skipping ONNX INT8 — this is only supported in experimental mode. Use --experimental-mode to enable.
```

**Default behavior (without `--experimental-mode`):**

| Precision | TF/TFLite | TorchScript | ONNX |
|---|---|---|---|
| FP32 | ✅ Exported | ✅ Exported | ✅ Exported |
| FP16 | ✅ Exported | ⚠️ Skipped | ✅ Exported |
| INT8 | ⚠️ Skipped | — | ⚠️ Skipped |

### Precision Variants

By default all *non-experimental* precision variants are exported. Use `-p` to select specific ones:

| Precision | Method | Notes |
|---|---|---|
| `fp32` | Standard export | Full precision, used as reference |
| `fp16` | Weight conversion to float16 | ~2× smaller, minimal accuracy loss |
| `int8` | Dynamic quantization (ONNX) / dynamic range quantization (TFLite) | **Experimental** — ~4× smaller but significant accuracy loss; requires `--experimental-mode` |

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
python test_models.py -e output/my_export/<model name>
```

Test with different segment durations or batch sizes:

```bash
python test_models.py --durations 3.0 5.0
python test_models.py --batch-size 2
```

```
usage: test_models.py [-h] [-e EXPORT_DIR] [-a AUDIO] [--batch-size N]
                      [--durations S [S ...]] [--gpu]

-e, --export-dir    Path to a model's export directory (default: export/<model name>/)
-a, --audio         Audio file for test input (default: example/soundscape.wav)
--batch-size        Batch size for test inputs (default: 1)
--durations         Segment durations in seconds (default: 3.0 5.0)
--gpu               Use GPU for TorchScript and ONNX inference
```

> **Interpreting results:** The test compares each format against TorchScript with `atol=0.09` (`0.1` for INT8).
> ONNX and LiteRT (combined) should pass at the default durations.
> **FP16** variants typically show slightly larger differences but should still pass.
> **INT8** variants are **unstable** — dynamic quantization introduces severe numerical
> differences that change the top predicted species entirely. INT8 is included as an
> experimental option and is **not recommended** for production use.

## License

The source code is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Trained model weights are licensed under the [Apache License, Version 2.0](MODEL_LICENSE.md).

## Acceptable Use

BirdNET is built to support biodiversity research and conservation. See [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) for how we expect the BirdNET V3.0 models to be used, including uses we consider unacceptable and guidance on branding derivative work.

## Funding

Our work in the Cornell K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)