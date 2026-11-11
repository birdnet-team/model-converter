<p align="center">
  <img src="birdnet-logo-circle.png" width="250" alt="BirdNET Logo">
</p>

<h1 align="center">BirdNET+ Model Converter</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="MODEL_LICENSE.md"><img src="https://img.shields.io/badge/model%20license-Apache--2.0-blue.svg" alt="Model license: Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg" alt="PyTorch 2.9+"></a>
  <img src="https://img.shields.io/badge/model-V3.0-green.svg" alt="BirdNET+ V3.0">
  <img src="https://img.shields.io/badge/species-11%2C709-yellow.svg" alt="11,709 animal species">
</p>

<p align="center">
  Convert BirdNET+ V3.0 PyTorch audio classification models into TensorFlow/LiteRT, TorchScript, ONNX, and Core ML formats.<br>
  Supports variable-length audio input, with post-conversion fixes for ONNX and TFLite.
</p>

## Overview

`convert_model.py` converts a BirdNET+ V3.0 PyTorch model (multi-head classifier on an EfficientNet backbone) into these formats:

| Format | Output (FP32) | FP16 | INT8 |
|---|---|---|---|
| TorchScript | `export/<model name>/torchscript/model.pt` | `model_fp16.pt` ¹ | — |
| ONNX | `export/<model name>/onnx/model.onnx` | `model_fp16.onnx` | `model_int8.onnx` ¹ |
| TFLite — preprocessing | `export/<model name>/tf/preprocessing/model.tflite` | — | — |
| TFLite — main (spectrogram in) | `export/<model name>/tf/main/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` |
| TFLite — combined (audio in) | `export/<model name>/tf/combined/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` ¹ |
| TF SavedModel — preprocessing | `export/<model name>/tf/preprocessing/` | — | — |
| TF SavedModel — main (spectrogram in) | `export/<model name>/tf/main/` ² | — | — |
| TF SavedModel — combined (audio in) | `export/<model name>/tf/combined/` ² | — | — |
| Core ML (audio in) | `export/<model name>/coreml/model.mlpackage` ³ | `model_fp16.mlpackage` | `model_int8.mlpackage` ¹ |

¹ Requires `--experimental-mode` (see [Precision Variants](#precision-variants)).  
² Intermediate model; does not support dynamic inputs.  
³ Fixed input shape; see [Core ML](#core-ml).

TorchScript and ONNX models accept any batch size; TFLite and Core ML models use the batch size set with `--batch-size`, and Core ML models also take a fixed audio length set with `--segment-duration`. The export directory also gets a label file (see [Class Pruning and Labels](#class-pruning-and-labels)) and plain-text copies of the model license and acceptable use policy (`MODEL_LICENSE.txt`, `ACCEPTABLE_USE.txt`).

Additional utilities:

- **`fix_conv2dsame.py`** — Replaces baked-in static Conv2dSame padding in ONNX exports with dynamic padding subgraphs.
- **`fix_tflite_dynamic_input.py`** — Patches TFLite flatbuffers to fix static batch/temporal dimensions after `litert_torch` conversion.
- **`test_models.py`** — Compares each converted format against the exported TorchScript model across batch sizes and segment durations.

## Setup

Clone the repository:

```bash
git clone https://github.com/birdnet-team/model-converter.git
cd model-converter
```

> **Git LFS required:** Model checkpoints and the BirdNET taxonomy are stored with [Git LFS](https://git-lfs.com/). Without it, they are small pointer files:
>
> ```bash
> sudo apt install git-lfs   # or: brew install git-lfs
> git lfs install
> git lfs pull
> ```

Create an environment and install the requirements, with conda:

```bash
conda create --name model-converter python=3.12
conda activate model-converter
pip install -r requirements.txt
```

or with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Python 3.11 also works. Python 3.10 is not supported because `onnxruntime-gpu` has no 3.10 wheels.

## Usage

Convert the default model (`02_BirdNET+_V3.0_Global_12K`) to all formats and all non-experimental precisions:

```bash
python convert_model.py
```

Common variations:

```bash
python convert_model.py -f onnx torchscript    # selected formats only
python convert_model.py -p fp32                # FP32 only
python convert_model.py -f onnx -p fp16        # ONNX FP16 only
python convert_model.py -f coreml              # Core ML only
python convert_model.py -o output/my_export    # write to output/my_export/<model name>/
python convert_model.py -m my_custom_model     # another model: a name in models/ or a path
python convert_model.py --list-models          # list the models in models/
```

### Class Pruning and Labels

A class pruning sheet is an Excel file listing the model's classes, with an `ix` column (the model's output index) and a `selected` column. Only rows with `selected=1` are exported. If the model directory has a `labels_pruning.xlsx`, it is used by default:

```bash
python convert_model.py                                     # uses labels_pruning.xlsx if present
python convert_model.py --class-pruning my_selection.xlsx   # use another sheet
python convert_model.py --no-class-pruning                  # export every class
```

For `02_BirdNET+_V3.0_Global_12K`, `labels_pruning.xlsx` keeps one class per BirdNET taxonomy id, drops classes with fewer than 10 training samples (`nParts`), and keeps 23 non-animal sound classes such as gun, chainsaw, engine and siren. That leaves 11,732 of 16,852 classes, 11,709 of them animal species. The sheet's `pruning_note` column says why each other class was dropped.

The export directory gets one label file, with one row per model output:

- **`labels.csv`** when the sheet has a `birdnet_id` column: `ix`, `birdnet_id`, `scientific_name`, `common_name`, `taxon_group`, `inat_id`, `ebird_code` and every `common_name_<lang>` translation, taken from the BirdNET taxonomy (`BirdNET_Taxonomy_v0.5-Nov2026.csv`; override with `--taxonomy`). The conversion stops before writing anything if a kept class has no `birdnet_id`, shares one with another kept class, or has one the taxonomy doesn't list.
- **`labels_pruned.xlsx`** otherwise: the selected rows, with `ix` renumbered to `0..n-1`.

Label files that a run doesn't write are removed from the export directory, so `--no-class-pruning` leaves none behind.

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
-f, --formats FMT        Output format(s): tf, tflite, torchscript, onnx, coreml (default: all)
-p, --precisions PREC    Precision variant(s): fp32, fp16, int8 (default: all)
--batch-size             Batch size baked into TFLite and Core ML models (default: 1)
--segment-duration       Audio segment duration in seconds for tracing; the fixed input length of Core ML models (default: 3.0)
--head-weights W W W     Weights for the three classifier heads (default: from model definition)
--class-pruning XLSX     Keep classes whose selected column is 1 (default: labels_pruning.xlsx in the model directory if present, else all classes)
--no-class-pruning       Export every class, ignoring labels_pruning.xlsx
--taxonomy CSV           BirdNET taxonomy for labels.csv (default: BirdNET_Taxonomy_v0.5-Nov2026.csv)
--experimental-mode      Enable experimental variants (see Precision Variants)
--list-models            List available models and exit
```

### Head Weights

The model has three classifier heads whose logits are combined as a weighted sum before the sigmoid:

```
predictions = sigmoid(w0 * head1_logits + w1 * head2_logits + w2 * head3_logits)
```

The default weights `[0.4, 0.35, 0.25]` come from the model definition. Override them with `--head-weights`:

```bash
python convert_model.py --head-weights 1.0 0.0 0.0   # Head 1 only
python convert_model.py --head-weights 0.5 0.5 0.0   # Heads 1 and 2 equally
```

Heads with weight `0.0` are replaced with 1×1 dummy layers, which shrinks the export. For the default model at FP32, all three heads take about 345 MB and Head 1 alone about 140 MB, which suits edge devices.

### Precision Variants

All non-experimental precisions are exported by default. Use `-p` to pick specific ones:

| Precision | Method | Notes |
|---|---|---|
| `fp32` | Standard export | Full precision, used as the reference |
| `fp16` | Weights converted to float16 (Core ML: float16 compute, see [Core ML](#core-ml)) | ~2× smaller, minimal accuracy loss |
| `int8` | Dynamic quantization (ONNX) / dynamic range quantization (TFLite) / weight-only quantization (Core ML) | ~4× smaller, significant accuracy loss |

Variants that are unstable or lose significant accuracy are only exported with `--experimental-mode`; otherwise they are skipped with a message:

| Variant | Why experimental |
|---|---|
| ONNX INT8 | Severe numerical differences; changes the top predicted species |
| TFLite INT8, combined model | Same quantization instability; not recommended for production |
| TorchScript FP16 | Needs BN folding and a stage-5 FP32 fallback; may produce NaN on untested models |
| Core ML INT8 | Treated like the other INT8 variants; weight-only, ~0.02 max prediction difference in conversion checks |

The INT8 version of the main TFLite model (weights-only quantization) is exported by default.

### Conversion Pipeline

The full pipeline (`-f tf tflite torchscript onnx`) runs these steps:

1. **TF Preprocessing** — Exports the mel-spectrogram preprocessing as a standalone TF SavedModel and TFLite model.
2. **TF Main Model** — Converts the classifier via `litert_torch`, re-wraps it with dynamic input signatures, and exports to TFLite.
3. **TF Combined** — Chains preprocessing and main model into a single end-to-end TFLite model (audio → predictions).
4. **TorchScript** — Exports the full model, preprocessing included, via `torch.jit.script`.
5. **ONNX** — Exports via `torch.onnx.export` with `dynamo=True`, then patches Resize and Conv2dSame ops for dynamic inputs.
6. **Core ML** — Traces the full model, preprocessing included, with `torch.jit.trace` and converts it with `coremltools` (see [Core ML](#core-ml)).

The separate preprocessing and main TFLite models let you plug in your own spectrogram pipeline or run the classifier on its own, for example computing spectrograms on a server and running only the main model on an edge device.

### Core ML

`-f coreml` writes Core ML ML Programs (`.mlpackage`) for iOS 16 / macOS 13 and later. Like the TorchScript and ONNX models, they take raw audio and run the preprocessing themselves:

- **Input** `input`: float32 `[batch, samples]`, mono 32 kHz audio. The shape is fixed at `--batch-size` × `--segment-duration` × 32000 (default `[1, 96000]`, 3 s), because the preprocessing computes the spectrogram width with Python integers that the trace bakes in. A fixed shape also lets Core ML run the model on the Neural Engine. Export again with other values for another shape.
- **Outputs** `predictions` (per-class sigmoid scores, in `labels.csv` order) and `embeddings`.
- **Precisions:** `model.mlpackage` computes in float32. `model_fp16.mlpackage` computes in float16, except the preprocessing (the power spectrum exceeds the float16 range) and the GeM pooling layers (the `x**p` and p-th root lose too much precision); both stay float32 and have no large weights. `model_int8.mlpackage` (`--experimental-mode` only) stores weights as per-channel int8 and computes in float32; the preprocessing weights stay float32.
- The model metadata records the sample rate, segment duration and head weights.

`coremltools` converts on any OS, but Core ML models only run on macOS (or iOS). For example, in Python on a Mac:

```python
import coremltools as ct
model = ct.models.MLModel('export/02_BirdNET+_V3.0_Global_12K/coreml/model.mlpackage')
scores = model.predict({'input': audio})['predictions']   # audio: float32 [1, 96000]
```

`convert_model.py` registers its own converters for three torch ops that `coremltools` 9.0 lacks or mishandles: `tensordot` (mel filterbank), `aminmax` (per-sample normalisation) and `int` (Conv2dSame padding, under NumPy 2.4).

### Testing Converted Models

Compare all converted formats against the TorchScript model:

```bash
python test_models.py
python test_models.py -a path/to/audio.wav
python test_models.py -e output/my_export/<model name>
python test_models.py --durations 3.0 5.0 --batch-size 2
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

Core ML models are tested only on macOS, and only for the batch size and duration they were exported with; other combinations are skipped.

> **Interpreting results:** Each format must match TorchScript within `atol=0.09` (`0.1` for INT8). ONNX, LiteRT and the FP16 variants should pass at the default durations. INT8 variants are unstable and not recommended for production. TFLite models only accept the batch size they were exported with.

## License

The source code is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Trained model weights are licensed under the [Apache License, Version 2.0](MODEL_LICENSE.md).

## Acceptable Use

BirdNET is built to support biodiversity research and conservation. See [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) for how we expect the BirdNET+ V3.0 models to be used, including uses we consider unacceptable and guidance on branding derivative work.

## Funding

Our work in the Cornell K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
