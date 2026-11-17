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
  ONNX and TorchScript models take any batch size and audio length; the other formats use the export shape.<br>
  The <a href="#birdnet-geomodel">BirdNET+ Geomodel</a>, which predicts where and when species occur, is converted here too.
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#precision-variants">Precision</a> ·
  <a href="#core-ml">Core ML</a> ·
  <a href="#testing-converted-models">Testing</a> ·
  <a href="#birdnet-geomodel">Geomodel</a> ·
  <a href="#license">License</a>
</p>

## Overview

`convert_model.py` converts a BirdNET+ V3.0 PyTorch model (multi-head classifier on an EfficientNet backbone) into these formats:

| Format | Output (FP32) | FP16 | INT8 |
|---|---|---|---|
| TorchScript | `export/<model name>/torchscript/model.pt` | `model_fp16.pt` ¹ | — |
| ONNX | `export/<model name>/onnx/model.onnx` | `model_fp16.onnx`, `model_fp16_gpu.onnx` ⁴ | `model_int8.onnx` ¹ |
| TFLite — preprocessing | `export/<model name>/tf/preprocessing/model.tflite` | — | — |
| TFLite — main (spectrogram in) | `export/<model name>/tf/main/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` ¹ |
| TFLite — combined (audio in) | `export/<model name>/tf/combined/model.tflite` | `model_fp16.tflite` | `model_int8.tflite` ¹ |
| TF SavedModel — preprocessing | `export/<model name>/tf/preprocessing/` | — | — |
| TF SavedModel — main (spectrogram in) | `export/<model name>/tf/main/` ² | — | — |
| TF SavedModel — combined (audio in) | `export/<model name>/tf/combined/` ² | — | — |
| Core ML (audio in) | `export/<model name>/coreml/model.mlpackage` ³ | `model_fp16.mlpackage` | `model_int8.mlpackage` ¹ |

¹ Requires `--experimental-mode` (see [Precision Variants](#precision-variants)).  
² Fixed input shape: the model takes the batch size set with `--batch-size` and the length set with `--segment-duration` only, and fails with an error on any other shape. Use the ONNX or TorchScript models for other batch sizes or lengths.  
³ Fixed input shape; see [Core ML](#core-ml).  
⁴ `model_fp16.onnx` is the default FP16 model (mixed precision); `model_fp16_gpu.onnx` is faster on GPUs. See [Precision Variants](#precision-variants).

Input shapes each format supports:

| Format | Batch size | Audio length |
|---|---|---|
| TorchScript, ONNX | 1 or more | 1025 samples or more (the spectrogram's reflect padding needs more than half an FFT window); outputs match the PyTorch model at every length tested, 2 s to 10 s |
| TFLite | fixed (`--batch-size`) | accurate at `--segment-duration` only ⁵ |
| TF SavedModel (main, combined), Core ML | fixed (`--batch-size`) | fixed (`--segment-duration`) |

⁵ The TFLite files are patched to accept other lengths, but this is experimental: at lengths other than the export length the scores deviate from the PyTorch model (max difference 0.05–0.1 at 2 s and 5 s for a 3 s export, against below 0.002 at 3 s). For another length, export again with that `--segment-duration`.

The export directory also gets a label file (see [Class Pruning and Labels](#class-pruning-and-labels)) and plain-text copies of the model license and acceptable use policy (`MODEL_LICENSE.txt`, `ACCEPTABLE_USE.txt`).

Additional utilities:

- **`fix_conv2dsame.py`** — Replaces baked-in static Conv2dSame padding in ONNX exports with dynamic padding subgraphs.
- **`fix_tflite_dynamic_input.py`** — Patches TFLite flatbuffers to fix static batch/temporal dimensions after `litert_torch` conversion.
- **`test_models.py`** — Compares each converted format against the exported TorchScript model across batch sizes and segment durations.
- **`convert_geomodel.py`**, **`test_geomodel.py`** — Convert and test the [BirdNET+ Geomodel](#birdnet-geomodel).

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

For `02_BirdNET+_V3.0_Global_12K`, `labels_pruning.xlsx` keeps one class per BirdNET taxonomy id, drops animal classes with fewer than 10 training samples (`nParts`), and keeps 16 non-animal sound classes: gun, chainsaw, siren, fireworks, hammer, power tool, sawing, engine, aircraft, vehicle horn, boat, motor vehicle, rail transport, boom, music and fire. That leaves 11,725 of 16,852 classes, 11,709 of them animal species. The sheet's `pruning_note` column says why each other class was dropped.

The export directory gets one label file, with one row per model output:

- **`labels.csv`** when the sheet has a `birdnet_id` column: `ix`, `birdnet_id`, `scientific_name`, `common_name`, `taxon_group`, `inat_id`, `ebird_code` and every `common_name_<lang>` translation, taken from the BirdNET taxonomy (`BirdNET_Taxonomy_v0.5-Nov2026.csv`; override with `--taxonomy`). The conversion stops before writing anything if a kept class has no `birdnet_id`, shares one with another kept class, or has one the taxonomy doesn't list.
- **`labels_pruned.xlsx`** for a pruning sheet without a `birdnet_id` column: the selected rows, with `ix` renumbered to `0..n-1`.
- **`labels.xlsx`** when classes are not pruned (`--no-class-pruning`, or no `labels_pruning.xlsx` in the model directory): the model directory's own `labels.xlsx`, copied unchanged. Its `ix` column is the output index. The conversion stops if its row count differs from the model's class count.

Label files that a run doesn't write are removed from the export directory, so it never holds labels for a different set of classes.

If the model directory has a `label_overrides.csv` (or with `--label-overrides CSV`; `none` disables it), its non-empty cells replace the taxonomy names in `labels.csv`. Rows are keyed by the original output `ix` and its `birdnet_id`. For `02_BirdNET+_V3.0_Global_12K` it names the 16 sound classes in all languages.

### All Options

```
usage: convert_model.py [-h] [-m MODEL_NAME] [-o OUTPUT_DIR] [-f FMT [FMT ...]]
                        [-p PREC [PREC ...]]
                        [--batch-size BATCH_SIZE]
                        [--segment-duration SEGMENT_DURATION]
                        [--head-weights W W W]
                        [--class-pruning XLSX | --no-class-pruning]
                        [--taxonomy CSV]
                        [--label-overrides CSV]
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
--label-overrides CSV    Names that replace taxonomy names in labels.csv (default: label_overrides.csv in the model directory if present; none to disable)
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
| `fp16` | Weights converted to float16 (Core ML: float16 compute, see [Core ML](#core-ml)) | ~2× smaller; small score differences (table below) |
| `int8` | Dynamic quantization (ONNX) / dynamic range quantization (TFLite) / weight-only quantization (Core ML) | ~4× smaller, significant accuracy loss |

Variants that are unstable or lose significant accuracy are only exported with `--experimental-mode`; otherwise they are skipped with a message:

| Variant | Why experimental |
|---|---|
| ONNX INT8 | Score differences above 0.1; changes the predicted species |
| TFLite INT8, combined model | Score differences up to 0.8; changes the top predicted species |
| TFLite INT8, main model | Weights-only quantization; score differences up to 0.1 |
| TorchScript FP16 | Needs BN folding and a stage-5 FP32 fallback; may produce NaN on untested models |
| Core ML INT8 | Treated like the other INT8 variants; weight-only, ~0.02 max prediction difference on the first 3 s of the example recording, up to ~0.06 over all 40 windows |

No INT8 variant is exported by default. An accurate INT8 model would need quantization-aware training, which this converter does not do.

Largest score difference to the FP32 PyTorch model for the default model with all three heads at the export length of 3 s. "Real audio" is the 40 consecutive 3 s windows of `example/soundscape.wav`; "synthetic" is 18 edge-case inputs (silence, DC, a 1e-7 sine, the soundscape scaled by 1e-4 to 10, white noise from 1e-5 to 30 000, clipped noise, sines, a sweep, soundscape plus a loud tone). The ONNX files return finite scores on all of them.

| Variant | Real audio | Synthetic | Top-1 on real audio |
|---|---|---|---|
| ONNX FP32 | below 0.00001 | below 0.00001 | 40/40 |
| TFLite FP32 ⁶ | below 0.00001 | — | same |
| ONNX FP16 (`model_fp16.onnx`), CPU / CUDA | 0.003 / 0.006 | 0.016 / 0.010 | 40/40 |
| ONNX FP16 for GPUs (`model_fp16_gpu.onnx`), CPU / CUDA | 0.003 / 0.008 | 0.016 / 0.015 | 40/40 |
| TFLite FP16 ⁶ | 0.001 | — | same |
| TorchScript FP16 ¹ ⁶ | 0.003 | — | same |
| TFLite INT8, main model ¹ ⁶ | 0.10 | — | same order, shifted scores |
| ONNX INT8 ¹ ⁶ | 0.11 | — | changed |
| TFLite INT8, combined model ¹ ⁶ | 0.83 | — | changed |

⁶ Measured with `test_models.py` on the first 3 s of the soundscape and on silence only.

What these numbers do and do not cover:

- With a single head (`--head-weights 1 0 0`) and pruned classes, the two ONNX FP16 files differ from FP32 by up to 0.007 (`model_fp16.onnx`) and 0.010 (`model_fp16_gpu.onnx`) on the real audio, with the same top-1 on all 40 windows. On white noise, where every score is below 0.03, single scores move by up to 0.04.
- The remaining FP16 error comes from rounding the backbone weights to float16. Inputs the model was not trained on (noise, clipping) are the most sensitive to it.
- ONNX Runtime on x86 CPUs computes float16 graphs in float32, so a CPU test on x86 only measures the weight rounding. The CUDA columns are real float16 arithmetic. The models were not run on arm64 or on a phone.
- Core ML models cannot be run on Linux and are not in this table; see [Core ML](#core-ml).

ONNX has two FP16 files of the same size. `model_fp16.onnx` is the default: it runs the backbone convolutions in float16 and everything else in float32 (mixed precision), with the classifier weight matrices stored as float16 and cast to float32 when the model loads. The squeeze-excite gate convolutions stay in float32, because their output can exceed the float16 range on noise-like input. This layout targets CPUs, arm64 in particular, where ONNX Runtime has fast float16 convolutions but slow float16 elementwise ops. `model_fp16_gpu.onnx` runs the whole backbone in float16 and is faster on GPUs (about 28% on an NVIDIA TITAN RTX at batch 32). Both keep the preprocessing in float32 with float32 constants and have the same input and output names as the FP32 model.

The ONNX and TorchScript models compute the spectrogram with matrix products instead of an STFT operator, so its heavy operations run on GPU execution providers (a few index and shape operations stay on the CPU), and the GeM pooling layers use a formulation that stays inside the float16 range.

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

- **Input** `input`: float32 `[batch, samples]`, mono 32 kHz audio. The shape is fixed at `--batch-size` × `--segment-duration` × 32000 (default `[1, 96000]`, 3 s), because the preprocessing computes the spectrogram width with Python integers that the trace bakes in. A fixed shape also lets Core ML use the Neural Engine for the FP16 model. Export again with other values for another shape.
- **Outputs** `predictions` (per-class sigmoid scores, in `labels.csv` order) and `embeddings`.
- **Precisions:** `model.mlpackage` computes in float32. `model_fp16.mlpackage` computes in float16, except the preprocessing (the power spectrum exceeds the float16 range) and the GeM pooling layers (the `x**p` and p-th root lose too much precision); both stay float32 and have no large weights. `model_int8.mlpackage` (`--experimental-mode` only) stores weights as per-channel int8 and computes in float32; the preprocessing weights stay float32.
- **Compute units:** only `model_fp16.mlpackage` can use the Neural Engine, which computes in float16: Core ML places most of its operations there and the float32 parts on the CPU or GPU. The FP32 and INT8 models run on the GPU or CPU only. How close the FP16 model's scores are to the PyTorch model's depends on where it runs: over the 40 windows of the example recording, within about 0.01 on the GPU, 0.02 on the CPU and 0.04 on the Neural Engine, where the top species changed in 1 of 40 windows (a near tie).
- **Silence:** for all-zero input the FP16 model's embeddings drift from the PyTorch model's on the CPU and Neural Engine (cosine similarity about 0.97; for real audio it is about 0.998, and 0.98 in the worst of the 40 windows on the Neural Engine). With the default head weights and class pruning the scores stay within 0.01. With `--head-weights 1 0 0 --no-class-pruning` one class's score differs by about 0.25, and `test_models.py` reports a failure for the FP16 model on silence. Use the FP32 model if you need exact scores or embeddings for silent segments.
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
                      [--durations S [S ...]] [--gpu] [--speed-runs N]

-e, --export-dir    Path to a model's export directory (default: export/<model name>/)
-a, --audio         Audio file for test input (default: example/soundscape.wav)
--batch-size        Batch size for test inputs (default: 1)
--durations         Segment durations in seconds (default: 3.0 5.0)
--gpu               Use GPU for TorchScript and ONNX inference
--speed-runs        Timed runs per model in the speed test (default: 10; 0 skips it)
```

Core ML models are tested only on macOS, and only for the batch size and duration they were exported with; other combinations are skipped.

After the comparison, a speed test times every model on the same input (the first duration and the batch size given): two warm-up runs, then the median of `--speed-runs` runs, reported as milliseconds per batch, seconds of audio per second, and speed relative to TorchScript FP32. The table also shows where each model ran; with `--gpu`, an ONNX model that falls back to the CPU because the CUDA libraries cannot be loaded is flagged. The speed test does not affect the pass/fail result, and the numbers only describe the machine it runs on: a desktop CPU says little about a phone.

Measured on one machine that was also running other jobs (x86 CPU, NVIDIA TITAN RTX; default model, all three heads, 3 s segments), in seconds of audio per second; treat the CPU column as approximate:

| Variant | CPU, batch 1 | GPU, batch 32 |
|---|---|---|
| TorchScript FP32 | 21 | 734 |
| TorchScript FP16 ¹ | 1 | 1069 |
| ONNX FP32 | 25 | 662 |
| ONNX FP16 (`model_fp16.onnx`) | 24 | 749 |
| ONNX FP16 for GPUs (`model_fp16_gpu.onnx`) | 24 | 959 |
| ONNX INT8 ¹ | 6 | — |
| TFLite FP32 | 48 | — |
| TFLite FP16 | 45 | — |

¹ `--experimental-mode` only. TorchScript FP16 is a GPU format: on a CPU it is far slower than FP32. ONNX INT8 is much slower than FP32 on an x86 CPU. All models use every CPU core. The ONNX GPU timings include copying the input to the GPU and the output back; the TorchScript timings do not.

> **Interpreting results:** Each format must match TorchScript within `atol=0.09` (`0.1` for INT8). ONNX, LiteRT and the FP16 variants should pass at the default durations* and on an all-zero input, which is tested as well. INT8 variants are unstable and not recommended for production. TFLite models only accept the batch size they were exported with.
>
> \* TFLite models match TorchScript almost exactly on 3 s input (the default `--segment-duration`) but drift on other lengths: about 0.04 max difference at 5 s in FP32, and up to 0.12 in FP16 with a single head (`--head-weights 1.0 0.0 0.0`), which fails the check. Use 3 s segments with TFLite FP16.

### Continuous Integration

[`.github/workflows/conversion-tests.yml`](.github/workflows/conversion-tests.yml) converts the default model to every format and runs `test_models.py` on the result: conversion and the TorchScript, ONNX and LiteRT tests on Linux with CPU-only PyTorch, and Core ML models, when the converter produces them, on a macOS runner. The same jobs convert the geomodel with `convert_geomodel.py` and run `test_geomodel.py`, including its Core ML models on macOS.

- **When it runs:** on every pull request and push to `main`, but it only converts when the change touches the converter (`convert_model.py`, `convert_geomodel.py`, the fix scripts, the test scripts, `requirements.txt`, the taxonomy or the directories of the default model and the geomodel; see `CONVERTER_FILES`). It can also be started by hand from the Actions tab.
- **Required check:** the `Conversion tests` job always reports a result, passing when the tests passed or had nothing to test, so it can be a required status check for `main` (*Settings → Rules* or *Branches*). Require that job, not the individual ones, which are skipped for unrelated changes.
- **Git LFS bandwidth:** the checkpoints (~490 MB with the taxonomy) are downloaded once into the Actions cache and restored from there. Pull requests read the cache saved on `main`, and a weekly run keeps it from expiring, so LFS bandwidth is only used when the checkpoint or taxonomy changes.

## BirdNET+ Geomodel

The [BirdNET+ Geomodel](https://github.com/birdnet-team/geomodel) predicts which species occur at a place in a given week of the year, typically to filter the audio model's detections by location and season. `convert_geomodel.py` converts a geomodel checkpoint; the default is `models/03_BirdNET+_Geomodel_V3.0.5_Global_14K`. V3.0.5 is a patch of the geomodel V3.0.4 release: the same trained weights, with the classes reduced to those in the BirdNET taxonomy and labelled with BirdNET IDs (see [Geomodel labels](#geomodel-labels)), and converted files that are closer to the PyTorch model.

```bash
python convert_geomodel.py                     # default geomodel, all formats, FP32 and FP16
python convert_geomodel.py -f onnx -p fp16     # ONNX FP16 only
python convert_geomodel.py -f coreml           # Core ML only
python convert_geomodel.py --batch-size 32     # batch size of the LiteRT, SavedModel and Core ML models
python test_geomodel.py                        # compare every converted file with TorchScript
```

Every model takes one float32 tensor `[batch, 3]` with the columns latitude (degrees, -90 to 90), longitude (degrees, -180 to 180) and week (1 to 48, four weeks per month), and returns one value per class, in label order:

| Format | Output in `export/<model name>/` | Returns | Batch size |
|---|---|---|---|
| PyTorch checkpoint | `checkpoint/model.pt`, `model_labels.txt` | — | — |
| TorchScript | `torchscript/model.pt` | raw logits ¹ | any |
| ONNX | `onnx/model.onnx`, `model_fp16.onnx` | probabilities | any |
| LiteRT (TFLite) | `tf/model.tflite`, `model_fp16.tflite`, `model_int8.tflite` ² | probabilities | fixed (`--batch-size`) ³ |
| TF SavedModel | `tf/` | probabilities | fixed (`--batch-size`) ³ |
| Core ML | `coreml/model.mlpackage`, `model_fp16.mlpackage` | probabilities | fixed (`--batch-size`) ⁴ |

¹ As in the geomodel releases, so that existing loaders keep working; apply a sigmoid, or export with `--torchscript-sigmoid` to get probabilities.  
² Experimental: exported with `--experimental-mode` only, and not part of the released files. It changes the top class for about one input in ten.  
³ `litert_torch` converts a fixed input shape. The SavedModel fails with an error on another batch size; a LiteRT model resized to another batch size returns output for the export batch size only. Use ONNX or TorchScript for other batch sizes.  
⁴ ML Programs for iOS 16 / macOS 13 and later, input `input`, output `probabilities`. See [Geomodel Core ML](#geomodel-core-ml); these models have not been run yet.

The checkpoint in `checkpoint/` holds the exported classes and nothing but weights, the model configuration and the species vocabulary (no optimizer state). It loads with `load_checkpoint` in the model directory's `model.py` and with the geomodel repository's `convert.py` and `predict.py`, which read the labels from `model_labels.txt` next to it.

Each run first deletes the model files an earlier run left in the export directory, so the models always match the label files written by that run.

Differences between these files and the files of the geomodel V3.0.4 release:

- **FP16 is more accurate.** The FP16 models store their weights as float16 and compute in float32. The model definition here also converts degrees and weeks to radians with constants that float16 represents without loss; rounding `pi / 180` to float16 was the main source of error in FP16 files.
- **LiteRT models need no Flex delegate.** They contain built-in operators only, so they run under plain LiteRT; the release's TFLite files need the TensorFlow Select ops (`FlexErf`).
- **Fixed batch size** for LiteRT and the SavedModel, as described above.
- **Core ML** is new.
- **No INT8 file** by default.

Largest difference in output probability to the original PyTorch model of the geomodel repository, over 3000 random locations and weeks:

| Model | Max difference | Top class agrees | Size |
|---|---|---|---|
| TorchScript, ONNX, LiteRT FP32, TF SavedModel | below 0.00004 | 100% | 15 MB |
| ONNX FP16 | 0.005 (V3.0.4 release file: 0.09) | 99.9% | 7 MB |
| LiteRT FP16 | 0.004 (V3.0.4 release file: 0.06) | 99.8% | 7 MB |
| LiteRT INT8 ² | 0.19 | 89% | 4 MB |

`convert_geomodel.py` compares every file it writes with the PyTorch model on 256 fixed locations and weeks and fails if one exceeds its tolerance (0.0001 for FP32, 0.01 for FP16, 0.3 for INT8). The FP32 models differ from the V3.0.4 release files by less than 0.0001 on the classes both have; that difference comes from the changed degree-to-radian conversion. The INT8 model changes the top class for about one input in ten, which is why it is experimental. Measured on an x86 CPU; LiteRT FP16 on a device that computes in float16 was not measured.

### Geomodel Core ML

`-f coreml` writes two ML Programs with a fixed input shape of `[--batch-size, 3]`:

- `model.mlpackage` computes in float32.
- `model_fp16.mlpackage` computes in float16, which the Neural Engine requires, except for the coordinate encoding: splitting the input, converting degrees and weeks to radians, and sin and cos stay float32. Latitude and longitude need more digits than float16 has. Input and output are float32.

`coremltools` converts on any OS, but Core ML models only run on macOS and iOS. **They have been converted and their graphs inspected on Linux, but not run.** A float16 simulation of the same split in PyTorch gives a largest difference of 0.01 to the FP32 model (0.17 with the encoding in float16 too); the real figure depends on the compute unit. On a Mac, `convert_geomodel.py` compares them with the PyTorch model like the other formats, and `test_geomodel.py` tests them (tolerance 0.001 for FP32, 0.03 for FP16); the macOS CI job runs that test.

### Geomodel labels

The export directory gets `labels.csv`, with the same columns as the audio model's `labels.csv` (BirdNET ID, names, translations), and `labels.txt` in the geomodel's own format (`id<TAB>scientific name<TAB>common name`) with the BirdNET ID as the id. Both have one row per model output. The audio model and the geomodel cover different sets of classes; rows are matched by `birdnet_id`.

Which outputs are exported, and their BirdNET IDs, come from a class pruning sheet as for the audio model (`labels_pruning.xlsx` in the model directory, or `--class-pruning`): rows with `selected = 1` are kept, `ix` is the output of the checkpoint, and `birdnet_id` links the row to the taxonomy. The sheet belongs to one checkpoint: the converter compares the sheet's `source_code` or `birdnet_id` with the checkpoint's species vocabulary and stops if they name different classes.

The default model's checkpoint is already reduced to its 14,057 selected classes.
`labels_pruning_source.xlsx` in the model directory lists all 14,082 classes of the release with their match and what was dropped; with it and the release checkpoint, `--checkpoint` and `--class-pruning` reproduce `checkpoint.pt`.

### Converting your own geomodel

A checkpoint trained with the [geomodel repository](https://github.com/birdnet-team/geomodel) comes with a `labels.txt` of eBird codes and iNaturalist ids. Match it to the taxonomy, review the sheet, then convert:

```bash
python convert_geomodel.py --map-labels my_run/checkpoint_best_labels.txt --sheet-out my_run/pruning.xlsx
python convert_geomodel.py --checkpoint my_run/checkpoint_best.pt --class-pruning my_run/pruning.xlsx -o export/my_run
```

`--map-labels` matches each label by scientific name, then by code, then by a scientific name alias, and selects one class per BirdNET ID; it prints what was not selected and why. It also lists rows to review by hand (column `review`): codes that belong to another taxon than the name, and alias matches with a different common name, which may be a form the taxonomy lumps into a wider species. Set `selected` to 0 for classes you do not want.

With `--checkpoint` and no `--class-pruning`, or with `--no-class-pruning`, every class of the checkpoint is exported. `labels.txt` is then copied from the labels file next to the checkpoint (`<name>_labels.txt` or `labels.txt`); without one, the names come from the checkpoint's vocabulary, and `labels.csv` is written if those are BirdNET IDs. The checkpoint is read with `torch.load(..., weights_only=True)`, and the model must match the architecture in `model.py` of the model directory given with `-m`.

## License

The source code is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Trained model weights are licensed under the [Apache License, Version 2.0](MODEL_LICENSE.md).

## Acceptable Use

BirdNET is built to support biodiversity research and conservation. See [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) for how we expect the BirdNET+ V3.0 models and the BirdNET+ Geomodel to be used, including uses we consider unacceptable and guidance on branding derivative work.

## Funding

Our work in the Cornell K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
