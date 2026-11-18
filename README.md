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

`convert_model.py` converts a BirdNET+ V3.0 PyTorch model (multi-head classifier on an EfficientNet backbone) into these formats, written to `export/<model name>/`:

| Format | Files | Input | FP16 | INT8 ¹ |
|---|---|---|---|---|
| TorchScript | `torchscript/model.pt` | audio | `model_fp16.pt` ¹ | — |
| ONNX | `onnx/model.onnx` | audio | `model_fp16.onnx`, `model_fp16_gpu.onnx` | `model_int8.onnx` |
| TFLite and TF SavedModel, preprocessing | `tf/preprocessing/` | audio | — | — |
| TFLite and TF SavedModel, main | `tf/main/` | spectrogram | `model_fp16.tflite` | `model_int8.tflite` |
| TFLite and TF SavedModel, combined | `tf/combined/` | audio | `model_fp16.tflite` | `model_int8.tflite` |
| Core ML | `coreml/model.mlpackage` | audio | `model_fp16.mlpackage` | `model_int8.mlpackage` |

¹ Only with `--experimental-mode`, see [Precision Variants](#precision-variants).

The default model is `02_BirdNET+_V3.0_Global_12K`. `01_BirdNET+_V3.0-preview3_Global_11K` is a legacy preview model: it still converts, but it is not tested, lacks the FP16 fixes of V3.0, and its ONNX exports return NaN on silent input. Do not publish models converted from it.

Audio input is mono, 32 kHz, float32 `[batch, samples]`. Supported input shapes:

| Format | Batch size | Audio length |
|---|---|---|
| TorchScript, ONNX | any | any, from 1025 samples |
| TFLite | fixed (`--batch-size`) | `--segment-duration` ² |
| TF SavedModel (main, combined), Core ML | fixed (`--batch-size`) | fixed (`--segment-duration`) |

² The TFLite files accept other lengths, but they are only accurate at the export length: at 5 s, a 3 s export differs from the PyTorch model by up to 0.08 in score and its embeddings are off by more than half their range. For another length, export again with that `--segment-duration`.

The separate preprocessing and main models let you plug in your own spectrogram pipeline or run the classifier on its own. The TF SavedModels are FP32 and are only kept when `-p` includes `fp32`; the preprocessing TFLite model is FP32 too and is always written, because the main models of every precision take its spectrogram. The export directory also gets a label file (see [Class Pruning and Labels](#class-pruning-and-labels)) and copies of the model license and acceptable use policy.

One export directory holds one model configuration: all formats share the label file. `export_config.json` records the classes, head weights, segment duration and batch size of the models in the directory. A run with another configuration removes the models exported before; a run with the same configuration keeps the formats it does not write.

Other scripts:

- **`test_models.py`** compares the converted models against the exported TorchScript model.
- **`convert_geomodel.py`**, **`test_geomodel.py`** convert and test the [BirdNET+ Geomodel](#birdnet-geomodel).
- **`fix_conv2dsame.py`**, **`fix_tflite_dynamic_input.py`** are used by the converter to make ONNX and TFLite exports accept dynamic input shapes.

## Setup

```bash
git clone https://github.com/birdnet-team/model-converter.git
cd model-converter
git lfs install && git lfs pull

python3 -m venv .venv && source .venv/bin/activate   # or a conda environment
pip install -r requirements.txt
```

Model checkpoints and the BirdNET taxonomy are stored with [Git LFS](https://git-lfs.com/); without it they are small pointer files. Python 3.11 or newer is required.

## Usage

Convert the default model (`02_BirdNET+_V3.0_Global_12K`) to all formats and all non-experimental precisions:

```bash
python convert_model.py
```

Common variations:

```bash
python convert_model.py -f onnx torchscript    # selected formats only
python convert_model.py -f onnx -p fp16        # ONNX FP16 only
python convert_model.py -o output/my_export    # write to output/my_export/<model name>/
python convert_model.py -m my_custom_model     # another model: a name in models/ or a path
python convert_model.py --list-models          # list the models in models/
```

### All Options

```
-m, --model-name         Model directory: a name in models/ or a path (default: 02_BirdNET+_V3.0_Global_12K)
-o, --output-dir         Directory that receives one folder per model (default: export/)
-f, --formats FMT        Output format(s): tf, tflite, torchscript, onnx, coreml (default: all)
-p, --precisions PREC    Precision variant(s): fp32, fp16, int8 (default: all); the SavedModels are fp32 only
--batch-size             Batch size of the fixed-shape models (default: 1)
--segment-duration       Audio length in seconds of the fixed-shape models (default: 3.0)
--head-weights W W W     Weights for the three classifier heads (default: from model definition)
--class-pruning XLSX     Keep classes whose selected column is 1 (default: labels_pruning.xlsx in the model directory if present)
--no-class-pruning       Export every class
--taxonomy CSV           BirdNET taxonomy for labels.csv (default: BirdNET+_Taxonomy_v0.5-Nov2026.csv)
--label-overrides CSV    Names that replace taxonomy names in labels.csv (default: label_overrides.csv in the model directory if present; none to disable)
--experimental-mode      Enable experimental variants
--list-models            List available models and exit
```

### Class Pruning and Labels

A class pruning sheet is an Excel file listing the model's classes, with an `ix` column (the model's output index) and a `selected` column. Only rows with `selected=1` are exported. A `labels_pruning.xlsx` in the model directory is used by default; pass `--class-pruning` for another sheet or `--no-class-pruning` to export every class.

For the default model, the sheet keeps one class per BirdNET taxonomy id, drops animal classes with fewer than 10 training samples, and keeps 16 non-animal sound classes (engine, siren, gun, music, ...). That leaves 11,725 of 16,852 classes, 11,709 of them animal species.

The export directory gets one label file, with one row per model output:

- **`labels.csv`** when the sheet has a `birdnet_id` column: `ix`, `birdnet_id`, scientific and common name, taxon group, iNaturalist id, eBird code and the common name translations, taken from the BirdNET taxonomy. Every kept class needs a unique `birdnet_id` that the taxonomy lists.
- **`labels_pruned.xlsx`** for a sheet without a `birdnet_id` column: the selected rows, with `ix` renumbered.
- **`labels.xlsx`** when classes are not pruned: a copy of the model directory's `labels.xlsx`.

A `label_overrides.csv` in the model directory replaces taxonomy names in `labels.csv`, keyed by `ix` and `birdnet_id`. The default model uses it to name the 16 sound classes.

### Head Weights

The model has three classifier heads whose logits are combined as a weighted sum before the sigmoid:

```
predictions = sigmoid(w0 * head1_logits + w1 * head2_logits + w2 * head3_logits)
```

The default weights are `[0.4, 0.35, 0.25]`. Heads with weight `0.0` are left out of the export, which shrinks it: the default model at FP32 takes about 345 MB with all three heads and about 140 MB with Head 1 alone.

```bash
python convert_model.py --head-weights 1.0 0.0 0.0   # Head 1 only
```

### Precision Variants

| Precision | Method | Notes |
|---|---|---|
| `fp32` | Standard export | Reference |
| `fp16` | float16 weights (Core ML: float16 compute) | ~2× smaller, small score differences |
| `int8` | Dynamic (ONNX), dynamic range (TFLite) or weight-only (Core ML) quantization | ~4× smaller, significant accuracy loss |

All INT8 variants and TorchScript FP16 are only exported with `--experimental-mode`. The INT8 models can change the predicted species; an accurate INT8 model would need quantization-aware training, which this converter does not do. TorchScript FP16 may produce NaN on models it was not tested with.

Largest score difference to the FP32 PyTorch model, for the default model with all three heads on 3 s windows of `example/soundscape.wav`:

| Variant | Max difference | Top species |
|---|---|---|
| ONNX FP32, TFLite FP32 | below 0.00001 | same |
| ONNX FP16 (both files), CPU and CUDA | 0.003–0.008 | same |
| TFLite FP16 | 0.001 | same |
| TorchScript FP16 | 0.003 | same |
| TFLite INT8, main model | 0.10 | same |
| ONNX INT8 | 0.11 | changed |
| TFLite INT8, combined model | 0.83 | changed |

On synthetic edge cases (silence, noise, clipping, very quiet or loud input) the ONNX FP16 models stay within 0.016. ONNX Runtime on x86 CPUs computes float16 graphs in float32, so only the CUDA figures reflect real float16 arithmetic; the models were not measured on arm64 devices. Silence is the least stable input for FP16 models: a TFLite FP16 model exported for 2 s segments differed by 0.04 on all-zero input, while the 3 s export stays within 0.001.

ONNX has two FP16 files of the same size and with the same inputs and outputs as the FP32 model:

- `model_fp16.onnx` (default) runs the backbone convolutions in float16 and everything else in float32. This suits CPUs, arm64 in particular.
- `model_fp16_gpu.onnx` runs the whole backbone in float16 and is faster on GPUs.

### Core ML

`-f coreml` writes Core ML ML Programs (`.mlpackage`) for iOS 16 / macOS 13 and later. Like the TorchScript and ONNX models, they take raw audio and run the preprocessing themselves.

- **Input** `input`: float32 `[batch, samples]`, fixed at `--batch-size` × `--segment-duration` × 32000 (default `[1, 96000]`).
- **Outputs** `predictions` (per-class scores, in label order) and `embeddings`.
- **Compute units:** only `model_fp16.mlpackage` can use the Neural Engine. The FP32 and INT8 models run on the GPU or CPU.
- **FP16 accuracy** depends on where the model runs: on the example recording, scores are within about 0.01 of the PyTorch model on the GPU, 0.02 on the CPU and 0.04 on the Neural Engine.
- **Silence:** for all-zero input the FP16 model's embeddings drift from the PyTorch model's, and with some head weight settings single scores do too. Use the FP32 model if you need exact output for silent segments.

`coremltools` converts on any OS, but the models only run on macOS and iOS:

```python
import coremltools as ct
model = ct.models.MLModel('export/02_BirdNET+_V3.0_Global_12K/coreml/model.mlpackage')
scores = model.predict({'input': audio})['predictions']   # audio: float32 [1, 96000]
```

### Testing Converted Models

`test_models.py` compares the scores and embeddings of every converted model in an export directory with the TorchScript model, on audio and on silence, and then times each one. That covers the ONNX files, the TFLite and SavedModel combined models, the preprocessing models chained into the main models, and on macOS the Core ML models:

```bash
python test_models.py
python test_models.py -a path/to/audio.wav
python test_models.py -e output/my_export/<model name>
python test_models.py --durations 3.0 5.0 --gpu
```

```
-e, --export-dir       Path to a model's export directory (default: export/<model name>/)
-a, --audio            Audio file for test input (default: example/soundscape.wav)
--batch-size           Batch size the fixed-shape models were exported with (default: 1)
--segment-duration     Segment duration the fixed-shape models were exported with (default: 3.0)
--durations            Segment durations in seconds to test (default: the segment duration and 5.0)
--expect MODEL ...     Models that must be found and compared; the test fails if one is missing
--gpu                  Use GPU for TorchScript and ONNX inference
--speed-runs           Timed runs per model in the speed test (default: 10; 0 skips it)
```

A model passes if its scores match TorchScript within 0.0001 (FP32), 0.02 (FP16) or 0.05 (Core ML FP16), and its embeddings within a matching share of their range. Models with a dynamic batch size are also run on a batch of three different segments. Fixed-shape models are tested at the shape they were exported with; TFLite models are also run at the other durations, where they must stay within 0.15 and keep the top class. INT8 models are reported but not checked, apart from returning finite output of the right shape. The test fails if no model was compared.

Throughput in seconds of audio per second on one machine (x86 CPU, NVIDIA TITAN RTX; default model, all three heads, 3 s segments), as a rough guide:

| Variant | CPU, batch 1 | GPU, batch 32 |
|---|---|---|
| TorchScript FP32 | 21 | 734 |
| ONNX FP32 | 25 | 662 |
| ONNX FP16 (`model_fp16.onnx`) | 24 | 749 |
| ONNX FP16 for GPUs (`model_fp16_gpu.onnx`) | 24 | 959 |
| TFLite FP32 | 48 | — |
| TFLite FP16 | 45 | — |

TorchScript FP16 is a GPU format (1069 on the GPU, far slower than FP32 on a CPU), and ONNX INT8 is much slower than FP32 on an x86 CPU.

A [GitHub Actions workflow](.github/workflows/conversion-tests.yml) converts the default model and the geomodel and runs both test scripts, with the Core ML models on a macOS runner, whenever a change touches the converter.

## BirdNET+ Geomodel

The [BirdNET+ Geomodel](https://github.com/birdnet-team/geomodel) predicts which species occur at a place in a given week of the year, typically to filter the audio model's detections by location and season. `convert_geomodel.py` converts a geomodel checkpoint; the default is `models/03_BirdNET+_Geomodel_V3.0.5_Global_14K`. V3.0.5 has the trained weights of the geomodel V3.0.4 release, with the classes reduced to those in the BirdNET taxonomy and labelled with BirdNET IDs.

```bash
python convert_geomodel.py                     # default geomodel, all formats, FP32 and FP16
python convert_geomodel.py -f onnx -p fp16     # ONNX FP16 only
python convert_geomodel.py --batch-size 32     # batch size of the LiteRT, SavedModel and Core ML models
python test_geomodel.py                        # compare every converted file with TorchScript
```

Every model takes one float32 tensor `[batch, 3]` with the columns latitude (degrees, -90 to 90), longitude (degrees, -180 to 180) and week (1 to 48, four weeks per month), and returns one value per class, in label order:

| Format | Output in `export/<model name>/` | Returns | Batch size |
|---|---|---|---|
| PyTorch checkpoint | `checkpoint/model.pt`, `model_labels.txt` | — | — |
| TorchScript | `torchscript/model.pt` | raw logits ¹ | any |
| ONNX | `onnx/model.onnx`, `model_fp16.onnx` | probabilities | any |
| LiteRT (TFLite) | `tf/model.tflite`, `model_fp16.tflite`, `model_int8.tflite` ² | probabilities | fixed (`--batch-size`) |
| TF SavedModel | `tf/` | probabilities | fixed (`--batch-size`) |
| Core ML | `coreml/model.mlpackage`, `model_fp16.mlpackage` | probabilities | fixed (`--batch-size`) |

¹ As in the geomodel releases, so that existing loaders keep working; apply a sigmoid, or export with `--torchscript-sigmoid`.  
² Only with `--experimental-mode`.

The checkpoint in `checkpoint/` holds the exported classes with weights, model configuration and species vocabulary. It loads with `load_checkpoint` in the model directory's `model.py` and with the geomodel repository's `convert.py` and `predict.py`.

Compared with the V3.0.4 release files, the FP16 models are more accurate, the LiteRT models use built-in operators only (no Flex delegate), LiteRT and the SavedModel have a fixed batch size, Core ML is new, and there is no INT8 file by default.

Largest difference in output probability to the original PyTorch model, over 3000 random locations and weeks:

| Model | Max difference | Top class agrees | Size |
|---|---|---|---|
| TorchScript, ONNX, LiteRT FP32, TF SavedModel | below 0.00004 | 100% | 15 MB |
| ONNX FP16 | 0.005 (V3.0.4 release file: 0.09) | 99.9% | 7 MB |
| LiteRT FP16 | 0.004 (V3.0.4 release file: 0.06) | 99.8% | 7 MB |
| Core ML FP32 ³ | 0.00001 | 100% | 15 MB |
| Core ML FP16 ³ | 0.003 | 99.4% | 7 MB |
| LiteRT INT8 ² | 0.19 | 89% | 4 MB |

³ On macOS (arm64), over 512 test inputs that include every week and coordinates at round fractions of 180°.

The converter also checks every file it writes against the PyTorch model and fails if one is out of tolerance. Off macOS it cannot run the Core ML models and only checks their operations; run `test_geomodel.py` on a Mac to compare them.

The Core ML models are ML Programs for iOS 16 / macOS 13 and later (input `input`, output `probabilities`). Both compute in float32; `model_fp16.mlpackage` stores its weights as float16, like the ONNX FP16 model, which is seven times closer to the PyTorch model than computing in float16 (0.003 against 0.018). Core ML's `sin` returns the wrong sign at some exact angles, which weeks 2, 3, 4 and 7 and coordinates at multiples of 7.5° produce, so the converter writes every sine as `cos(x - π/2)`. The models have been run on macOS, not on iOS devices.

### Geomodel labels

The export directory gets `labels.csv`, with the same columns as the audio model's `labels.csv`, and `labels.txt` in the geomodel's own format (`id<TAB>scientific name<TAB>common name`) with the BirdNET ID as the id. The audio model and the geomodel cover different sets of classes; match rows by `birdnet_id`.

Which outputs are exported comes from a class pruning sheet, as for the audio model. The sheet belongs to one checkpoint: the converter stops if the sheet and the checkpoint's species vocabulary name different classes.

The default model's checkpoint is already reduced to its 14,057 selected classes. `labels_pruning_source.xlsx` in the model directory lists all 14,082 classes of the release and why the others were dropped.

### Converting your own geomodel

A checkpoint trained with the [geomodel repository](https://github.com/birdnet-team/geomodel) comes with a `labels.txt` of eBird codes and iNaturalist ids. Match it to the taxonomy, review the sheet, then convert:

```bash
python convert_geomodel.py --map-labels my_run/checkpoint_best_labels.txt --sheet-out my_run/pruning.xlsx
python convert_geomodel.py --checkpoint my_run/checkpoint_best.pt --class-pruning my_run/pruning.xlsx -o export/my_run
```

`--map-labels` matches each label to the taxonomy by scientific name, code or alias, selects one class per BirdNET ID, and marks matches to check in a `review` column. Where the taxonomy lumps several source classes into one BirdNET ID, one of them is kept and the others are dropped, so that class predicts only the range of the kept form; the `review` column names these. Set `selected` to 0 for classes you do not want. Without a pruning sheet, every class of the checkpoint is exported. With `--checkpoint`, the export folder is named after the checkpoint file (here `export/my_run/checkpoint_best/`). The model must match the architecture in `model.py` of the model directory given with `-m`.

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
