# Copilot Instructions — BirdNET Model Converter

## Project Overview

This repository converts BirdNET V3.0 PyTorch audio classification models into other inference formats:
TensorFlow SavedModel, TFLite/LiteRT, TorchScript, ONNX, and Core ML.

The project is **experimental** — conversion pipelines often require post-processing hacks
to work around limitations in export tools (baked static shapes, unsupported ops, etc.).

## Repository Structure

- `convert_model.py` — Main conversion pipeline (TF preprocessing, TF main model, combined TFLite, TorchScript, ONNX, Core ML).
- `fix_conv2dsame.py` — ONNX post-processor: replaces static Conv2dSame padding with dynamic padding subgraphs.
- `fix_tflite_dynamic_input.py` — TFLite post-processor: patches static batch/temporal dimensions in flatbuffers.
- `test_models.py` — Comparison harness: runs all converted formats against the TorchScript reference.
- `models/` — Model directories containing PyTorch checkpoints and `model.py` definitions.
- `example/` — Example audio files for testing (`soundscape.wav`).

## Architecture

The BirdNET V3.0 model uses:
- **Backbone:** EfficientNetV2-S (from `timm`, pretrained on ImageNet-21k)
- **Preprocessing:** Audio → FFT → Mel spectrogram → dB scaling → normalize → resize → 3-channel image
- **Heads:** Three parallel classification heads (GeM pooling, frequency attention, spectral attention) averaged with sigmoid
- **Classes:** ~11K bird species

## Key Technical Constraints

- **Dynamic shapes:** All exported models must support variable-length audio input and (where possible) dynamic batch sizes.
  - `litert_torch` traces with static shapes; the TFLite fix script patches RESHAPE/SLICE shape tensors afterward.
  - ONNX `torch.export` bakes Conv2dSame padding and Resize output sizes; separate fix scripts patch these.
- **TFLite limitations:** `F.avg_pool1d` lowers to unsupported StableHLO ops; the model uses depthwise conv1d as a workaround.
- **Preprocessing is split:** TF exports three TFLite variants: preprocessing-only (spectrogram), main-only (classifier, spectrogram input), and combined (audio → predictions). The split lets users swap in their own spectrogram pipeline or run classifier on pre-computed spectrograms. TorchScript/ONNX/Core ML use the PyTorch `PreprocessingLayerTorch` and export end-to-end.
- **Core ML:** traced with `torch.jit.trace` at a fixed `(batch, samples)` shape and converted with coremltools; custom op converters live in `_register_coreml_torch_ops()`. Core ML models only run on macOS, so `test_models.py` skips them elsewhere.
- **Precision variants:** The converter supports `fp32`, `fp16`, and `int8` exports for TFLite, TorchScript, and ONNX. TFLite INT8 uses representative-dataset quantization (calibration audio). ONNX and TorchScript use dynamic quantization for INT8. **INT8 is unstable** — dynamic quantization introduces severe accuracy loss and is not recommended for production.

## Coding Conventions

- Python 3.10+, follow PEP 8.
- `snake_case` for functions/variables, `PascalCase` for classes.
- Keep lines ≤100 characters where practical.
- No additional dependencies without discussion — the requirements are already heavy (PyTorch, TensorFlow, ONNX, etc.).

## When Modifying Conversion Code

- Always verify converted model outputs match the PyTorch reference using `test_models.py`.
- Be aware that `torch.export` / `litert_torch` / `tf.lite.TFLiteConverter` each have their own quirks with dynamic shapes.
- The fix scripts (`fix_conv2dsame.py`, `fix_tflite_dynamic_input.py`) operate on raw model bytes/graphs — changes here need careful testing.
- Model definitions live inside `models/.../pytorch/model.py`, not at the repo root.
- FP16/INT8 variants are generated alongside FP32 when requested via `-p`. Changes to the conversion functions must handle all precision paths.
