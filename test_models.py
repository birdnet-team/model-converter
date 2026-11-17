"""
Compare every converted audio model in an export directory with the TorchScript model.

All models take raw audio ``[batch, samples]`` (mono, 32 kHz, float32) and return
predictions and embeddings; both are compared:

- TorchScript FP32 is the reference (``torchscript/model.pt``)
- TorchScript FP16, ONNX: any batch size and length
- TFLite combined, and TFLite preprocessing chained into TFLite main: fixed batch size;
  other lengths than the export length run, with a looser tolerance
- TF SavedModel combined, and preprocessing chained into main: fixed shape
- Core ML: fixed shape, macOS only

INT8 models are experimental: their differences are reported but do not fail the test,
as long as the output has the right shape and is finite.
"""

import argparse
import csv
import os
import sys
import time

import librosa
import numpy as np
import onnxruntime as ort
import torch
from ai_edge_litert.interpreter import Interpreter

MODEL_NAME = '02_BirdNET+_V3.0_Global_12K'

SAMPLE_RATE = 32000
DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export', MODEL_NAME)
DEFAULT_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example', 'soundscape.wav')

# Largest allowed difference in score to the TorchScript reference. Measured on the example
# recording: FP32 formats 0.000002, ONNX and TFLite FP16 0.002 (ONNX on CUDA 0.008),
# Core ML FP16 0.01 to 0.04 depending on the compute unit.
TOLERANCE = {'fp32': 1e-4, 'fp16': 0.02, 'coreml_fp16': 0.05}
# TFLite at another length than it was exported with: the resize of the spectrogram is
# baked in for the export length (FP32 0.05, FP16 0.08 at 5 s for a 3 s export)
OFF_LENGTH_TOLERANCE = 0.15
# Embeddings are compared relative to the largest reference embedding value
EMBEDDING_TOLERANCE = {'fp32': 1e-4, 'fp16': 0.05, 'coreml_fp16': 0.1}
TOP_K = 3
DYNAMIC_BATCH = 3  # rows of the extra batch test for models with a dynamic batch size


# ── Audio ────────────────────────────────────────────────────────────────────

def load_audio(audio_path, duration, batch_size):
    """Return ``(batch_size, samples)`` float32: consecutive windows of the recording.

    Every row is a different part of the recording, so a model that mixes up or drops
    rows is caught. Rows beyond the end of the recording are zero-padded.
    """
    samples = int(duration * SAMPLE_RATE)
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=duration * batch_size,
                            offset=0.0, dtype=np.float32)
    audio = np.pad(audio, (0, max(0, samples * batch_size - len(audio))))
    return np.ascontiguousarray(audio[:samples * batch_size].reshape(batch_size, samples))


# ── Models ───────────────────────────────────────────────────────────────────

class Candidate:
    """One converted model (or chain of two) with a uniform ``run(audio)``.

    ``run`` returns ``(predictions, embeddings)`` as float32 arrays.
    """

    dynamic_batch = False   # accepts any batch size
    fixed_length = False    # runs only at the length it was exported with
    off_length = False      # runs at other lengths, less accurately

    def __init__(self, name, precision, paths, where='CPU'):
        self.name, self.precision, self.paths, self.where = name, precision, paths, where

    @property
    def experimental(self):
        return self.precision == 'int8'

    def load(self):
        pass

    def input_shape(self):
        """Fixed input shape, or None if the model accepts any."""
        return None

    def run(self, audio):
        raise NotImplementedError


class TorchScriptModel(Candidate):
    dynamic_batch = True

    def __init__(self, name, precision, path, device):
        super().__init__(name, precision, [path], 'CUDA' if device != 'cpu' else 'CPU')
        self.device = device

    def load(self):
        self.model = torch.jit.load(self.paths[0], map_location=self.device).eval()

    def run(self, audio):
        with torch.no_grad():
            predictions, embeddings = self.model(torch.from_numpy(audio).to(self.device))
            if self.device != 'cpu':
                torch.cuda.synchronize()
        return predictions.float().cpu().numpy(), embeddings.float().cpu().numpy()


class OnnxModel(Candidate):
    dynamic_batch = True

    def __init__(self, name, precision, path, use_gpu):
        super().__init__(name, precision, [path])
        self.use_gpu = use_gpu

    def load(self):
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu
                     else ['CPUExecutionProvider'])
        self.session = ort.InferenceSession(self.paths[0], providers=providers)
        self.where = self.session.get_providers()[0].replace('ExecutionProvider', '')
        if self.use_gpu and self.where != 'CUDA':
            raise RuntimeError('the CUDA execution provider could not be loaded')
        self.input_name = self.session.get_inputs()[0].name
        names = [output.name for output in self.session.get_outputs()]
        self.outputs = ['predictions', 'embeddings']
        if set(self.outputs) - set(names):
            raise RuntimeError(f'outputs are {names}, expected predictions and embeddings')

    def run(self, audio):
        predictions, embeddings = self.session.run(self.outputs, {self.input_name: audio})
        return predictions, embeddings


class _TfliteRunner:
    """A TFLite model called through its serving signature, resized on demand."""

    def __init__(self, path):
        self.interpreter = Interpreter(model_path=path, num_threads=os.cpu_count())
        signature = self.interpreter.get_signature_list()['serving_default']
        self.input_name = signature['inputs'][0]
        self.runner = self.interpreter.get_signature_runner('serving_default')

    def __call__(self, array):
        return self.runner(**{self.input_name: array})


class TfliteModel(Candidate):
    """TFLite combined model, or TFLite preprocessing chained into a TFLite main model."""

    off_length = True

    def load(self):
        self.runners = [_TfliteRunner(path) for path in self.paths]

    def run(self, audio):
        x = audio
        if len(self.runners) == 2:
            spectrogram = next(iter(self.runners[0](x).values()))   # [B, H, W, C]
            x = np.transpose(spectrogram, (0, 3, 1, 2))             # [B, C, H, W]
        outputs = self.runners[-1](x)
        return outputs['predictions'], outputs['embeddings']


class SavedModel(Candidate):
    """TF SavedModel combined model, or preprocessing chained into the main model."""

    fixed_length = True

    def load(self):
        import tensorflow as tf
        self.tf = tf
        self.functions = [tf.saved_model.load(path).signatures['serving_default']
                          for path in self.paths]

    def input_shape(self):
        if len(self.functions) == 2:
            return None  # the preprocessing model accepts any shape; main checks its own
        spec = next(iter(self.functions[0].structured_input_signature[1].values()))
        return tuple(spec.shape.as_list())

    def _call(self, function, array):
        name = next(iter(function.structured_input_signature[1]))
        return function(**{name: self.tf.constant(array)})

    def run(self, audio):
        x = audio
        if len(self.functions) == 2:
            spectrogram = next(iter(self._call(self.functions[0], x).values())).numpy()
            x = np.transpose(spectrogram, (0, 3, 1, 2))
            expected = tuple(next(iter(
                self.functions[1].structured_input_signature[1].values())).shape.as_list())
            if x.shape != expected:
                raise FixedShape(f'main model input is fixed at {expected}')
        outputs = self._call(self.functions[-1], x)
        return outputs['predictions'].numpy(), outputs['embeddings'].numpy()


class CoreMLModel(Candidate):
    fixed_length = True

    def __init__(self, name, precision, path):
        super().__init__(name, precision, [path], 'Core ML')

    def load(self):
        import coremltools as ct
        self.model = ct.models.MLModel(self.paths[0])  # default compute units, as an app gets them

    def input_shape(self):
        return tuple(self.model.get_spec().description.input[0].type.multiArrayType.shape)

    def run(self, audio):
        out = self.model.predict({'input': audio})
        return np.asarray(out['predictions']), np.asarray(out['embeddings'])


class FixedShape(Exception):
    """The model cannot take this input shape by design."""


def find_candidates(export_dir, device, use_gpu):
    """Return the reference and all other models found in ``export_dir``, SavedModels last.

    TensorFlow adds its operators to the LiteRT runtime once it is imported, which would
    hide a TFLite model that needs them, so the SavedModels are loaded and run last.
    """
    def path(*parts):
        return os.path.join(export_dir, *parts)

    reference = TorchScriptModel('TS', 'fp32', path('torchscript', 'model.pt'), device)
    candidates = [TorchScriptModel('TS_fp16', 'fp16', path('torchscript', 'model_fp16.pt'), device)]
    for name, precision, file_name in (('ONNX', 'fp32', 'model.onnx'),
                                       ('ONNX_fp16', 'fp16', 'model_fp16.onnx'),
                                       ('ONNX_fp16_gpu', 'fp16', 'model_fp16_gpu.onnx'),
                                       ('ONNX_int8', 'int8', 'model_int8.onnx')):
        candidates.append(OnnxModel(name, precision, path('onnx', file_name),
                                    use_gpu and precision != 'int8'))
    preprocessing = path('tf', 'preprocessing', 'model.tflite')
    for suffix, precision in (('', 'fp32'), ('_fp16', 'fp16'), ('_int8', 'int8')):
        candidates.append(TfliteModel(f'LiteRT{suffix}', precision,
                                      [path('tf', 'combined', f'model{suffix}.tflite')]))
        candidates.append(TfliteModel(f'LRT_main{suffix}', precision,
                                      [preprocessing, path('tf', 'main', f'model{suffix}.tflite')]))
    for suffix, precision in (('', 'fp32'), ('_fp16', 'fp16'), ('_int8', 'int8')):
        candidates.append(CoreMLModel(f'CoreML{suffix}', 'coreml_fp16' if precision == 'fp16' else precision,
                                      path('coreml', f'model{suffix}.mlpackage')))
    candidates.append(SavedModel('SavedModel', 'fp32', [path('tf', 'combined')]))
    candidates.append(SavedModel('SM_main', 'fp32', [path('tf', 'preprocessing'), path('tf', 'main')]))
    return reference, candidates


def model_exists(candidate):
    """True if every file of the candidate is in the export directory."""
    def exists(path):
        if os.path.isdir(path) and not path.endswith('.mlpackage'):
            return os.path.isfile(os.path.join(path, 'saved_model.pb'))
        return os.path.exists(path)
    return all(exists(path) for path in candidate.paths)


# ── Comparison ───────────────────────────────────────────────────────────────

def fmt_topk(out, k=TOP_K):
    """Format the top-k class indices and scores of the first batch row."""
    topk = np.argsort(out[0])[-k:][::-1]
    return " ".join(f"{i}({out[0][i]:.3f})" for i in topk)


def compare(candidate, case, reference, output, off_length=False):
    """Print one comparison and return 'PASS', 'FAIL' or 'INFO' (experimental model)."""
    (ref_pred, ref_emb), (pred, emb) = reference, output
    pred, emb = np.asarray(pred, dtype=np.float32), np.asarray(emb, dtype=np.float32)
    problems = []
    if pred.shape != ref_pred.shape or emb.shape != ref_emb.shape:
        problems.append(f'output shapes {pred.shape}, {emb.shape} instead of '
                        f'{ref_pred.shape}, {ref_emb.shape}')
    elif not (np.isfinite(pred).all() and np.isfinite(emb).all()):
        problems.append('output is not finite')
    if problems:
        print(f"❌ FAIL  {candidate.name:<14} {case}  {problems[0]}")
        return 'FAIL'

    max_diff = float(np.abs(ref_pred - pred).max())
    emb_diff = float(np.abs(ref_emb - emb).max() / max(float(np.abs(ref_emb).max()), 1e-12))
    same_top1 = bool((ref_pred.argmax(1) == pred.argmax(1)).all())
    note = ''
    if candidate.experimental:
        status, icon, limits = 'INFO', 'ℹ️  INFO', 'experimental, not checked'
    else:
        tolerance = OFF_LENGTH_TOLERANCE if off_length else TOLERANCE[candidate.precision]
        emb_tolerance = None if off_length else EMBEDDING_TOLERANCE[candidate.precision]
        passed = max_diff <= tolerance and (emb_tolerance is None or emb_diff <= emb_tolerance)
        if off_length:
            passed = passed and same_top1
            note = '  (not the export length)'
        status, icon = ('PASS', '✅ PASS') if passed else ('FAIL', '❌ FAIL')
        limits = f'tolerance {tolerance:g}' + (f' / {emb_tolerance:g}' if emb_tolerance else '')
    print(f"{icon}  {candidate.name:<14} {case}  max_diff={max_diff:.6f}  "
          f"embeddings={emb_diff:.6f}  top-1 {'same' if same_top1 else 'differs'}  "
          f"({limits}){note}")
    if status != 'PASS':
        print(f"         TS:    {fmt_topk(ref_pred)}")
        print(f"         model: {fmt_topk(pred)}")
    return status


def count_labels(export_dir):
    """Return ``(file name, number of label rows)`` for the export's label file, or None."""
    path = os.path.join(export_dir, 'labels.csv')
    if os.path.isfile(path):
        with open(path, newline='', encoding='utf-8-sig') as f:
            return 'labels.csv', sum(1 for _ in csv.reader(f)) - 1
    for name in ('labels_pruned.xlsx', 'labels.xlsx'):
        path = os.path.join(export_dir, name)
        if os.path.isfile(path):
            import pandas as pd
            return name, len(pd.read_excel(path))
    return None


# ── Speed test ───────────────────────────────────────────────────────────────

def speed_test(models, audio, n_runs):
    """Time every model on the same input and print a table.

    Each model gets two warm-up runs and ``n_runs`` timed runs; the median is reported,
    with the audio throughput (seconds of audio per second) and the speed relative to
    the TorchScript FP32 reference.
    """
    batch_size, samples = audio.shape
    audio_seconds = batch_size * samples / SAMPLE_RATE
    print(f"Speed test: batch={batch_size}, {samples / SAMPLE_RATE:.1f} s segments, "
          f"median of {n_runs} runs after 2 warm-up runs")
    print(f"  {'model':<18}{'runs on':<10}{'ms/batch':>10}{'x real-time':>13}{'vs TS':>8}")
    ts_ms = None
    for model in models:
        shape = model.input_shape()
        if shape is not None and shape != audio.shape:
            print(f"  {model.name:<18}skipped: model input is fixed at {shape}")
            continue
        try:
            for _ in range(2):
                model.run(audio)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model.run(audio)
                times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"  {model.name:<18}failed: {str(e).split(chr(10))[0][:70]}")
            continue
        ms = float(np.median(times)) * 1000.0
        if model.name == 'TS':
            ts_ms = ms
        rel = f"{ts_ms / ms:.2f}x" if ts_ms else ''
        print(f"  {model.name:<18}{model.where:<10}{ms:>10.1f}"
              f"{audio_seconds / (ms / 1000.0):>13.0f}{rel:>8}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    parser = argparse.ArgumentParser(
        description='Compare converted model outputs against the TorchScript reference.',
    )
    parser.add_argument(
        '-e', '--export-dir', default=DEFAULT_EXPORT_DIR,
        help="Path to a model's export directory (default: export/<model name>/)",
    )
    parser.add_argument(
        '-a', '--audio', default=DEFAULT_AUDIO,
        help='Audio file for test input (default: example/soundscape.wav)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size the fixed-shape models were exported with (default: 1)',
    )
    parser.add_argument(
        '--segment-duration', type=float, default=3.0,
        help='Segment duration in seconds the fixed-shape models were exported with (default: 3.0)',
    )
    parser.add_argument(
        '--durations', type=float, nargs='+', default=None,
        help='Segment durations in seconds to test (default: the segment duration and 5.0)',
    )
    parser.add_argument(
        '--expect', nargs='+', default=[], metavar='MODEL',
        help='Model names that must be found and compared, e.g. ONNX LiteRT_fp16; '
             'the test fails if one is missing or skipped',
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU for TorchScript and ONNX (LiteRT and TensorFlow run on the CPU; '
             'Core ML picks its own units)',
    )
    parser.add_argument(
        '--speed-runs', type=int, default=10,
        help='Timed runs per model in the speed test, at the segment duration (default: 10; 0 skips it)',
    )
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()
    durations = args.durations or sorted({args.segment_duration, 5.0})
    if args.batch_size < 1 or args.speed_runs < 0 or args.segment_duration <= 0:
        sys.exit('Error: --batch-size must be at least 1, --speed-runs 0 or more and '
                 '--segment-duration positive')
    if min(durations) * SAMPLE_RATE < 1025:
        sys.exit('Error: --durations must be at least 1025 samples (0.033 s) long')
    if not os.path.isfile(args.audio):
        sys.exit(f'Error: audio file not found: {args.audio}')
    if args.gpu and not torch.cuda.is_available():
        sys.exit('Error: --gpu given but CUDA is not available')
    device = 'cuda' if args.gpu else 'cpu'
    export_dir = os.path.abspath(args.export_dir)

    print(f"Model comparison (reference: TorchScript)  |  tolerance: "
          + ", ".join(f"{k} {v:g}" for k, v in TOLERANCE.items())
          + f", other length (TFLite) {OFF_LENGTH_TOLERANCE:g}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name()})" if args.gpu else ""))
    print(f"Audio: {args.audio}")
    print(f"Export dir: {export_dir}\n")

    if args.gpu and hasattr(ort, 'preload_dlls'):
        # Load the CUDA/cuDNN libraries that ship with PyTorch; without them ONNX Runtime
        # cannot load its CUDA provider
        ort.preload_dlls()

    reference, candidates = find_candidates(export_dir, device, args.gpu)
    unknown = set(args.expect) - {c.name for c in candidates}
    if unknown:
        sys.exit(f'Error: --expect names unknown model(s) {", ".join(sorted(unknown))}; '
                 f'known: {", ".join(c.name for c in candidates)}')
    if not model_exists(reference):
        sys.exit(f'Error: TorchScript reference not found at {reference.paths[0]}')
    reference.load()

    # Inputs and reference outputs: every duration at the export batch size, a larger
    # batch for the models with a dynamic batch size, and silence. A constant spectrogram
    # is the degenerate case of the per-sample normalization (0/0); every model must
    # return finite scores that match the reference.
    cases = []
    for duration in durations:
        cases.append((f'dur={duration:.1f}s batch={args.batch_size}',
                      load_audio(args.audio, duration, args.batch_size), False))
    cases.append((f'dur={args.segment_duration:.1f}s batch={DYNAMIC_BATCH}',
                  load_audio(args.audio, args.segment_duration, DYNAMIC_BATCH), True))
    cases.append((f'silence  batch={args.batch_size}',
                  np.zeros((args.batch_size, int(args.segment_duration * SAMPLE_RATE)), np.float32), False))
    references = [reference.run(audio) for _, audio, _ in cases]
    native_samples = int(args.segment_duration * SAMPLE_RATE)

    results, compared, loaded = [], set(), [reference]
    labels = count_labels(export_dir)
    n_outputs = references[0][0].shape[1]
    if labels is None:
        print('⚠️  No label file in the export directory')
    elif labels[1] != n_outputs:
        print(f"❌ FAIL  {labels[0]} has {labels[1]} rows but the model has {n_outputs} outputs")
        results.append('FAIL')
    else:
        print(f"✅ PASS  {labels[0]} has one row per model output ({n_outputs})")
        results.append('PASS')
    print()

    for candidate in candidates:
        if not model_exists(candidate):
            print(f"   ----  {candidate.name:<14} not in the export directory")
            continue
        if isinstance(candidate, CoreMLModel) and sys.platform != 'darwin':
            print(f"   ----  {candidate.name:<14} needs macOS")
            continue
        try:
            candidate.load()
        except Exception as e:
            print(f"⚠️ ERR   {candidate.name:<14} does not load: {str(e).split(chr(10))[0][:100]}")
            results.append('ERROR')
            continue
        loaded.append(candidate)
        for (case, audio, batch_test), ref in zip(cases, references):
            if batch_test and not candidate.dynamic_batch:
                continue
            shape = candidate.input_shape()
            off_length = candidate.off_length and audio.shape[1] != native_samples
            if (shape is not None and shape != audio.shape) or (
                    candidate.fixed_length and audio.shape[1] != native_samples):
                print(f"   skip  {candidate.name:<14} {case}  (fixed input shape)")
                continue
            try:
                status = compare(candidate, case, ref, candidate.run(audio), off_length)
            except FixedShape as e:
                print(f"   skip  {candidate.name:<14} {case}  ({e})")
                continue
            except Exception as e:
                print(f"⚠️ ERR   {candidate.name:<14} {case}  {str(e).split(chr(10))[0][:100]}")
                status = 'ERROR'
            results.append(status)
            if status in ('PASS', 'FAIL', 'INFO'):
                compared.add(candidate.name)
    print()

    if args.speed_runs > 0:
        speed_test(loaded, load_audio(args.audio, args.segment_duration, args.batch_size),
                   args.speed_runs)

    n_pass, n_fail, n_error = (results.count(s) for s in ('PASS', 'FAIL', 'ERROR'))
    print(f"{n_pass} passed, {n_fail} failed, {n_error} errors, "
          f"{results.count('INFO')} experimental (not checked)")
    missing = [name for name in args.expect if name not in compared]
    if missing:
        print(f"❌ Expected model(s) not compared: {', '.join(missing)}")
    if not compared:
        print('❌ No converted model was compared with the reference')
    sys.exit(0 if n_fail == 0 and n_error == 0 and compared and not missing else 1)


if __name__ == '__main__':
    main()
