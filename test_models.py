"""
Compare TorchScript (reference) against ONNX, LiteRT, and quantized/reduced-precision
model outputs across different batch sizes and segment durations.

All models take raw audio [batch, samples].
- TorchScript  → (embeddings, predictions)
- ONNX         → (embeddings, predictions)
- LiteRT       → predictions   (combined preprocessing + main)
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import librosa
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

MODEL_NAME = '03_BirdNET+_V3.0-preview4_Global_12K_EffNetB0' # 01_BirdNET+_V3.0-preview3_Global_11K 03_BirdNET+_V3.0-preview4_Global_12K_EffNetB0

SAMPLE_RATE = 32000
DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export', MODEL_NAME)
DEFAULT_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example', 'soundscape.wav')

# Tolerance for "match" (absolute + relative)
ATOL = 0.09 # 1e-3
ATOL_INT8 = 0.1 # 0.025   # INT8 dynamic quantization introduces larger per-output error
RTOL = 1e-5
TOP_K = 3


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio(audio_path: str, duration: float, batch_size: int) -> np.ndarray:
    """Load or generate an audio batch.

    Reads ``duration`` seconds from ``audio_path`` at 32 kHz, zero-pads or
    truncates to the exact sample count, and tiles to ``batch_size`` rows.
    Falls back to deterministic random noise if the file is missing.

    Returns:
        np.ndarray: Float32 array of shape ``(batch_size, samples)``.
    """
    samples = int(duration * SAMPLE_RATE)
    if audio_path and os.path.isfile(audio_path):
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=duration,
                                 offset=0.0, dtype=np.float32)
        assert sr == SAMPLE_RATE
    else:
        if audio_path:
            print(f"Warning: audio file not found: {audio_path} — using random noise")
        rng = np.random.default_rng(seed=42)
        audio = rng.standard_normal(samples).astype(np.float32) * 0.1
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    else:
        audio = audio[:samples]
    return np.tile(audio[np.newaxis, :], (batch_size, 1))


# ── Model runners ────────────────────────────────────────────────────────────

def run_torchscript(model, audio_np: np.ndarray, device: str, input_dtype=torch.float32) -> np.ndarray:
    """Run inference with a TorchScript model.

    Returns:
        tuple[np.ndarray, float]: Predictions array and elapsed time in seconds.
    """
    with torch.no_grad():
        audio_t = torch.from_numpy(audio_np).to(device=device, dtype=input_dtype)
        t0 = time.perf_counter()
        #embeddings, predictions = model(audio_t)
        predictions, embeddings = model(audio_t)
        if device != 'cpu':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return predictions.float().cpu().numpy(), dt


def run_onnx(session, audio_np: np.ndarray) -> np.ndarray:
    """Run inference with an ONNX Runtime session.

    Returns:
        tuple[np.ndarray, float]: Predictions array and elapsed time in seconds.
    """
    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: audio_np})
    dt = time.perf_counter() - t0
    # outputs[0] = predictions, outputs[1] = embeddings
    return outputs[0], dt  # Return index 0 for predictions


def run_tflite(model_path: str, audio_np: np.ndarray) -> np.ndarray:
    """Run inference with a TFLite/LiteRT model.

    Creates a fresh interpreter, resizes input tensors to match ``audio_np``,
    and invokes the model.

    Returns:
        tuple[np.ndarray, float]: Predictions array and elapsed time in seconds.
    """
    interpreter = Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]['index'], list(audio_np.shape))
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], audio_np)
    t0 = time.perf_counter()
    interpreter.invoke()
    dt = time.perf_counter() - t0
    
    # Grab the tensor explicitly named 'predictions'
    for detail in output_details:
        if 'predictions' in detail['name']:
            return interpreter.get_tensor(detail['index']), dt
            
    # Fallback just in case
    return interpreter.get_tensor(output_details[0]['index']), dt


def run_tflite_partial(pre_path: str, main_path: str, audio_np: np.ndarray):
    """Chain preprocessing tflite + main (optimised) tflite."""
    # Preprocessing
    pre = Interpreter(model_path=pre_path)
    pre_in = pre.get_input_details()
    pre.resize_tensor_input(pre_in[0]['index'], list(audio_np.shape))
    pre.allocate_tensors()
    pre_in  = pre.get_input_details()
    pre_out = pre.get_output_details()
    pre.set_tensor(pre_in[0]['index'], audio_np)
    t0 = time.perf_counter()
    pre.invoke()
    spec = pre.get_tensor(pre_out[0]['index'])  # [B, H, W, C]

    # Transpose to [B, C, H, W] for main model
    spec = np.transpose(spec, (0, 3, 1, 2))

    # Main (optimised)
    main = Interpreter(model_path=main_path)
    main_in = main.get_input_details()
    main.resize_tensor_input(main_in[0]['index'], list(spec.shape))
    main.allocate_tensors()
    main_in  = main.get_input_details()
    main_out = main.get_output_details()
    main.set_tensor(main_in[0]['index'], spec)
    main.invoke()
    dt = time.perf_counter() - t0
    # Grab the tensor explicitly named 'predictions'
    for detail in main_out:
        if 'predictions' in detail['name']:
            return main.get_tensor(detail['index']), dt
            
    # Fallback
    return main.get_tensor(main_out[0]['index']), dt

# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_topk(out, k=TOP_K):
    """Format the top-k class indices and scores from the first batch element."""
    topk = np.argsort(out[0])[-k:][::-1]
    return " ".join(f"{i}({out[0][i]:.3f})" for i in topk)


def compare_and_print(name: str, ref_out, other_out, duration, batch_size, ref_dt, other_dt,
                      atol=ATOL):
    """Compare other_out against ref_out (TorchScript). Returns 'PASS'/'FAIL'."""
    max_d = np.max(np.abs(ref_out - other_out))
    mean_d = np.mean(np.abs(ref_out - other_out))
    passed = np.allclose(ref_out, other_out, atol=atol, rtol=RTOL)
    icon = "✅ PASS" if passed else "❌ FAIL"
    dt_diff = other_dt - ref_dt
    sign = "+" if dt_diff >= 0 else ""
    print(f"{icon}  {name:<10}  dur={duration:.1f}s  batch={batch_size}  "
          f"max_diff={max_d:.6f}  mean_diff={mean_d:.6f}  "
          f"time: {other_dt:.3f}s ({sign}{dt_diff:.3f}s vs TS)")
    print(f"  TS:   {fmt_topk(ref_out)}")
    print(f"  {name + ':':<5} {fmt_topk(other_out)}")
    return 'PASS' if passed else 'FAIL'


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    """Build the argparse parser for the test CLI."""
    parser = argparse.ArgumentParser(
        description='Compare converted model outputs against TorchScript reference.',
    )
    parser.add_argument(
        '-e', '--export-dir',
        default=DEFAULT_EXPORT_DIR,
        help='Path to export directory containing converted models (default: export/)',
    )
    parser.add_argument(
        '-a', '--audio',
        default=DEFAULT_AUDIO,
        help='Path to audio file for test input (default: example/soundscape.wav).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for test inputs (default: 1)',
    )
    parser.add_argument(
        '--durations',
        type=float,
        nargs='+',
        default=[3.0, 5.0],
        help='Segment durations in seconds to test (default: 3.0, 5.0)',
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for TorchScript and ONNX (LiteRT always runs on CPU)',
    )
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Entry point: parse CLI arguments, load models, run comparisons, and report results."""
    parser = _build_parser()
    args = parser.parse_args()

    export_dir = os.path.abspath(args.export_dir)
    audio_path = args.audio
    batch_sizes = [args.batch_size]
    durations = args.durations
    use_gpu = args.gpu and torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    # Derive model paths from export directory
    torchscript_path = os.path.join(export_dir, 'torchscript', 'model.pt')
    onnx_path = os.path.join(export_dir, 'onnx', 'model.onnx')
    tflite_path = os.path.join(export_dir, 'tf', 'combined', 'model.tflite')

    # FP16 / INT8 variants
    torchscript_fp16_path = os.path.join(export_dir, 'torchscript', 'model_fp16.pt')
    onnx_fp16_path = os.path.join(export_dir, 'onnx', 'model_fp16.onnx')
    onnx_int8_path = os.path.join(export_dir, 'onnx', 'model_int8.onnx')
    tflite_fp16_path = os.path.join(export_dir, 'tf', 'combined', 'model_fp16.tflite')
    tflite_int8_path = os.path.join(export_dir, 'tf', 'combined', 'model_int8.tflite')

    # tflite int8 partial (preprocessing float32 + main int8 optimised)
    tflite_pre_path = os.path.join(export_dir, 'tf', 'preprocessing', 'model.tflite')
    tflite_main_opt_path = os.path.join(export_dir, 'tf', 'main', 'model_int8.tflite')

    audio_label = audio_path if audio_path else 'random noise (seed=42)'
    print(f"Model comparison (reference: TorchScript)  |  atol={ATOL} (int8: {ATOL_INT8})  rtol={RTOL}  top-{TOP_K}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name()})" if use_gpu else ""))
    print(f"Audio: {audio_label}")
    print(f"Export dir: {export_dir}\n")

    # Check which models exist
    models = {}

    print("Loading models...", flush=True)

    if not os.path.exists(torchscript_path):
        sys.exit(f'Error: TorchScript reference not found at {torchscript_path}')
    print(f"  TorchScript: {torchscript_path}")
    ts_model = torch.jit.load(torchscript_path, map_location=device)
    ts_model.eval()
    models['TS'] = ts_model

    if os.path.exists(torchscript_fp16_path):
        print(f"  TS fp16:     {torchscript_fp16_path}")
        ts_fp16_model = torch.jit.load(torchscript_fp16_path, map_location=device)
        ts_fp16_model.eval()
        models['TS_fp16'] = ts_fp16_model
    else:
        print(f"  TS fp16:     NOT FOUND — skipping")

    if os.path.exists(onnx_path):
        onnx_providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                          if use_gpu else ['CPUExecutionProvider'])
        print(f"  ONNX:        {onnx_path}  (providers: {onnx_providers[0]})")
        onnx_session = ort.InferenceSession(onnx_path, providers=onnx_providers)
        models['ONNX'] = onnx_session
    else:
        print(f"  ONNX:        NOT FOUND — skipping")

    if os.path.exists(onnx_fp16_path):
        print(f"  ONNX fp16:   {onnx_fp16_path}")
        models['ONNX_fp16'] = ort.InferenceSession(
            onnx_fp16_path, providers=(['CUDAExecutionProvider', 'CPUExecutionProvider']
                                       if use_gpu else ['CPUExecutionProvider']))

    if os.path.exists(onnx_int8_path):
        print(f"  ONNX int8:   {onnx_int8_path}")
        models['ONNX_int8'] = ort.InferenceSession(
            onnx_int8_path, providers=['CPUExecutionProvider'])

    if os.path.exists(tflite_path):
        print(f"  LiteRT:      {tflite_path}")
        models['LiteRT'] = tflite_path
    else:
        print(f"  LiteRT:      NOT FOUND — skipping")

    if os.path.exists(tflite_fp16_path):
        print(f"  LiteRT fp16: {tflite_fp16_path}")
        models['LRT_fp16'] = tflite_fp16_path

    if os.path.exists(tflite_int8_path):
        print(f"  LiteRT int8: {tflite_int8_path}")
        models['LRT_int8'] = tflite_int8_path

    if os.path.exists(tflite_pre_path) and os.path.exists(tflite_main_opt_path):
        print(f"  LRT int8 partial: {tflite_pre_path} + {tflite_main_opt_path}")
        models['LRT_int8_partial'] = (tflite_pre_path, tflite_main_opt_path)
    else:
        print(f"  LRT int8 partial: NOT FOUND — skipping")

    print("done.\n")

    results = []

    for duration in durations:
        for batch_size in batch_sizes:
            audio = load_audio(audio_path, duration, batch_size)
            ts_out, ts_dt = run_torchscript(ts_model, audio, device)

            # TorchScript FP16
            if 'TS_fp16' in models:
                try:
                    out, dt = run_torchscript(models['TS_fp16'], audio, device)
                    results.append(compare_and_print(
                        'TS_fp16', ts_out, out, duration, batch_size, ts_dt, dt))
                except Exception as e:
                    print(f"⚠️ ERR   TS_fp16     dur={duration:.1f}s  batch={batch_size}  {str(e)[:80]}")
                    results.append('ERROR')

            # ONNX
            if 'ONNX' in models:
                try:
                    onnx_out, onnx_dt = run_onnx(models['ONNX'], audio)
                    results.append(compare_and_print(
                        'ONNX', ts_out, onnx_out, duration, batch_size, ts_dt, onnx_dt))
                except Exception as e:
                    print(f"⚠️ ERR   ONNX        dur={duration:.1f}s  batch={batch_size}  {str(e)[:80]}")
                    results.append('ERROR')

            # ONNX FP16
            if 'ONNX_fp16' in models:
                try:
                    out, dt = run_onnx(models['ONNX_fp16'], audio)
                    results.append(compare_and_print(
                        'ONNX_fp16', ts_out, out, duration, batch_size, ts_dt, dt))
                except Exception as e:
                    print(f"⚠️ ERR   ONNX_fp16   dur={duration:.1f}s  batch={batch_size}  {str(e)[:80]}")
                    results.append('ERROR')

            # ONNX INT8
            if 'ONNX_int8' in models:
                try:
                    out, dt = run_onnx(models['ONNX_int8'], audio)
                    results.append(compare_and_print(
                        'ONNX_int8', ts_out, out, duration, batch_size, ts_dt, dt,
                        atol=ATOL_INT8))
                except Exception as e:
                    print(f"⚠️ ERR   ONNX_int8   dur={duration:.1f}s  batch={batch_size}  {str(e)[:80]}")
                    results.append('ERROR')

            # LiteRT
            if 'LiteRT' in models:
                try:
                    tfl_out, tfl_dt = run_tflite(models['LiteRT'], audio)
                    results.append(compare_and_print(
                        'LiteRT', ts_out, tfl_out, duration, batch_size, ts_dt, tfl_dt))
                except RuntimeError as e:
                    print(f"⚠️ ERR   LiteRT      dur={duration:.1f}s  batch={batch_size}  {str(e).split(chr(10))[0][:80]}")
                    results.append('ERROR')

            # LiteRT FP16
            if 'LRT_fp16' in models:
                try:
                    out, dt = run_tflite(models['LRT_fp16'], audio)
                    results.append(compare_and_print(
                        'LRT_fp16', ts_out, out, duration, batch_size, ts_dt, dt))
                except RuntimeError as e:
                    print(f"⚠️ ERR   LRT_fp16    dur={duration:.1f}s  batch={batch_size}  {str(e).split(chr(10))[0][:80]}")
                    results.append('ERROR')

            # LRT int8 partially optimized (preprocessing + main optimized)
            if 'LRT_int8_partial' in models:
                try:
                    pre_path, main_opt_path = models['LRT_int8_partial']
                    part_out, part_dt = run_tflite_partial(pre_path, main_opt_path, audio)
                    results.append(compare_and_print(
                        'LRT_int8_partial', ts_out, part_out, duration, batch_size, ts_dt, part_dt, atol=ATOL_INT8))
                except RuntimeError as e:
                    print(f"⚠️ ERR   LRT_int8_partial     dur={duration:.1f}s  batch={batch_size}  {str(e).split(chr(10))[0][:80]}")
                    results.append('ERROR')

            # LiteRT INT8
            if 'LRT_int8' in models:
                try:
                    out, dt = run_tflite(models['LRT_int8'], audio)
                    results.append(compare_and_print(
                        'LRT_int8', ts_out, out, duration, batch_size, ts_dt, dt,
                        atol=ATOL_INT8))
                except RuntimeError as e:
                    print(f"⚠️ ERR   LRT_int8    dur={duration:.1f}s  batch={batch_size}  {str(e).split(chr(10))[0][:80]}")
                    results.append('ERROR')

            print()

    n_pass  = results.count('PASS')
    n_fail  = results.count('FAIL')
    n_error = results.count('ERROR')
    print(f"{n_pass} passed, {n_fail} failed, {n_error} errors  "
          f"(out of {len(results)} combos)")

    sys.exit(0 if n_fail == 0 and n_error == 0 else 1)


if __name__ == '__main__':
    main()
