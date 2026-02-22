"""
Compare TorchScript (reference) against ONNX, LiteRT, and LiteRT optimized
model outputs across different batch sizes and segment durations.

All models take raw audio [batch, samples].
- TorchScript  → (embeddings, predictions)
- ONNX         → (embeddings, predictions)
- LiteRT       → predictions   (combined preprocessing + main)
- LiteRT opt.  → predictions   (combined preprocessing + main, quantised)
- LiteRT part  → predictions   (preprocessing tflite + main optimised tflite)
"""

import os
import sys
import time
import numpy as np
import torch
import librosa
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from convert_model import batchSize as tflite_batch_size

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models', '01_BirdNET+_V3.0-preview3_Global_11K'
)
TORCHSCRIPT_PATH = os.path.join(MODEL_DIR, 'torchscript', 'model.pt')
ONNX_PATH        = os.path.join(MODEL_DIR, 'onnx', 'model.onnx')          # Set to None to skip
TFLITE_PATH      = os.path.join(MODEL_DIR, 'tf', 'combined', 'model.tflite')  # Set to None to skip
TFLITE_OPT_PATH  = os.path.join(MODEL_DIR, 'tf', 'combined', 'model_optimized.tflite')  # Set to None to skip

# Partly optimised: unoptimised preprocessing + optimised main (set both to None to skip)
TFLITE_PRE_PATH      = os.path.join(MODEL_DIR, 'tf', 'preprocessing', 'model.tflite')  # Set to None to skip
TFLITE_MAIN_OPT_PATH = os.path.join(MODEL_DIR, 'tf', 'main', 'model_optimized.tflite')  # Set to None to skip

AUDIO_PATH = '/home/malk/projects/241017-1-BirdNetPlus/10_1_Inferencing/05_1_BirdNetPlusInference/birdnetplus-inference/example/LusMeg00027.mp3'

SAMPLE_RATE = 32000

# ── Test configurations ──────────────────────────────────────────────────────
BATCH_SIZES       = [tflite_batch_size]
SEGMENT_DURATIONS = [3.0, 5.0, 10.0]

# Tolerance for "match" (absolute + relative)
ATOL = 0.025
RTOL = 1e-4
TOP_K = 3

# Device: set to True to run TorchScript and ONNX on GPU (LiteRT stays on CPU)
USE_GPU = False #torch.cuda.is_available()


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio(duration: float, batch_size: int) -> np.ndarray:
    samples = int(duration * SAMPLE_RATE)
    audio, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, duration=duration,
                             offset=0.0, dtype=np.float32)
    assert sr == SAMPLE_RATE
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    else:
        audio = audio[:samples]
    return np.tile(audio[np.newaxis, :], (batch_size, 1))


# ── Model runners ────────────────────────────────────────────────────────────

def run_torchscript(model, audio_np: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        audio_t = torch.from_numpy(audio_np).to(device)
        t0 = time.perf_counter()
        embeddings, predictions = model(audio_t)
        if device != 'cpu':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return predictions.cpu().numpy(), dt


def run_onnx(session, audio_np: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: audio_np})
    dt = time.perf_counter() - t0
    # outputs[0] = embeddings, outputs[1] = predictions
    return outputs[1], dt


def run_tflite(model_path: str, audio_np: np.ndarray) -> np.ndarray:
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
    return main.get_tensor(main_out[0]['index']), dt


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_topk(out, k=TOP_K):
    topk = np.argsort(out[0])[-k:][::-1]
    return " ".join(f"{i}({out[0][i]:.3f})" for i in topk)


def compare_and_print(name: str, ref_out, other_out, duration, batch_size, ref_dt, other_dt):
    """Compare other_out against ref_out (TorchScript). Returns 'PASS'/'FAIL'."""
    max_d = np.max(np.abs(ref_out - other_out))
    mean_d = np.mean(np.abs(ref_out - other_out))
    passed = np.allclose(ref_out, other_out, atol=ATOL, rtol=RTOL)
    icon = "✅ PASS" if passed else "❌ FAIL"
    dt_diff = other_dt - ref_dt
    sign = "+" if dt_diff >= 0 else ""
    print(f"{icon}  {name:<10}  dur={duration:.1f}s  batch={batch_size}  "
          f"max_diff={max_d:.6f}  mean_diff={mean_d:.6f}  "
          f"time: {other_dt:.3f}s ({sign}{dt_diff:.3f}s vs TS)")
    print(f"  TS:   {fmt_topk(ref_out)}")
    print(f"  {name + ':':<5} {fmt_topk(other_out)}")
    return 'PASS' if passed else 'FAIL'


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if USE_GPU else 'cpu'
    print(f"Model comparison (reference: TorchScript)  |  atol={ATOL}  rtol={RTOL}  top-{TOP_K}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name()})" if USE_GPU else ""))
    print(f"Audio: {AUDIO_PATH}\n")

    # Check which models exist
    models = {}

    print("Loading models...", flush=True)

    print(f"  TorchScript: {TORCHSCRIPT_PATH}")
    ts_model = torch.jit.load(TORCHSCRIPT_PATH, map_location=device)
    ts_model.eval()
    models['TS'] = ts_model

    if ONNX_PATH and os.path.exists(ONNX_PATH):
        onnx_providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                          if USE_GPU else ['CPUExecutionProvider'])
        print(f"  ONNX:        {ONNX_PATH}  (providers: {onnx_providers[0]})")
        onnx_session = ort.InferenceSession(ONNX_PATH, providers=onnx_providers)
        models['ONNX'] = onnx_session
    else:
        print(f"  ONNX:        {'SKIPPED' if not ONNX_PATH else 'NOT FOUND'}")

    if TFLITE_PATH and os.path.exists(TFLITE_PATH):
        print(f"  LiteRT:      {TFLITE_PATH}")
        models['LiteRT'] = TFLITE_PATH
    else:
        print(f"  LiteRT:      {'SKIPPED' if not TFLITE_PATH else 'NOT FOUND'}")

    if (TFLITE_PRE_PATH and TFLITE_MAIN_OPT_PATH
            and os.path.exists(TFLITE_PRE_PATH) and os.path.exists(TFLITE_MAIN_OPT_PATH)):
        print(f"  LiteRT part: {TFLITE_PRE_PATH}")
        print(f"             + {TFLITE_MAIN_OPT_PATH}")
        models['LRTpart'] = (TFLITE_PRE_PATH, TFLITE_MAIN_OPT_PATH)
    else:
        skip = not TFLITE_PRE_PATH or not TFLITE_MAIN_OPT_PATH
        print(f"  LiteRT part: {'SKIPPED' if skip else 'NOT FOUND'}")

    if TFLITE_OPT_PATH and os.path.exists(TFLITE_OPT_PATH):
        print(f"  LiteRT opt:  {TFLITE_OPT_PATH}")
        models['LiteRTopt'] = TFLITE_OPT_PATH
    else:
        print(f"  LiteRT opt:  {'SKIPPED' if not TFLITE_OPT_PATH else 'NOT FOUND'}")

    print("done.\n")

    results = []

    for duration in SEGMENT_DURATIONS:
        for batch_size in BATCH_SIZES:
            audio = load_audio(duration, batch_size)
            ts_out, ts_dt = run_torchscript(ts_model, audio, device)

            # ONNX
            if 'ONNX' in models:
                try:
                    onnx_out, onnx_dt = run_onnx(models['ONNX'], audio)
                    results.append(compare_and_print(
                        'ONNX', ts_out, onnx_out, duration, batch_size, ts_dt, onnx_dt))
                except Exception as e:
                    print(f"⚠️ ERR   ONNX        dur={duration:.1f}s  batch={batch_size}  {str(e)[:80]}")
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

            # LiteRT partly optimized (preprocessing + main optimized)
            if 'LRTpart' in models:
                try:
                    pre_path, main_opt_path = models['LRTpart']
                    part_out, part_dt = run_tflite_partial(pre_path, main_opt_path, audio)
                    results.append(compare_and_print(
                        'LRTpart', ts_out, part_out, duration, batch_size, ts_dt, part_dt))
                except RuntimeError as e:
                    print(f"⚠️ ERR   LRTpart     dur={duration:.1f}s  batch={batch_size}  {str(e).split(chr(10))[0][:80]}")
                    results.append('ERROR')

            # LiteRT optimized
            if 'LiteRTopt' in models:
                try:
                    tfl_opt_out, tfl_opt_dt = run_tflite(models['LiteRTopt'], audio)
                    results.append(compare_and_print(
                        'LiteRTopt', ts_out, tfl_opt_out, duration, batch_size, ts_dt, tfl_opt_dt))
                except RuntimeError as e:
                    print(f"⚠️ ERR   LiteRTopt   dur={duration:.1f}s  batch={batch_size}  {str(e).split(chr(10))[0][:80]}")
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
