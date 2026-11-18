"""Post-process a TFLite flatbuffer so the model accepts other input lengths.

The litert_torch converter traces the model at one input shape and bakes the time
dimension of that shape into the shape tensors of RESHAPE ops. This script patches those
tensors to -1, so the time dimension follows the input. The batch size stays fixed.
"""

import numpy as np
import struct


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _score_candidates(buf, candidates, pattern_len):
    """Score candidate offsets. Prefer metadata regions over weight blobs.

    Metadata regions (flatbuffer tables, vectors, offsets) tend to contain
    small integers. Weight blobs contain float32 values that look like
    large or negative int32s.
    """
    best_candidate = candidates[0]
    best_score = -1

    for cand in candidates:
        score = 0
        # Check 32 bytes before and after the pattern
        for region in [buf[max(0, cand - 32):cand],
                       buf[cand + pattern_len:min(len(buf), cand + pattern_len + 32)]]:
            for i in range(0, len(region) - 3, 4):
                val = struct.unpack('<i', region[i:i + 4])[0]
                # Metadata values are typically small non-negative integers
                # or small offsets
                if 0 <= val <= 100000:
                    score += 2
                elif -10 <= val < 0:
                    score += 1
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate


def _patch_int32_in_buffer(buf, val, patched_val, patched_offsets, label=""):
    """Find the byte sequence for `val` in `buf` and replace it with `patched_val`.
    Uses flatbuffer vector headers to ensure uniqueness."""
    
    target_bytes = np.array(val, dtype=np.int32).tobytes()
    patch_bytes = np.array(patched_val, dtype=np.int32).tobytes()
    byte_len = len(target_bytes)
    shape_len = len(val)

    # Strategy 1: search for vector-header pattern (more unique)
    vec_byte_count = shape_len * 4
    vec_target = np.array([vec_byte_count] + list(val), dtype=np.int32).tobytes()

    pos = 0
    while True:
        pos = buf.find(vec_target, pos)
        if pos == -1:
            break
        if pos % 4 == 0 and pos not in patched_offsets:
            patch_pos = pos + 4  # skip the length field
            buf[patch_pos : patch_pos + byte_len] = patch_bytes
            patched_offsets.add(pos)
            print(f"      ✅ Patched via vector header at offset {pos}")
            return True
        pos += 1

    # Strategy 2: search for raw shape bytes without vector header
    candidates = []
    pos = 0
    while True:
        pos = buf.find(target_bytes, pos)
        if pos == -1:
            break
        if pos % 4 == 0 and pos not in patched_offsets:
            candidates.append(pos)
        pos += 1

    if not candidates:
        print(f"      ❌ Could not find byte pattern in buffer")
        return False

    if len(candidates) == 1:
        cand = candidates[0]
        buf[cand : cand + byte_len] = patch_bytes
        patched_offsets.add(cand)
        print(f"      ✅ Patched (unique match) at offset {cand}")
        return True
    else:
        print(f"      Found {len(candidates)} candidates, scoring...")
        best = _score_candidates(buf, candidates, byte_len)
        buf[best : best + byte_len] = patch_bytes
        patched_offsets.add(best)
        print(f"      ✅ Patched (best score) at offset {best}")
        return True


# ─── RESHAPE fix ─────────────────────────────────────────────────────────────

def _find_all_reshape_shapes(interpreter, expected_t_dims):
    """Find all Reshape ops whose shape tensor starts with 1 (static batch)
    and whose input/output are batch-dependent."""

    tensor_details = {t['index']: t for t in interpreter.get_tensor_details()}

    try:
        op_details = interpreter._get_ops_details()
    except AttributeError:
        return []

    results = []
    for op in op_details:
        op_name = op.get('op_name', '')
        if op_name != 'RESHAPE':
            continue

        op_idx = op.get('index', '?')
        input_indices = op.get('inputs', [])
        output_indices = op.get('outputs', [])

        if len(input_indices) < 2 or not output_indices:
            continue

        # First input to Reshape is the tensor to reshape
        input_tensor_idx = input_indices[0]
        input_tensor = tensor_details.get(input_tensor_idx)
        if input_tensor is None:
            continue

        input_name = input_tensor['name']
        input_shape = input_tensor['shape']
        input_dtype = input_tensor['dtype']

        # Heuristic: skip if input tensor is constant or weight
        if (
            input_name.startswith('arith.constant')
            or input_name.startswith('Const')
            or input_dtype not in [np.float32, np.float16]
            or len(input_shape) < 2
            or (input_shape[0] <= 0 and input_shape[0] != -1)
        ):
            continue

        # Output tensor of the Reshape op
        output_tensor_idx = output_indices[0]
        output_tensor = tensor_details.get(output_tensor_idx)
        if output_tensor is None:
            continue
        output_shape = output_tensor['shape']
        if not (len(output_shape) > 0 and (output_shape[0] > 0 or output_shape[0] == -1)):
            continue

        # Second input to Reshape is the shape tensor
        shape_tensor_idx = input_indices[1]
        shape_tensor = tensor_details.get(shape_tensor_idx)

        if shape_tensor is None or shape_tensor['dtype'] != np.int32:
            continue

        try:
            shape_val = interpreter.get_tensor(shape_tensor_idx).flatten()
        except Exception:
            continue

        # Match shape tensor
        # We look for shapes that do NOT already have -1, and contain a know temporal dimension
        # Note: We now fix batch=1 and make time dynamic.
        # Temporal dimensions we expect (derived from 3.0s window): 7, 9, 14, 281
        if len(shape_val) >= 2 and -1 not in shape_val:
            for t_dim in expected_t_dims:
                if t_dim in shape_val:
                    patched_shape = list(shape_val)
                    # Replace the temporal dimension with -1
                    # Since temporal dim is usually the last one, search from the end to avoid replacing frequency bins like 14 or 7.
                    idx_to_replace = -1
                    for idx_t in range(len(shape_val)-1, 0, -1):
                        if shape_val[idx_t] == t_dim:
                            idx_to_replace = idx_t
                            break
                    if idx_to_replace != -1:
                        patched_shape[idx_to_replace] = -1
                    
                    results.append({
                        'op_idx': op_idx,
                        'tensor_idx': shape_tensor_idx,
                        'tensor_name': shape_tensor['name'],
                        'shape_val': list(shape_val),
                        'patched_val': patched_shape,
                    })
                    break

    return results


def _fix_reshape(buf, interpreter, patched_offsets, expected_t_dims):
    """Fix all RESHAPE ops with static batch. Returns number of fixes."""

    reshape_ops = _find_all_reshape_shapes(interpreter, expected_t_dims)

    if not reshape_ops:
        print("  RESHAPE: No ops with static batch found.")
        return 0

    print(f"  RESHAPE: Found {len(reshape_ops)} op(s) with static batch dim:")
    for r in reshape_ops:
        print(f"    Op #{r['op_idx']}: tensor [{r['tensor_idx']}] "
              f"'{r['tensor_name']}' = {r['shape_val']}")

    # Group by unique shape tensor index (multiple ops can share the same tensor)
    unique_tensors = {}
    for r in reshape_ops:
        tidx = r['tensor_idx']
        if tidx not in unique_tensors:
            unique_tensors[tidx] = r

    print(f"  {len(unique_tensors)} unique shape tensor(s) to patch:")

    n_fixed = 0
    for tidx, r in unique_tensors.items():
        print(f"\n    Tensor [{tidx}] '{r['tensor_name']}':")
        if _patch_int32_in_buffer(buf, r['shape_val'], r['patched_val'], patched_offsets, label=""):
            n_fixed += 1

    return n_fixed


# ─── SLICE fix ───────────────────────────────────────────────────────────────

# ─── Public API ──────────────────────────────────────────────────────────────

def fix_all_for_dynamic_batch(tflite_path, output_path=None, image_width=281, audio_length=96000, expected_t_dims=None):
    """Patch the RESHAPE shape tensors of a TFLite model; return the number patched."""
    import tensorflow as tf

    if expected_t_dims is None:
        expected_t_dims = [7, 9, 14, 281]
        
    if output_path is None:
        output_path = tflite_path

    with open(tflite_path, 'rb') as f:
        buf = bytearray(f.read())

    # Use BUILTIN_WITHOUT_DEFAULT_DELEGATES to inspect the raw graph
    # (not fused by XNNPACK)
    interpreter = tf.lite.Interpreter(
        model_content=bytes(buf),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )

    # Pre-allocate resize: TFLite's default dynamic shape (e.g. [1, 3, 224, 1]) can
    # shrink to 0 after strided ops (like EfficientNet downsampling by 32x).
    # This causes RESHAPE ops to fail (0 != expected). We resize to a valid
    # dummy shape (e.g. 281 for width) before allocating tensors.
    for details in interpreter.get_input_details():
        shape = list(details['shape'])
        sig = list(details.get('shape_signature', shape))
        new_shape = shape.copy()
        need_resize = False
        ndim = len(shape)
        
        # Batch dimension
        if sig[0] == -1 or shape[0] <= 1:
            new_shape[0] = 1
            need_resize = True
            
        # Other dimensions
        for i in range(1, ndim):
            if sig[i] == -1 or shape[i] <= 1:
                if ndim == 4 and i == 3:
                    new_shape[i] = image_width  # Dynamic dimension passed in
                elif ndim == 2 and i == 1:
                    new_shape[i] = audio_length # Dynamic dimension passed in
                else:
                    new_shape[i] = max(shape[i], 1)
                need_resize = True
                
        if need_resize:
            interpreter.resize_tensor_input(details['index'], new_shape)

    interpreter.allocate_tensors()

    patched_offsets = set()

    n_reshape = _fix_reshape(buf, interpreter, patched_offsets, expected_t_dims)

    # NOTE: SLICE and STRIDED_SLICE fixes are intentionally NOT applied here.
    # TFLite does not propagate -1 in SLICE size tensors to output tensor shapes
    # during allocate_tensors(), so patching size[0] to -1 breaks shape inference
    # for all downstream ops. The proper fix for SLICE-based batch issues is to
    # eliminate SLICE ops at the PyTorch level (e.g., use pool2d instead of
    # manual slicing in _pool1d_max/_pool1d_avg).

    total = n_reshape

    if total > 0:
        with open(output_path, 'wb') as f:
            f.write(buf)
        print(f"\n  Total: Fixed {n_reshape} RESHAPE tensor(s). Saved to {output_path}")
    else:
        print(f"\n  No static batch dims found to fix.")

    return total


def verify_fix(tflite_path, image_height=None, image_width=None, audio_length=96000):
    """Run the patched model once at the export shape; raise if it does not run.

    ``image_height``/``image_width`` (main model) and ``audio_length`` (preprocessing and
    combined models) fill in the dynamic dimensions.
    """
    import tensorflow as tf

    print(f"\n  Verifying {tflite_path}...")
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    input_details = interpreter.get_input_details()
    shape = list(input_details[0]['shape'])
    sig = list(input_details[0].get('shape_signature', shape))
    ndim = len(shape)
    for i in range(1, ndim):
        if sig[i] == -1 or shape[i] <= 0:
            if ndim == 2:
                shape[i] = audio_length
            elif ndim == 4 and image_height is not None and image_width is not None:
                shape[i] = [3, image_height, image_width][i - 1]
            else:
                shape[i] = 1
    shape[0] = max(int(shape[0]), 1)
    try:
        interpreter.resize_tensor_input(input_details[0]['index'], shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], np.random.rand(*shape).astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    except Exception as e:
        raise RuntimeError(f'{tflite_path} does not run at input shape {shape}: {e}') from e
    if not np.isfinite(output).all():
        raise RuntimeError(f'{tflite_path} returns non-finite values at input shape {shape}')
    print(f"    input {shape}: output shape={output.shape} ✅")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_tflite_dynamic_input.py <model.tflite> [output.tflite]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    fix_all_for_dynamic_batch(input_path, output_path)
