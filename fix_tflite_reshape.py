"""Post-process TFLite flatbuffer to fix static Reshape nodes for dynamic batch.

The litert_torch converter bakes Reshape shapes with batch=1 when tracing.
This script finds all Reshape ops whose shape tensor starts with 1 and
patches that 1 to -1, enabling dynamic batch at runtime.

Handles shape tensors of any length: [1, N], [1, 1, 1, N], [1, N, M], etc.
"""

import numpy as np
import struct


def _find_all_reshape_shapes(interpreter):
    """Find all Reshape ops whose shape tensor starts with 1 (static batch) and whose input/output are batch-dependent."""

    import numpy as np

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
            or input_shape[0] not in [1, -1]
        ):
            continue

        # Output tensor of the Reshape op
        output_tensor_idx = output_indices[0]
        output_tensor = tensor_details.get(output_tensor_idx)
        if output_tensor is None:
            continue
        output_shape = output_tensor['shape']
        # Only patch if output tensor has batch as first dim
        if not (len(output_shape) > 0 and output_shape[0] in [1, -1]):
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

        # Match any shape tensor that starts with 1 (baked batch dim)
        if len(shape_val) >= 2 and shape_val[0] == 1:
            results.append({
                'op_idx': op_idx,
                'tensor_idx': shape_tensor_idx,
                'tensor_name': shape_tensor['name'],
                'shape_val': shape_val.copy(),
            })

    return results


def fix_reshape_for_dynamic_batch(tflite_path, output_path=None):
    """
    Find all Reshape ops whose shape starts with 1 and patch to -1.
    Handles shapes of any dimensionality.
    """
    import tensorflow as tf

    if output_path is None:
        output_path = tflite_path

    with open(tflite_path, 'rb') as f:
        buf = bytearray(f.read())

    # Load with interpreter to inspect ops and tensors
    interpreter = tf.lite.Interpreter(
        model_content=bytes(buf),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interpreter.allocate_tensors()

    reshape_ops = _find_all_reshape_shapes(interpreter)

    if not reshape_ops:
        print("  No Reshape ops with static batch found.")
        return 0

    print(f"  Found {len(reshape_ops)} Reshape op(s) with static batch dim:")
    for r in reshape_ops:
        print(f"    Op #{r['op_idx']} RESHAPE: tensor [{r['tensor_idx']}] "
              f"'{r['tensor_name']}' = {r['shape_val']}")

    # Group by unique shape tensor index — multiple ops can share the same
    # shape tensor (e.g., tensor 378 'arith.constant27' is used by 14 ops)
    unique_tensors = {}
    for r in reshape_ops:
        tidx = r['tensor_idx']
        if tidx not in unique_tensors:
            unique_tensors[tidx] = r

    print(f"\n  {len(unique_tensors)} unique shape tensor(s) to patch:")

    n_fixed = 0
    patched_offsets = set()

    for tidx, r in unique_tensors.items():
        shape_val = r['shape_val']
        shape_len = len(shape_val)

        # Build the full byte pattern for the shape: [1, x, y, ...] as int32 LE
        target_bytes = np.array(shape_val, dtype=np.int32).tobytes()

        # Replacement: [-1, x, y, ...]
        new_shape = shape_val.copy()
        new_shape[0] = -1
        replacement_bytes = np.array(new_shape, dtype=np.int32).tobytes()

        byte_len = len(target_bytes)

        print(f"\n    Tensor [{tidx}] '{r['tensor_name']}': "
              f"{list(shape_val)} -> {list(new_shape)}")

        # Also build the vector-header version: flatbuffers vectors are
        # stored as [length (int32)] [elem0] [elem1] ...
        # Buffer.data in TFLite is [ubyte], so the length prefix is the
        # BYTE count (shape_len * 4), NOT the element count (shape_len).
        # Using shape_len would match Tensor.shape [int] vectors instead,
        # which corrupts tensor metadata rather than patching buffer data.
        vec_byte_count = shape_len * 4  # 3 int32s = 12 bytes
        vec_target = np.array([vec_byte_count] + list(shape_val), dtype=np.int32).tobytes()
        vec_replacement = np.array([vec_byte_count] + list(new_shape), dtype=np.int32).tobytes()

        # Strategy 1: Search for the vector-header pattern first (more unique)
        found = False
        pos = 0
        while True:
            pos = buf.find(vec_target, pos)
            if pos == -1:
                break
            if pos % 4 == 0 and pos not in patched_offsets:
                # Patch only the first element after the length
                patch_pos = pos + 4  # skip the length field
                buf[patch_pos:patch_pos + 4] = struct.pack('<i', -1)
                patched_offsets.add(pos)
                print(f"      ✅ Patched via vector header at offset {pos} "
                      f"(element at {patch_pos})")
                n_fixed += 1
                found = True
                break
            pos += 1

        if found:
            continue

        # Strategy 2: Search for the raw shape bytes without vector header
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
            print(f"      ❌ Could not find byte pattern {list(shape_val)} in buffer")
            continue

        if len(candidates) == 1:
            cand = candidates[0]
            buf[cand:cand + 4] = struct.pack('<i', -1)
            patched_offsets.add(cand)
            print(f"      ✅ Patched (unique match) at offset {cand}")
            n_fixed += 1
        else:
            # Multiple candidates — use heuristic scoring
            print(f"      Found {len(candidates)} candidates, scoring...")
            best = _score_candidates(buf, candidates, byte_len)
            buf[best:best + 4] = struct.pack('<i', -1)
            patched_offsets.add(best)
            print(f"      ✅ Patched (best score) at offset {best}")
            n_fixed += 1

    if n_fixed > 0:
        with open(output_path, 'wb') as f:
            f.write(buf)
        print(f"\n  Total: Fixed {n_fixed} unique shape tensor(s) "
              f"(affecting {len(reshape_ops)} Reshape ops). "
              f"Saved to {output_path}")
    else:
        print(f"\n  No Reshape shapes were fixed.")

    return n_fixed


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


def verify_fix(tflite_path, image_height=None, image_width=None):
    """Verify the fix by testing batch=1, 2, 4.
    
    Auto-detects input shape from the model. image_height and image_width
    are used as fallback values for any dynamic dimensions.
    """
    import tensorflow as tf

    print(f"\n  Verifying {tflite_path}...")

    for batch in [1, 2, 4]:
        try:
            interpreter = tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            input_details = interpreter.get_input_details()
            shape = list(input_details[0]['shape'])
            sig = list(input_details[0].get('shape_signature', shape))

            # Replace batch dim and any dynamic dims (-1 in signature) with concrete values
            shape[0] = batch
            ndim = len(shape)
            for i in range(1, ndim):
                if sig[i] == -1 or shape[i] <= 0:
                    if ndim == 2:
                        shape[i] = 96000  # audio samples (3s * 32000)
                    elif ndim == 4 and image_height is not None and image_width is not None:
                        shape[i] = [3, image_height, image_width][i - 1]
                    else:
                        shape[i] = 1

            interpreter.resize_tensor_input(input_details[0]['index'], shape)
            interpreter.allocate_tensors()

            test_data = np.random.rand(*shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_data)
            interpreter.invoke()

            output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            print(f"    batch={batch}: output shape={output.shape} ✅")
        except Exception as e:
            print(f"    batch={batch}: ❌ {e}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_tflite_reshape.py <model.tflite> [output.tflite]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    fix_reshape_for_dynamic_batch(input_path, output_path)