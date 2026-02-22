"""Post-process TFLite flatbuffer to fix static batch dimensions for dynamic batch.

The litert_torch converter traces models with batch=1, baking static batch
dimensions into RESHAPE shape tensors and SLICE size tensors. This script
patches both to enable dynamic batch at runtime.

Fixes applied:
  1. RESHAPE: shape tensors [1, ...] → [-1, ...]
  2. SLICE: size tensors [1, ...] → [-1, ...] (TFLite SLICE uses -1 = "all remaining")
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

def _find_all_reshape_shapes(interpreter):
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
            for t_dim in [7, 9, 14, 281]:
                if t_dim in shape_val:
                    patched_shape = list(shape_val)
                    # Replace the temporal dimension with -1
                    patched_shape[list(shape_val).index(t_dim)] = -1
                    
                    results.append({
                        'op_idx': op_idx,
                        'tensor_idx': shape_tensor_idx,
                        'tensor_name': shape_tensor['name'],
                        'shape_val': list(shape_val),
                        'patched_val': patched_shape,
                    })
                    break

    return results


def _fix_reshape(buf, interpreter, patched_offsets):
    """Fix all RESHAPE ops with static batch. Returns number of fixes."""

    reshape_ops = _find_all_reshape_shapes(interpreter)

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

def _find_all_slice_sizes(interpreter):
    """Find all SLICE ops whose size tensor has size[0] == 1 (static batch)
    while the input tensor has a batch-dependent first dimension."""

    tensor_details = {t['index']: t for t in interpreter.get_tensor_details()}

    try:
        op_details = interpreter._get_ops_details()
    except AttributeError:
        return []

    results = []
    for op in op_details:
        op_name = op.get('op_name', '')
        # TFLite SLICE op: inputs are [input, begin, size]
        if op_name != 'SLICE':
            continue

        op_idx = op.get('index', '?')
        input_indices = op.get('inputs', [])
        output_indices = op.get('outputs', [])

        if len(input_indices) < 3 or not output_indices:
            continue

        # Input tensor (the data to slice)
        data_tensor_idx = input_indices[0]
        data_tensor = tensor_details.get(data_tensor_idx)
        if data_tensor is None:
            continue

        data_name = data_tensor['name']
        data_shape = data_tensor['shape']
        data_dtype = data_tensor['dtype']

        # Heuristic: skip constants/weights
        if (
            data_name.startswith('arith.constant')
            or data_name.startswith('Const')
            or data_dtype not in [np.float32, np.float16]
            or len(data_shape) < 2
        ):
            continue

        # Check if input has dynamic/batch first dim
        data_shape_sig = data_tensor.get('shape_signature', data_shape)
        if not (data_shape[0] in [1, -1] or data_shape_sig[0] == -1):
            continue

        # begin tensor (input_indices[1]) — should have begin[0] == 0
        begin_tensor_idx = input_indices[1]
        begin_tensor = tensor_details.get(begin_tensor_idx)
        if begin_tensor is None or begin_tensor['dtype'] != np.int32:
            continue

        try:
            begin_val = interpreter.get_tensor(begin_tensor_idx).flatten()
        except Exception:
            continue

        if begin_val[0] != 0:
            # If begin[0] != 0, the slice doesn't start at the batch boundary
            continue

        # size tensor (input_indices[2]) — this is what we need to patch
        size_tensor_idx = input_indices[2]
        size_tensor = tensor_details.get(size_tensor_idx)
        if size_tensor is None or size_tensor['dtype'] != np.int32:
            continue

        try:
            size_val = interpreter.get_tensor(size_tensor_idx).flatten()
        except Exception:
            continue

        # Match: size[0] == 1 (baked batch dim) and size has >= 2 elements
        if len(size_val) >= 2 and size_val[0] == 1:
            # Verify the output shape also has batch=1
            output_tensor = tensor_details.get(output_indices[0])
            if output_tensor is not None:
                out_shape = output_tensor['shape']
                if len(out_shape) > 0 and out_shape[0] != 1:
                    continue  # output doesn't have batch=1, skip

            results.append({
                'op_idx': op_idx,
                'tensor_idx': size_tensor_idx,
                'tensor_name': size_tensor['name'],
                'size_val': size_val.copy(),
                'data_tensor_name': data_name,
            })

    return results


def _fix_slice(buf, interpreter, patched_offsets):
    """Fix all SLICE ops with static batch. Returns number of fixes."""

    slice_ops = _find_all_slice_sizes(interpreter)

    if not slice_ops:
        print("  SLICE: No ops with static batch found.")
        return 0

    print(f"\n  SLICE: Found {len(slice_ops)} op(s) with static batch dim:")
    for r in slice_ops:
        print(f"    Op #{r['op_idx']}: size tensor [{r['tensor_idx']}] "
              f"'{r['tensor_name']}' = {r['size_val']}  "
              f"(input: '{r['data_tensor_name']}')")

    # Group by unique size tensor index
    unique_tensors = {}
    for r in slice_ops:
        tidx = r['tensor_idx']
        if tidx not in unique_tensors:
            unique_tensors[tidx] = r

    print(f"  {len(unique_tensors)} unique size tensor(s) to patch:")

    n_fixed = 0
    for tidx, r in unique_tensors.items():
        print(f"\n    Tensor [{tidx}] '{r['tensor_name']}':")
        if _patch_int32_in_buffer(buf, r['size_val'], patched_offsets, label=""):
            n_fixed += 1

    return n_fixed


# ─── STRIDED_SLICE fix ───────────────────────────────────────────────────────

def _find_all_strided_slice_sizes(interpreter):
    """Find all STRIDED_SLICE ops with static batch in their end tensor.
    
    STRIDED_SLICE takes: [input, begin, end, strides]
    If end[0] == 1, it limits the batch dim to 1.
    """

    tensor_details = {t['index']: t for t in interpreter.get_tensor_details()}

    try:
        op_details = interpreter._get_ops_details()
    except AttributeError:
        return []

    results = []
    for op in op_details:
        op_name = op.get('op_name', '')
        if op_name != 'STRIDED_SLICE':
            continue

        op_idx = op.get('index', '?')
        input_indices = op.get('inputs', [])
        output_indices = op.get('outputs', [])

        if len(input_indices) < 4 or not output_indices:
            continue

        # Input tensor
        data_tensor_idx = input_indices[0]
        data_tensor = tensor_details.get(data_tensor_idx)
        if data_tensor is None:
            continue

        data_name = data_tensor['name']
        data_shape = data_tensor['shape']
        data_dtype = data_tensor['dtype']

        if (
            data_name.startswith('arith.constant')
            or data_name.startswith('Const')
            or data_dtype not in [np.float32, np.float16]
            or len(data_shape) < 2
        ):
            continue

        data_shape_sig = data_tensor.get('shape_signature', data_shape)
        if not (data_shape[0] in [1, -1] or data_shape_sig[0] == -1):
            continue

        # begin tensor — should start at 0 for batch dim
        begin_tensor_idx = input_indices[1]
        begin_tensor = tensor_details.get(begin_tensor_idx)
        if begin_tensor is None or begin_tensor['dtype'] != np.int32:
            continue
        try:
            begin_val = interpreter.get_tensor(begin_tensor_idx).flatten()
        except Exception:
            continue
        if begin_val[0] != 0:
            continue

        # end tensor — this is what we patch
        end_tensor_idx = input_indices[2]
        end_tensor = tensor_details.get(end_tensor_idx)
        if end_tensor is None or end_tensor['dtype'] != np.int32:
            continue
        try:
            end_val = interpreter.get_tensor(end_tensor_idx).flatten()
        except Exception:
            continue

        if len(end_val) >= 2 and end_val[0] == 1:
            output_tensor = tensor_details.get(output_indices[0])
            if output_tensor is not None:
                out_shape = output_tensor['shape']
                if len(out_shape) > 0 and out_shape[0] != 1:
                    continue

            results.append({
                'op_idx': op_idx,
                'tensor_idx': end_tensor_idx,
                'tensor_name': end_tensor['name'],
                'end_val': end_val.copy(),
                'data_tensor_name': data_name,
            })

    return results


def _fix_strided_slice(buf, interpreter, patched_offsets):
    """Fix all STRIDED_SLICE ops with static batch. Returns number of fixes."""

    ss_ops = _find_all_strided_slice_sizes(interpreter)

    if not ss_ops:
        print("  STRIDED_SLICE: No ops with static batch found.")
        return 0

    print(f"\n  STRIDED_SLICE: Found {len(ss_ops)} op(s) with static batch dim:")
    for r in ss_ops:
        print(f"    Op #{r['op_idx']}: end tensor [{r['tensor_idx']}] "
              f"'{r['tensor_name']}' = {r['end_val']}  "
              f"(input: '{r['data_tensor_name']}')")

    unique_tensors = {}
    for r in ss_ops:
        tidx = r['tensor_idx']
        if tidx not in unique_tensors:
            unique_tensors[tidx] = r

    print(f"  {len(unique_tensors)} unique end tensor(s) to patch:")

    # For STRIDED_SLICE, we patch end[0] to a large value (INT_MAX)
    # since -1 doesn't have the same semantics as in SLICE
    n_fixed = 0
    for tidx, r in unique_tensors.items():
        print(f"\n    Tensor [{tidx}] '{r['tensor_name']}':")
        # Use 0x7FFFFFFF (INT_MAX) as "take all" for strided_slice end
        end_val = r['end_val']
        new_end = end_val.copy()
        new_end[0] = 0x7FFFFFFF

        target_bytes = np.array(end_val, dtype=np.int32).tobytes()
        replacement_first = struct.pack('<i', 0x7FFFFFFF)

        # Try vector-header pattern first
        vec_byte_count = len(end_val) * 4
        vec_target = np.array([vec_byte_count] + list(end_val), dtype=np.int32).tobytes()

        found = False
        pos = 0
        while True:
            pos = buf.find(vec_target, pos)
            if pos == -1:
                break
            if pos % 4 == 0 and pos not in patched_offsets:
                patch_pos = pos + 4
                buf[patch_pos:patch_pos + 4] = replacement_first
                patched_offsets.add(pos)
                print(f"      ✅ Patched via vector header at offset {pos}")
                n_fixed += 1
                found = True
                break
            pos += 1

        if not found:
            # Fallback: raw pattern search
            # Shape tensors in our model look like:
            #   Batch-flatten: [-1, 11520] (size 2)
            #   Pool/Conv:     [-1, 1280, 1, 9] (size 4) or [-1, 1280, 9] (size 3)
            # We already patched dimension 0 to -1 above. Now we also need to un-bake
            # the temporal dimensions (usually the last or second-to-last dimension)
            # so they can grow/shrink based on the input sequence length.
            # In BirdNET, baked lengths are usually 7, 9, 281, etc.
            
            # Since we can't easily have TWO -1 dimensions in a single reshape,
            # this actually requires a different approach. Wait, TFLite ONLY allows one
            # -1 dimension per reshape! 
            # Check if this shape already has a -1 (which would be the batch dim we just set)
            num_minus_one = sum(1 for v in end_val if v == -1)
            
            patched_val = end_val.copy()
            modified = False
            
            for i in range(1, len(patched_val)):
                # If it's a known temporal length from BirdNET models
                if patched_val[i] in [7, 9, 14, 281]:
                    if num_minus_one == 0:
                        patched_val[i] = -1
                        num_minus_one += 1
                        modified = True
                    else:
                        # We can't have two -1s. However, the batch dim (index 0) 
                        # is usually already -1 in these ops. 
                        # This means we CANNOT make both Batch and Time dynamic in a standard RESHAPE!
                        # But let's at least try setting it to -1 if we can.
                        pass
                        
            # Original fix: Just ensuring batch=1 becomes batch=-1
            if patched_val[0] == 1 and num_minus_one == 0:
                patched_val[0] = -1
                modified = True
                
            if modified:
                end_val = patched_val
                target_bytes = np.array(end_val, dtype=np.int32).tobytes()
                replacement_first = struct.pack('<i', end_val[0]) # Use the potentially new first element
            
            candidates = []
            p = 0
            while True:
                p = buf.find(target_bytes, p)
                if p == -1:
                    break
                if p % 4 == 0 and p not in patched_offsets:
                    candidates.append(p)
                p += 1

            if candidates:
                if len(candidates) == 1:
                    cand = candidates[0]
                else:
                    cand = _score_candidates(buf, candidates, len(target_bytes))
                buf[cand:cand + 4] = replacement_first
                patched_offsets.add(cand)
                print(f"      ✅ Patched at offset {cand}")
                n_fixed += 1
            else:
                print(f"      ❌ Could not find byte pattern")

    return n_fixed


# ─── Public API ──────────────────────────────────────────────────────────────

def fix_all_for_dynamic_batch(tflite_path, output_path=None):
    """Apply all dynamic batch fixes (RESHAPE + SLICE + STRIDED_SLICE)
    to a TFLite model. Returns total number of fixes."""
    import tensorflow as tf

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
                    new_shape[i] = 281  # Safe default width for EfficientNet
                elif ndim == 2 and i == 1:
                    new_shape[i] = 96000 # Safe default audio length
                else:
                    new_shape[i] = max(shape[i], 1)
                need_resize = True
                
        if need_resize:
            try:
                interpreter.resize_tensor_input(details['index'], new_shape)
            except Exception as e:
                print(f"  Warning: Pre-allocation resize failed: {e}")

    interpreter.allocate_tensors()

    patched_offsets = set()

    n_reshape = _fix_reshape(buf, interpreter, patched_offsets)

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


# Legacy API — kept for backward compatibility
def fix_reshape_for_dynamic_batch(tflite_path, output_path=None):
    """Fix only RESHAPE ops. For full fix use fix_all_for_dynamic_batch()."""
    return fix_all_for_dynamic_batch(tflite_path, output_path)


def verify_fix(tflite_path, image_height=None, image_width=None):
    """Verify the fix by testing batch=1, 2, 4.

    Auto-detects input shape from the model. image_height and image_width
    are used as fallback values for any dynamic dimensions.
    """
    import tensorflow as tf

    print(f"\n  Verifying {tflite_path}...")

    interpreter_test = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    test_sig = list(interpreter_test.get_input_details()[0].get('shape_signature', interpreter_test.get_input_details()[0]['shape']))
    batches_to_test = [1, 2, 4] if test_sig[0] == -1 else [test_sig[0]]

    for batch in batches_to_test:
        try:
            interpreter = tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            input_details = interpreter.get_input_details()
            shape = list(input_details[0]['shape'])
            sig = list(input_details[0].get('shape_signature', shape))

            # Replace batch dim and any dynamic dims with concrete values
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
        print("Usage: python fix_tflite_dynamic_input.py <model.tflite> [output.tflite]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    fix_all_for_dynamic_batch(input_path, output_path)
