"""
Compare the converted BirdNET+ Geomodel files against the TorchScript model.

Every model takes float32 [batch, 3] = latitude (degrees), longitude (degrees), week (1-48)
and returns one value per class. The TorchScript model is the reference; it returns raw
logits unless it was exported with --torchscript-sigmoid, and this script applies the
sigmoid itself when it sees values outside [0, 1].

Checked for every file found in the export directory:
- the output shape, and that no value is NaN or infinite
- the largest difference in probability to the reference, against a tolerance
- how often the top class agrees with the reference
- ONNX: that batch sizes 1, 7 and all inputs at once give the same result
- LiteRT: that the model runs without TensorFlow (no Flex delegate)
- checkpoint/model.pt: that it loads with the model definition and gives the same output
- Core ML: run on macOS only, the one system that runs Core ML; skipped elsewhere
- labels.csv / labels.txt: one row per model output, unique birdnet_id
- labels against the model: a few well-known species must be likely where they live and
  unlikely on another continent, which fails if labels and outputs are out of step

The script fails if it finds no converted model to compare.

Usage:
    python test_geomodel.py
    python test_geomodel.py -e export/03_BirdNET+_Geomodel_V3.0.5_Global_14K --speed-runs 20
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'export', '03_BirdNET+_Geomodel_V3.0.5_Global_14K')

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'models', '03_BirdNET+_Geomodel_V3.0.5_Global_14K')

# Largest allowed difference in probability to the TorchScript reference. Measured on
# 3000 random inputs: FP32 below 0.00004, FP16 0.005, INT8 0.2.
ATOL = {'fp32': 1e-4, 'fp16': 0.01, 'int8': 0.3,
        # Core ML runs on other hardware, and its FP16 model really computes in float16
        'coreml_fp32': 1e-4, 'coreml_fp16': 0.03}
# Smallest allowed share of inputs whose top class matches the reference. Near ties flip
# with any rounding difference; measured 100% (FP32), 99.6% (FP16), 90% (INT8).
MIN_TOP1 = {'fp32': 0.99, 'fp16': 0.98, 'int8': 0.8, 'coreml_fp32': 0.99, 'coreml_fp16': 0.97}

# (scientific name, latitude, longitude, week, expected to occur) — checked when the labels
# contain the species. Occurring means a probability above 0.5, not occurring below 0.1.
ANCHORS = [
    ('Turdus merula', 50.83, 12.92, 20, True),            # Common Blackbird, Germany, May
    ('Turdus merula', 42.45, -76.47, 22, False),          # ... not in New York
    ('Cardinalis cardinalis', 42.45, -76.47, 22, True),   # Northern Cardinal, New York, June
    ('Cardinalis cardinalis', 50.83, 12.92, 20, False),   # ... not in Germany
    ('Dacelo novaeguineae', -33.87, 151.21, 44, True),    # Laughing Kookaburra, Sydney
    ('Dacelo novaeguineae', 50.83, 12.92, 20, False),     # ... not in Germany
]


def test_inputs(n=256):
    """Return ``(n, 3)`` float32 inputs.

    Real places, the poles, the date line, every week of the year and coordinates at round
    fractions of 180 degrees, then random ones. Weeks and round coordinates give angles at
    exact fractions of pi, where Core ML's sin was found to return the wrong sign.
    """
    fixed = np.array([
        [50.83, 12.92, 20], [42.45, -76.47, 22], [-33.87, 151.21, 44], [0.0, 0.0, 1],
        [-1.29, 36.82, 10], [64.13, -21.90, 26], [90.0, 0.0, 48], [-90.0, 0.0, 1],
        [35.68, 180.0, 12], [35.68, -180.0, 12], [-54.80, -68.30, 48], [1.35, 103.82, 30],
    ] + [[50.83, 12.92, week] for week in range(1, 49)]
      + [[lat, lon, 20] for lat in (45.0, 22.5, -45.0, 30.0, 7.5)
         for lon in (45.0, -135.0, 22.5, 90.0, 7.5)], dtype=np.float32)
    rng = np.random.RandomState(7)  # not the seed convert_geomodel.py validates with
    random = np.stack([rng.uniform(-90, 90, n), rng.uniform(-180, 180, n),
                       rng.randint(1, 49, n)], axis=1).astype(np.float32)
    return np.concatenate([fixed, random])[:n]


def load_torchscript(path):
    """Return ``run(inputs) -> probabilities`` for the TorchScript reference."""
    model = torch.jit.load(path, map_location='cpu').eval()

    def run(inputs):
        with torch.no_grad():
            output = model(torch.from_numpy(inputs))
            if output.min() < 0.0 or output.max() > 1.0:  # raw logits
                output = torch.sigmoid(output)
        return output.numpy()
    return run


def load_onnx(path):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name
    return lambda inputs: session.run(None, {name: inputs})[0]


def load_tflite(path):
    """Return a runner that feeds the fixed-batch LiteRT model one model batch at a time."""
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=path, num_threads=os.cpu_count())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    batch = int(input_details['shape'][0])

    def run(inputs):
        outputs = []
        for start in range(0, len(inputs), batch):
            chunk = inputs[start:start + batch]
            n = len(chunk)
            if n < batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch - n, axis=0)])
            interpreter.set_tensor(input_details['index'], chunk)
            interpreter.invoke()
            outputs.append(interpreter.get_tensor(output_details['index'])[:n].copy())
        return np.concatenate(outputs)
    run.batch = batch
    return run


def load_coreml(path):
    """Return a runner for a fixed-batch Core ML model (macOS only)."""
    import coremltools as ct
    model = ct.models.MLModel(path)
    spec = model.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    batch = int(spec.description.input[0].type.multiArrayType.shape[0])

    def run(inputs):
        outputs = []
        for start in range(0, len(inputs), batch):
            chunk = inputs[start:start + batch]
            n = len(chunk)
            if n < batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch - n, axis=0)])
            outputs.append(np.asarray(model.predict({input_name: chunk})[output_name])[:n])
        return np.concatenate(outputs)
    run.batch = batch
    return run


def load_saved_model(path):
    """Return a runner for the fixed-batch SavedModel, or None if TensorFlow is missing."""
    try:
        import tensorflow as tf
    except ImportError:
        return None
    serving = tf.saved_model.load(path).signatures['serving_default']
    key = list(serving.structured_input_signature[1].keys())[0]
    batch = int(serving.structured_input_signature[1][key].shape[0])

    def run(inputs):
        outputs = []
        with tf.device('/CPU:0'):
            for start in range(0, len(inputs), batch):
                chunk = inputs[start:start + batch]
                n = len(chunk)
                if n < batch:
                    chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch - n, axis=0)])
                outputs.append(list(serving(**{key: tf.constant(chunk)}).values())[0].numpy()[:n])
        return np.concatenate(outputs)
    run.batch = batch
    return run


def load_checkpoint_runner(path, model_dir):
    """Return a runner for the exported PyTorch checkpoint, built with model_dir/model.py."""
    spec = importlib.util.spec_from_file_location('geomodel_module', os.path.join(model_dir, 'model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, _ = module.load_checkpoint(path)
    wrapper = module.ExportWrapper(model).eval()

    def run(inputs):
        with torch.no_grad():
            return wrapper(torch.from_numpy(inputs)).numpy()
    return run


def read_label_names(export_dir):
    """Return the scientific name of every output from labels.csv or labels.txt, or None."""
    csv_path = os.path.join(export_dir, 'labels.csv')
    txt_path = os.path.join(export_dir, 'labels.txt')
    if os.path.isfile(csv_path):
        import pandas as pd
        return list(pd.read_csv(csv_path, dtype=str, keep_default_na=False)['scientific_name'])
    if os.path.isfile(txt_path):
        with open(txt_path, encoding='utf-8') as f:
            rows = [line.rstrip('\n').split('\t') for line in f if line.strip()]
        if rows and all(len(row) > 1 for row in rows):
            return [row[1] for row in rows]
    return None


def check_anchors(export_dir, run):
    """Check well-known species at known places; return ``(checked, errors)``."""
    names = read_label_names(export_dir)
    if names is None:
        return 0, []
    column = {name: i for i, name in enumerate(names)}
    anchors = [a for a in ANCHORS if a[0] in column]
    if not anchors:
        return 0, []
    output = run(np.array([a[1:4] for a in anchors], dtype=np.float32))
    errors = []
    for row, (name, lat, lon, week, occurs) in enumerate(anchors):
        if row >= len(output) or column[name] >= output.shape[1]:
            errors.append(f'{name} is label {column[name]}, but the model has {output.shape[1]} outputs')
            continue
        p = float(output[row, column[name]])
        if (p <= 0.5) if occurs else (p >= 0.1):
            errors.append(f'{name} at ({lat}, {lon}), week {week}: probability {p:.3f}, '
                          f'expected {"above 0.5" if occurs else "below 0.1"}')
    return len(anchors), errors


def compare(name, output, reference, precision):
    """Print one result line and return True if ``output`` is close enough to ``reference``."""
    output = np.asarray(output, dtype=np.float32)
    if output.shape != reference.shape:
        print(f'❌ FAIL  {name:14s} output shape {output.shape}, expected {reference.shape}')
        return False
    if not np.isfinite(output).all():
        print(f'❌ FAIL  {name:14s} output has NaN or infinite values')
        return False
    max_diff = float(np.abs(output - reference).max())
    mean_diff = float(np.abs(output - reference).mean())
    top1 = float((output.argmax(1) == reference.argmax(1)).mean())
    passed = max_diff <= ATOL[precision] and top1 >= MIN_TOP1[precision]
    print(f'{"✅ PASS" if passed else "❌ FAIL"}  {name:14s} max_diff={max_diff:.6f}  '
          f'mean_diff={mean_diff:.8f}  top-1 agreement={top1:.1%}  '
          f'(atol={ATOL[precision]:g}, min top-1={MIN_TOP1[precision]:.0%})')
    return passed


def check_labels(export_dir, n_classes):
    """Check the label files against the number of model outputs; return a list of errors."""
    errors = []
    csv_path = os.path.join(export_dir, 'labels.csv')
    txt_path = os.path.join(export_dir, 'labels.txt')
    ids = None
    if os.path.isfile(csv_path):
        import pandas as pd
        labels = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        ids = list(labels['birdnet_id'])
        if len(labels) != n_classes:
            errors.append(f'labels.csv has {len(labels)} rows, the models have {n_classes} outputs')
        if len(set(ids)) != len(ids) or '' in ids:
            errors.append('labels.csv has empty or repeated birdnet_id values')
        if list(labels['ix']) != [str(i) for i in range(len(labels))]:
            errors.append('labels.csv ix column is not 0..n-1')
    if os.path.isfile(txt_path):
        with open(txt_path, encoding='utf-8') as f:
            rows = [line.rstrip('\n').split('\t') for line in f if line.strip()]
        if len(rows) != n_classes:
            errors.append(f'labels.txt has {len(rows)} rows, the models have {n_classes} outputs')
        if ids is not None and [row[0] for row in rows] != ids:
            errors.append('labels.txt and labels.csv list different ids or a different order')
    if not os.path.isfile(csv_path) and not os.path.isfile(txt_path):
        print('⚠️  No labels.csv or labels.txt in the export directory')
    return errors


def speed(name, run, inputs, runs):
    """Print the median time of ``run`` on one model batch."""
    batch = getattr(run, 'batch', 1)
    sample = inputs[:batch]
    for _ in range(2):
        run(sample)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run(sample)
        times.append(time.perf_counter() - start)
    print(f'  {name:14s} batch={batch}  {1000 * float(np.median(times)):8.3f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--export-dir', default=DEFAULT_EXPORT_DIR,
                        help='Directory written by convert_geomodel.py')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help='Model directory whose model.py loads checkpoint/model.pt')
    parser.add_argument('--speed-runs', type=int, default=10,
                        help='Timed runs per model in the speed test (0 to skip it)')
    args = parser.parse_args()
    export_dir = os.path.abspath(args.export_dir)

    reference_path = os.path.join(export_dir, 'torchscript', 'model.pt')
    if not os.path.isfile(reference_path):
        sys.exit(f'Error: TorchScript reference not found at {reference_path}')
    print(f'Geomodel comparison (reference: TorchScript)\nExport dir: {export_dir}\n')

    inputs = test_inputs()
    runners = {'TS': load_torchscript(reference_path)}
    reference = runners['TS'](inputs)
    print(f'Reference output: {reference.shape}, range [{reference.min():.4f}, {reference.max():.4f}]')
    if not np.isfinite(reference).all():
        sys.exit('Error: the TorchScript reference returns NaN or infinite values')

    passed = failed = models = 0

    def record(ok):
        nonlocal passed, failed
        passed += bool(ok)
        failed += not ok

    # LiteRT first: TensorFlow is not imported yet, so a model that needs the Flex
    # delegate fails here instead of passing because TensorFlow happened to be loaded.
    for name, file_name, precision in (('LiteRT', 'model.tflite', 'fp32'),
                                       ('LRT_fp16', 'model_fp16.tflite', 'fp16'),
                                       ('LRT_int8', 'model_int8.tflite', 'int8')):
        path = os.path.join(export_dir, 'tf', file_name)
        if not os.path.isfile(path):
            continue
        models += 1
        try:
            runners[name] = load_tflite(path)
            record(compare(name, runners[name](inputs), reference, precision))
        except Exception as error:
            runners.pop(name, None)
            print(f'❌ FAIL  {name:14s} does not run under LiteRT: {str(error)[:160]}')
            record(False)

    for name, file_name, precision in (('ONNX', 'model.onnx', 'fp32'),
                                       ('ONNX_fp16', 'model_fp16.onnx', 'fp16')):
        path = os.path.join(export_dir, 'onnx', file_name)
        if not os.path.isfile(path):
            continue
        models += 1
        try:
            runners[name] = load_onnx(path)
            whole = runners[name](inputs)
            record(compare(name, whole, reference, precision))
            for batch in (1, 7):
                part = runners[name](inputs[:batch])
                same = part.shape == (batch, reference.shape[1]) and np.allclose(part, whole[:batch], atol=1e-5)
                print(f'{"✅ PASS" if same else "❌ FAIL"}  {name:14s} batch={batch} matches the full batch')
                record(same)
        except Exception as error:
            runners.pop(name, None)
            print(f'❌ FAIL  {name:14s} {str(error)[:160]}')
            record(False)

    saved_model_dir = os.path.join(export_dir, 'tf')
    if os.path.isfile(os.path.join(saved_model_dir, 'saved_model.pb')):
        models += 1
        try:
            runner = load_saved_model(saved_model_dir)
            if runner is None:
                print('⚠️  SKIP  SavedModel     TensorFlow is not installed')
            else:
                runners['SavedModel'] = runner
                record(compare('SavedModel', runner(inputs), reference, 'fp32'))
        except Exception as error:
            runners.pop('SavedModel', None)
            print(f'❌ FAIL  SavedModel     {str(error)[:160]}')
            record(False)

    for name, file_name, precision in (('CoreML', 'model.mlpackage', 'coreml_fp32'),
                                       ('CML_fp16', 'model_fp16.mlpackage', 'coreml_fp16')):
        path = os.path.join(export_dir, 'coreml', file_name)
        if not os.path.isdir(path):
            continue
        if sys.platform != 'darwin':
            print(f'⚠️  SKIP  {name:14s} Core ML models only run on macOS')
            continue
        models += 1
        try:
            runners[name] = load_coreml(path)
            record(compare(name, runners[name](inputs), reference, precision))
        except Exception as error:
            runners.pop(name, None)
            print(f'❌ FAIL  {name:14s} {str(error)[:160]}')
            record(False)

    checkpoint_path = os.path.join(export_dir, 'checkpoint', 'model.pt')
    if os.path.isfile(checkpoint_path) and not os.path.isfile(os.path.join(args.model_dir, 'model.py')):
        print(f'⚠️  SKIP  Checkpoint     no model.py in {args.model_dir} (see --model-dir)')
    elif os.path.isfile(checkpoint_path):
        models += 1
        try:
            runners['Checkpoint'] = load_checkpoint_runner(checkpoint_path, args.model_dir)
            record(compare('Checkpoint', runners['Checkpoint'](inputs), reference, 'fp32'))
        except Exception as error:
            runners.pop('Checkpoint', None)
            print(f'❌ FAIL  Checkpoint     {str(error)[:160]}')
            record(False)

    if models == 0:
        print('❌ FAIL  no converted model found to compare with the TorchScript reference')
        record(False)

    label_errors = check_labels(export_dir, reference.shape[1])
    for error in label_errors:
        print(f'❌ FAIL  labels         {error}')
        record(False)
    n_anchors, anchor_errors = check_anchors(export_dir, runners['TS'])
    for error in anchor_errors:
        print(f'❌ FAIL  labels         {error}')
        record(False)
    if n_anchors and not anchor_errors:
        print(f'✅ PASS  labels         {n_anchors} species/place checks agree with the labels')
        record(True)
    elif not n_anchors:
        print('⚠️  SKIP  labels         none of the reference species is in the labels; '
              'labels were not checked against the model')

    if args.speed_runs > 0:
        print(f'\nSpeed test: one model batch on the CPU, median of {args.speed_runs} runs after 2 warm-up runs')
        for name, run in runners.items():
            try:
                speed(name, run, inputs, args.speed_runs)
            except Exception as error:
                print(f'  {name:14s} failed: {str(error)[:120]}')

    print(f'\n{passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
