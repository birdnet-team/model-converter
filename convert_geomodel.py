"""Convert a BirdNET+ Geomodel checkpoint to TorchScript, ONNX, LiteRT, TensorFlow and Core ML.

The geomodel predicts which species occur at a place and time of year. Every exported
model takes one float32 tensor ``[batch, 3]`` with the columns latitude (degrees),
longitude (degrees) and week (1-48), and returns ``[batch, n_species]``.

Output, in ``{output_dir}/{model name}/``:

    checkpoint/model.pt         PyTorch checkpoint with the exported classes (weights only),
                                with model_labels.txt next to it
    torchscript/model.pt        TorchScript, raw logits (probabilities with --torchscript-sigmoid)
    onnx/model.onnx             ONNX FP32, probabilities, any batch size
    onnx/model_fp16.onnx        ONNX with weights stored as float16, computed in float32
    tf/model.tflite             LiteRT FP32, probabilities, batch size fixed at --batch-size
    tf/model_fp16.tflite        LiteRT with float16 weights
    tf/model_int8.tflite        LiteRT with int8 weights (--experimental-mode only)
    tf/saved_model.pb, ...      TensorFlow SavedModel, batch size fixed at --batch-size
    coreml/model.mlpackage      Core ML FP32, probabilities, batch size fixed at --batch-size
    coreml/model_fp16.mlpackage Core ML computing in float16, coordinate encoding in float32
    labels.csv, labels.txt      one row per model output
    MODEL_LICENSE.txt, ACCEPTABLE_USE.txt

Each run deletes the model files of an earlier run in that directory first, so the models
always match the label files. Every exported model is run on a fixed set of locations and weeks and compared with the
PyTorch model; the script exits with an error if one differs by more than its tolerance.
Core ML models only run on macOS; on other systems they are written without being run.

Usage:
    python convert_geomodel.py                                  # default model, everything
    python convert_geomodel.py -f onnx -p fp16                  # ONNX FP16 only
    python convert_geomodel.py --checkpoint my_run/best.pt --no-class-pruning
    python convert_geomodel.py --map-labels my_run/best_labels.txt --sheet-out my_sheet.xlsx
"""

import argparse
import copy
import importlib.util
import math
import os
import shutil
import sys

import numpy as np
import torch

# Label, taxonomy and license handling is shared with the audio model converter
from convert_model import (
    COREML_DEPLOYMENT_TARGET,
    DEFAULT_CLASS_PRUNING_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TAXONOMY_FILE,
    _list_models,
    _resave_saved_model_static,
    build_taxonomy_labels,
    load_class_ixs_from_excel,
    read_pruned_labels,
    write_terms_files,
)

DEFAULT_MODEL_NAME = '03_BirdNET+_Geomodel_V3.0.5_Global_14K'
DEFAULT_CHECKPOINT_FILE = 'checkpoint.pt'

ALL_FORMATS = ['checkpoint', 'torchscript', 'onnx', 'tflite', 'tf', 'coreml']
ALL_PRECISIONS = ['fp32', 'fp16', 'int8']

ONNX_OPSET = 18
INPUT_NAME = 'input'
OUTPUT_NAME = 'probabilities'

# Largest allowed difference in output probability between an export and the PyTorch model.
# INT8 reaches 0.13 on the reference inputs and 0.2 on other random locations.
TOLERANCE = {'fp32': 1e-4, 'fp16': 0.01, 'int8': 0.3,
             'coreml_fp32': 1e-4, 'coreml_fp16': 0.01}
# ONNX FP16 stores a weight as float16 only if it has at least this many elements;
# biases, normalisation parameters and scalar constants stay float32
FP16_MIN_ELEMENTS = 4096

# Columns of the class pruning sheet written by --map-labels
SHEET_COLUMNS = ['ix', 'birdnet_id', 'scientific_name', 'common_name', 'source_code',
                 'source_scientific_name', 'source_common_name', 'matched_by', 'review', 'selected']
# Subdirectories of the export directory that hold model files; emptied at the start of a run
MODEL_SUBDIRS = ('checkpoint', 'torchscript', 'onnx', 'tf', 'coreml')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_model_dir(model_name):
    """Return the model directory for a name in models/ or a path."""
    model_arg = model_name.rstrip('/\\')
    if os.path.dirname(model_arg):
        return os.path.abspath(model_arg)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_arg)


def load_model_module(model_dir):
    """Import model.py from the model directory and check that it defines a geomodel."""
    path = os.path.join(model_dir, 'model.py')
    if not os.path.isfile(path):
        sys.exit(f'Error: model definition not found at {path}')
    spec = importlib.util.spec_from_file_location('geomodel_module', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['geomodel_module'] = module
    spec.loader.exec_module(module)
    if getattr(module, 'MODEL_TYPE', None) != 'geomodel':
        sys.exit(f'Error: {path} is not a geomodel definition; audio models are converted '
                 f'with convert_model.py')
    return module


def reference_inputs(n=256):
    """Return a fixed ``(n, 3)`` float32 array of latitude, longitude and week.

    The first rows are real places, the poles, the date line, every week of the year and
    coordinates at round fractions of 180 degrees; the rest are uniformly random. Weeks and
    round coordinates put the encoded angles on exact fractions of pi, where a runtime's
    sin or cos can be wrong (see ``_register_coreml_ops``).
    """
    fixed = np.array([
        [50.83, 12.92, 20], [42.45, -76.47, 22], [-33.87, 151.21, 44], [0.0, 0.0, 1],
        [-1.29, 36.82, 10], [64.13, -21.90, 26], [90.0, 0.0, 48], [-90.0, 0.0, 1],
        [35.68, 180.0, 12], [35.68, -180.0, 12], [-54.80, -68.30, 48], [1.35, 103.82, 30],
    ] + [[50.83, 12.92, week] for week in range(1, 49)]
      + [[lat, lon, 20] for lat in (45.0, 22.5, -45.0, 30.0, 7.5)
         for lon in (45.0, -135.0, 22.5, 90.0, 7.5)], dtype=np.float32)
    rng = np.random.RandomState(42)
    random = np.stack([rng.uniform(-90, 90, n), rng.uniform(-180, 180, n),
                       rng.randint(1, 49, n)], axis=1).astype(np.float32)
    return np.concatenate([fixed, random])[:n]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def read_source_labels(path):
    """Read a geomodel labels.txt: one ``code<TAB>scientific name<TAB>common name`` per line."""
    import pandas as pd
    if not os.path.isfile(path):
        sys.exit(f'Error: labels file not found: {path}')
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for number, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                sys.exit(f'Error: {path} line {number} is not tab-separated: {line!r}')
            rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip() if len(parts) > 2 else ''))
    if not rows:
        sys.exit(f'Error: {path} has no labels')
    labels = pd.DataFrame(rows, columns=['code', 'scientific_name', 'common_name'])
    repeated = labels['code'][labels['code'].duplicated()].unique()
    if len(repeated):
        sys.exit(f'Error: {path} lists code(s) more than once: {", ".join(repeated[:10])}')
    return labels


def build_pruning_sheet(labels_path, taxonomy_path):
    """Match geomodel labels to the BirdNET taxonomy and return a class pruning sheet.

    Each label is matched by its scientific name, then by its code (iNaturalist id if
    numeric, eBird code otherwise), then by a scientific name alias that belongs to one
    taxon only. Labels without a match are not selected. Where the taxonomy lumps several
    labels into one taxon, the best match is selected (scientific name before code before
    alias, then alias order) and the others are not, so every selected row has its own
    birdnet_id. Returns one row per label with the columns in SHEET_COLUMNS.

    The ``review`` column marks rows a person should look at before converting: a code that
    belongs to a different taxon than the scientific name, and alias matches whose common
    name differs from the taxon's. An alias can be a plain synonym or a form the taxonomy
    lumps into a wider species; in the second case the class covers only part of the
    taxon's range. Set ``selected`` to 0 for rows that should not be exported.
    """
    import pandas as pd
    labels = read_source_labels(labels_path)
    if not os.path.isfile(taxonomy_path):
        sys.exit(f'Error: BirdNET taxonomy not found: {taxonomy_path}')
    taxonomy = pd.read_csv(taxonomy_path, dtype=str, keep_default_na=False)
    required = ['birdnet_id', 'scientific_name', 'common_name', 'inat_id', 'ebird_code',
                'scientific_name_aliases']
    missing = [c for c in required if c not in taxonomy.columns]
    if missing:
        sys.exit(f'Error: {taxonomy_path} is missing column(s): {", ".join(missing)}')

    for column in ('birdnet_id', 'scientific_name'):
        duplicated = taxonomy[column][taxonomy[column].duplicated()]
        if len(duplicated):
            sys.exit(f'Error: {taxonomy_path} lists {column} {duplicated.iloc[0]!r} more than once')
    by_name = dict(zip(taxonomy['scientific_name'], taxonomy['birdnet_id']))
    by_ebird = {c: b for c, b in zip(taxonomy['ebird_code'], taxonomy['birdnet_id']) if c}
    by_inat = {c: b for c, b in zip(taxonomy['inat_id'], taxonomy['birdnet_id']) if c}
    by_alias, alias_position = {}, {}
    for birdnet_id, aliases in zip(taxonomy['birdnet_id'], taxonomy['scientific_name_aliases']):
        for position, alias in enumerate(a.strip() for a in aliases.split('|')):
            if alias:
                by_alias.setdefault(alias, set()).add(birdnet_id)
                alias_position[(birdnet_id, alias)] = position

    names = taxonomy.set_index('birdnet_id')

    def differs(common, birdnet_id):
        return common.strip().lower() != names['common_name'][birdnet_id].strip().lower()

    ids, matched_by, review = [], [], []
    for code, name, common in zip(labels['code'], labels['scientific_name'], labels['common_name']):
        by_code = (by_inat if code.isdigit() else by_ebird).get(code)
        note = ''
        if name in by_name:
            ids.append(by_name[name]); matched_by.append('scientific_name')
            if by_code and by_code != by_name[name]:
                note = (f'code {code} belongs to {by_code} '
                        f'{names["scientific_name"][by_code]} in the taxonomy')
        elif by_code:
            ids.append(by_code); matched_by.append('code')
            if differs(common, by_code):  # codes move when taxa are split or lumped
                note = (f'matched by code only, to {names["scientific_name"][by_code]} '
                        f'({names["common_name"][by_code]}); check that it is the same taxon')
        elif len(by_alias.get(name, ())) == 1:
            ids.append(next(iter(by_alias[name]))); matched_by.append('alias')
            if differs(common, ids[-1]):
                note = (f'alias of {names["scientific_name"][ids[-1]]} '
                        f'({names["common_name"][ids[-1]]}); '
                        f'check that it is the same taxon and not a part of it')
        else:
            ids.append(''); matched_by.append('unmatched')
        review.append(note)

    # One selected row per birdnet_id: the best match of every lumped group
    rank = {'scientific_name': 0, 'code': 1, 'alias': 2}
    selected = [1 if birdnet_id else 0 for birdnet_id in ids]
    groups = {}
    for row, birdnet_id in enumerate(ids):
        if birdnet_id:
            groups.setdefault(birdnet_id, []).append(row)
    for birdnet_id, rows in groups.items():
        if len(rows) > 1:
            rows.sort(key=lambda r: (rank[matched_by[r]],
                                     alias_position.get((birdnet_id, labels['scientific_name'][r]), 0), r))
            for row in rows[1:]:
                selected[row] = 0
                matched_by[row] += f'; lumped into {labels["scientific_name"][rows[0]]}'
            dropped = ', '.join(labels['scientific_name'][row] for row in rows[1:])
            review[rows[0]] = '; '.join(filter(None, [
                review[rows[0]],
                f'the taxonomy lumps {dropped} into this taxon and their outputs are dropped; '
                f'this class predicts only the range of {labels["scientific_name"][rows[0]]}']))

    sheet = pd.DataFrame({
        'ix': range(len(labels)),
        'birdnet_id': ids,
        'scientific_name': [names['scientific_name'].get(b, '') for b in ids],
        'common_name': [names['common_name'].get(b, '') for b in ids],
        'source_code': labels['code'],
        'source_scientific_name': labels['scientific_name'],
        'source_common_name': labels['common_name'],
        'matched_by': matched_by,
        'review': review,
        'selected': selected,
    })
    return sheet[SHEET_COLUMNS]


def write_label_files(output_dir, taxonomy_labels, plain_labels):
    """Write labels.csv and labels.txt, removing copies left by an earlier run.

    taxonomy_labels is the table from build_taxonomy_labels or None; plain_labels is a list
    of ``(id, scientific name, common name)`` per output, or None when nothing is known.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'labels.csv')
    txt_path = os.path.join(output_dir, 'labels.txt')
    for path in (csv_path, txt_path):
        if os.path.exists(path):
            os.remove(path)
    if taxonomy_labels is not None:
        taxonomy_labels.to_csv(csv_path, index=False)
        print(f'Wrote {len(taxonomy_labels)} labels to {csv_path}')
    if plain_labels is not None:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for row in plain_labels:
                f.write('\t'.join(str(v).replace('\t', ' ') for v in row) + '\n')
        print(f'Wrote {len(plain_labels)} labels to {txt_path}')


def check_sheet_matches_checkpoint(sheet_path, kept_ixs, checkpoint, checkpoint_path):
    """Exit if the pruning sheet was made for a different checkpoint.

    The sheet's ``ix`` only means something for the checkpoint it was built from. Where
    the checkpoint has a species vocabulary and the sheet has ``source_code`` or
    ``birdnet_id``, the name the checkpoint gives each selected output must be the row's
    source code or its BirdNET ID. Without those the sheet cannot be checked.
    """
    import pandas as pd
    vocab = checkpoint.get('species_vocab') or {}
    idx_to_species = vocab.get('idx_to_species')
    sheet = pd.read_excel(sheet_path, dtype=str, keep_default_na=False)
    columns = [c for c in ('source_code', 'birdnet_id') if c in sheet.columns]
    if not idx_to_species or not columns:
        print(f'⚠️  Cannot check that {sheet_path} belongs to {checkpoint_path}: '
              + ('the checkpoint has no species vocabulary' if not idx_to_species
                 else 'the sheet has no source_code or birdnet_id column'))
        return
    selected = sheet[pd.to_numeric(sheet['selected'], errors='coerce') == 1]
    wrong = []
    for ix, (_, row) in zip(kept_ixs, selected.iterrows()):
        name = str(idx_to_species[ix])
        if name not in {row[c].strip() for c in columns}:
            wrong.append(f'ix {ix}: checkpoint has {name}, sheet has '
                         + ' / '.join(row[c] for c in columns))
    if wrong:
        sys.exit(f'Error: {sheet_path} does not belong to {checkpoint_path}; {len(wrong)} of '
                 f'{len(kept_ixs)} selected rows name another class, e.g. {"; ".join(wrong[:3])}')


def taxonomy_labels_for_ids(ids, taxonomy_path):
    """Return the labels.csv table for outputs named by BirdNET IDs, or None.

    Used when no pruning sheet applies: a checkpoint whose vocabulary already holds BirdNET
    IDs (such as one exported by this script) still gets a full labels.csv.
    """
    import pandas as pd
    if not ids:
        return None
    if not os.path.isfile(taxonomy_path):
        print(f'⚠️  BirdNET taxonomy not found: {taxonomy_path}; labels.csv is not written')
        return None
    known = set(pd.read_csv(taxonomy_path, usecols=['birdnet_id'], dtype=str)['birdnet_id'])
    if len(set(ids)) != len(ids) or not set(ids) <= known:
        return None
    return build_taxonomy_labels(pd.DataFrame({'birdnet_id': list(ids)}), taxonomy_path)


def clear_model_files(output_dir):
    """Delete the model files of an earlier run from the export directory.

    The label files are rewritten on every run, so a model left from a run with other
    classes or another class order would no longer match them.
    """
    for name in MODEL_SUBDIRS:
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f'Removed {path} left by an earlier run')


def find_sidecar_labels(checkpoint_path):
    """Return the labels.txt that geomodel training writes next to a checkpoint, or None."""
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    for name in (f'{stem}_labels.txt', 'labels.txt'):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class Results:
    """Collects the comparison of every exported file with the PyTorch reference."""

    def __init__(self, reference):
        self.reference = reference
        self.rows = []

    def check(self, name, path, output, precision, size_of=None):
        """Record how far ``output`` is from the reference and return whether it passes.

        ``size_of`` names the entries of the directory ``path`` that make up the model.
        """
        tolerance = TOLERANCE[precision]
        output = np.asarray(output, dtype=np.float32)
        if output.shape != self.reference.shape or not np.isfinite(output).all():
            max_diff, top1, passed = float('nan'), float('nan'), False
        else:
            max_diff = float(np.abs(output - self.reference).max())
            top1 = float((output.argmax(1) == self.reference.argmax(1)).mean())
            passed = max_diff <= tolerance
        size_mb = (sum(_size_mb(os.path.join(path, entry)) for entry in size_of) if size_of
                   else _size_mb(path))
        print(f'  {"✅" if passed else "❌"} {name}: max diff {max_diff:.6f} (tolerance {tolerance:g}), '
              f'top-1 agreement {top1:.1%}, {size_mb:.1f} MB')
        self.rows.append((name, passed, max_diff, top1, size_mb))
        return passed

    def fail(self, name, error):
        print(f'  ❌ {name}: {error}')
        self.rows.append((name, False, float('nan'), float('nan'), 0.0))

    def not_run(self, name, path, reason):
        """Record a file that was written and checked for structure but not executed."""
        print(f'  ⚠️  {name}: not run. {reason}')
        self.rows.append((name, None, float('nan'), float('nan'), _size_mb(path)))

    def summary(self):
        """Print all results and return True if every export passed."""
        print('\nSummary (difference to the PyTorch model on '
              f'{len(self.reference)} locations and weeks)')
        for name, passed, max_diff, top1, size_mb in self.rows:
            if passed is None:
                print(f'  NOT RUN  {name:22s} {size_mb:.1f} MB')
            else:
                print(f'  {"PASS" if passed else "FAIL"}     {name:22s} max diff {max_diff:.6f}  '
                      f'top-1 {top1:.1%}  {size_mb:.1f} MB')
        return all(row[1] is not False for row in self.rows)


def _size_mb(path):
    """Return the size of a file, or of all files in a directory, in MB."""
    if os.path.isdir(path):
        total = sum(os.path.getsize(os.path.join(root, name))
                    for root, _, names in os.walk(path) for name in names)
    else:
        total = os.path.getsize(path)
    return total / (1024 * 1024)


def run_tflite(path, inputs):
    """Run a fixed-batch LiteRT model on ``inputs``, one model batch at a time."""
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    batch = int(input_details['shape'][0])
    outputs = []
    for start in range(0, len(inputs), batch):
        chunk = inputs[start:start + batch]
        n = len(chunk)
        if n < batch:  # pad the last batch with its first row
            chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch - n, axis=0)])
        interpreter.set_tensor(input_details['index'], chunk.astype(np.float32))
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_details['index'])[:n].copy())
    return np.concatenate(outputs)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_checkpoint(model, source_checkpoint, label_ids, kept_ixs, output_dir, plain_labels=None):
    """Write the exported classes as a weights-only geomodel checkpoint.

    The result loads with ``load_checkpoint`` and with the geomodel repository's own
    scripts. Optimizer, scheduler and training history are left out. The species
    vocabulary is rewritten to the exported order; its names are ``label_ids`` when given.
    ``plain_labels`` is written next to it as ``model_labels.txt``.
    """
    path = os.path.join(output_dir, 'checkpoint', 'model.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config = dict(source_checkpoint['model_config'])
    config['n_species'] = model.n_species
    checkpoint = {'model_config': config, 'model_state_dict': model.state_dict()}
    for key in ('epoch', 'best_geoscore'):
        if key in source_checkpoint:
            checkpoint[key] = source_checkpoint[key]

    vocab = source_checkpoint.get('species_vocab')
    old_names = None
    if isinstance(vocab, dict) and 'idx_to_species' in vocab:
        idx_to_species = vocab['idx_to_species']
        old_names = [idx_to_species[i] for i in kept_ixs]
    names = list(label_ids) if label_ids is not None else old_names
    if names is not None:
        new_vocab = {k: v for k, v in (vocab or {}).items()
                     if k not in ('species_to_idx', 'idx_to_species', 'ocean_species')}
        new_vocab['species_to_idx'] = {name: i for i, name in enumerate(names)}
        new_vocab['idx_to_species'] = {i: name for i, name in enumerate(names)}
        if old_names is not None and 'ocean_species' in vocab:
            ocean = set(vocab['ocean_species'])
            new_vocab['ocean_species'] = [new for old, new in zip(old_names, names) if old in ocean]
        checkpoint['species_vocab'] = new_vocab
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path} ({_size_mb(path):.1f} MB, {model.n_species} classes)')
    if plain_labels is not None:
        # The geomodel scripts look for <checkpoint name>_labels.txt next to the checkpoint
        with open(os.path.join(os.path.dirname(path), 'model_labels.txt'), 'w', encoding='utf-8') as f:
            for row in plain_labels:
                f.write('\t'.join(str(v).replace('\t', ' ') for v in row) + '\n')
    return path


def export_torchscript(module, model, inputs, results, output_dir, apply_sigmoid):
    """Export a traced, frozen TorchScript module that takes ``[batch, 3]``."""
    path = os.path.join(output_dir, 'torchscript', 'model.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wrapper = module.ExportWrapper(model, apply_sigmoid=apply_sigmoid).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, torch.from_numpy(inputs[:2]))
    torch.jit.save(torch.jit.freeze(traced), path)

    reloaded = torch.jit.load(path, map_location='cpu')
    with torch.no_grad():
        output = reloaded(torch.from_numpy(inputs))
        if not apply_sigmoid:
            output = torch.sigmoid(output)
    print(f'TorchScript model saved to {path} '
          f'(returns {"probabilities" if apply_sigmoid else "raw logits"})')
    results.check('TorchScript', path, output.numpy(), 'fp32')


def store_weights_as_fp16(onnx_model):
    """Store the large weight matrices of an ONNX model as float16, in place.

    Each float32 initializer with two or more dimensions and at least FP16_MIN_ELEMENTS
    elements is replaced by a float16 copy and a Cast back to float32, so the graph still
    computes in float32 and only weight rounding changes the result. Biases, normalisation
    parameters and scalar constants stay float32. Weights outside the float16 range are
    left alone. Returns ``(converted, kept)`` initializer counts.
    """
    from onnx import TensorProto, helper, numpy_helper
    graph = onnx_model.graph
    fp16_max = float(np.finfo(np.float16).max)
    converted, kept, casts = 0, 0, []
    for index in range(len(graph.initializer)):
        initializer = graph.initializer[index]
        if initializer.data_type != TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(initializer)
        if array.ndim < 2 or array.size < FP16_MIN_ELEMENTS or np.abs(array).max() > fp16_max:
            kept += 1
            continue
        name = initializer.name
        fp16_name = f'{name}_fp16'
        graph.initializer[index].CopyFrom(numpy_helper.from_array(array.astype(np.float16), fp16_name))
        casts.append(helper.make_node('Cast', [fp16_name], [name], to=TensorProto.FLOAT,
                                      name=f'cast_fp16_{converted}'))
        converted += 1
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return converted, kept


def export_onnx(module, model, inputs, results, output_dir, precisions):
    """Export ONNX FP32 and, from it, the FP16 model with float16 weights."""
    import onnx
    import onnxruntime as ort

    onnx_dir = os.path.join(output_dir, 'onnx')
    os.makedirs(onnx_dir, exist_ok=True)
    fp32_path = os.path.join(onnx_dir, 'model.onnx')
    tmp_path = os.path.join(onnx_dir, 'temp.onnx')
    wrapper = module.ExportWrapper(model).eval()
    torch.onnx.export(
        wrapper, torch.from_numpy(inputs[:2]), tmp_path,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: 'batch'}, OUTPUT_NAME: {0: 'batch'}},
        opset_version=ONNX_OPSET, dynamo=False,
    )
    fp32_model = onnx.load(tmp_path, load_external_data=True)
    for name in os.listdir(onnx_dir):
        if name.startswith('temp.onnx'):
            os.remove(os.path.join(onnx_dir, name))
    onnx.checker.check_model(fp32_model)

    def validate(name, path, precision):
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        whole = session.run(None, {INPUT_NAME: inputs})[0]
        single = session.run(None, {INPUT_NAME: inputs[:1]})[0]
        if not np.allclose(single, whole[:1], atol=1e-5):
            results.fail(name, 'batch size 1 gives a different result than a larger batch')
            return
        results.check(name, path, whole, precision)

    if 'fp32' in precisions:
        onnx.save(fp32_model, fp32_path)
        print(f'ONNX model saved to {fp32_path}')
        validate('ONNX', fp32_path, 'fp32')
    if 'fp16' in precisions:
        fp16_path = os.path.join(onnx_dir, 'model_fp16.onnx')
        converted, kept = store_weights_as_fp16(fp32_model)
        onnx.checker.check_model(fp32_model)
        onnx.save(fp32_model, fp16_path)
        print(f'ONNX FP16 model saved to {fp16_path} '
              f'({converted} weight matrices as float16, {kept} small tensors kept float32)')
        validate('ONNX FP16', fp16_path, 'fp16')
    if 'int8' in precisions and not {'fp32', 'fp16'} & set(precisions):
        print('  ⚠️  Skipping ONNX — there is no INT8 ONNX export for the geomodel.')


class _ErfGELU(torch.nn.Module):
    """GELU written with erf: the same function as ``nn.GELU()``, without the composite op."""

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def _replace_gelu(module):
    """Replace every ``nn.GELU`` below ``module`` with ``_ErfGELU``, in place."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, _ErfGELU())
        else:
            _replace_gelu(child)


def export_tf(module, model, inputs, results, output_dir, formats, precisions, batch_size,
              experimental):
    """Export the LiteRT models and the TensorFlow SavedModel with litert_torch.

    litert_torch converts a fixed input shape, so all of these take exactly
    ``[batch_size, 3]``. The LiteRT models use only built-in operators (GELU included), so
    they run without the TensorFlow Flex delegate.
    """
    import litert_torch
    import tensorflow as tf

    tf_dir = os.path.join(output_dir, 'tf')
    os.makedirs(tf_dir, exist_ok=True)
    wrapper = module.ExportWrapper(model).eval()
    sample = (torch.zeros(batch_size, 3),)

    if 'tflite' in formats:
        variants = [
            ('fp32', 'LiteRT', 'model.tflite', {}),
            ('fp16', 'LiteRT FP16', 'model_fp16.tflite',
             {'optimizations': [tf.lite.Optimize.DEFAULT],
              'target_spec': {'supported_types': [tf.float16]}}),
            # Dynamic range quantisation: int8 weights, float32 computation
            ('int8', 'LiteRT INT8', 'model_int8.tflite', {'optimizations': [tf.lite.Optimize.DEFAULT]}),
        ]
        for precision, name, file_name, flags in variants:
            path = os.path.join(tf_dir, file_name)
            if precision not in precisions:
                continue
            if precision == 'int8' and not experimental:
                print('  ⚠️  Skipping LiteRT INT8 — this is only supported in experimental mode. '
                      'Use --experimental-mode to enable.')
                continue
            edge_model = litert_torch.convert(wrapper, sample, _ai_edge_converter_flags=flags)
            edge_model.export(path)
            print(f'{name} model saved to {path}')
            try:
                results.check(name, path, run_tflite(path, inputs), precision)
            except Exception as error:  # e.g. an operator the LiteRT runtime lacks
                results.fail(name, f'does not run: {str(error)[:200]}')

    if 'tf' in formats:
        if 'fp32' not in precisions:
            print('  ⚠️  Skipping TensorFlow SavedModel — it is FP32 only.')
            return
        # TensorFlow cannot run the composite GELU that litert_torch emits for nn.GELU,
        # so the SavedModel is converted from a copy that spells GELU out with erf.
        erf_wrapper = copy.deepcopy(wrapper)
        _replace_gelu(erf_wrapper)
        tmp_dir = os.path.join(tf_dir, '_saved_model_tmp')
        shutil.rmtree(tmp_dir, ignore_errors=True)
        litert_torch.convert(erf_wrapper, sample, _saved_model_dir=tmp_dir)
        for name in os.listdir(tmp_dir):
            shutil.move(os.path.join(tmp_dir, name), os.path.join(tf_dir, name))
        shutil.rmtree(tmp_dir)
        _resave_saved_model_static(tf_dir, [batch_size, 3])

        with tf.device('/CPU:0'):
            serving = tf.saved_model.load(tf_dir).signatures['serving_default']
            key = list(serving.structured_input_signature[1].keys())[0]
            outputs = []
            for start in range(0, len(inputs), batch_size):
                chunk = inputs[start:start + batch_size]
                n = len(chunk)
                if n < batch_size:
                    chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch_size - n, axis=0)])
                outputs.append(list(serving(**{key: tf.constant(chunk)}).values())[0].numpy()[:n])
        print(f'TensorFlow SavedModel saved to {tf_dir}')
        results.check('SavedModel', tf_dir, np.concatenate(outputs), 'fp32', size_of=('saved_model.pb', 'variables'))


def _register_coreml_ops():
    """Make coremltools convert ``sin(x)`` as ``cos(x - pi/2)``.

    Core ML's sin returns the wrong sign for some exact arguments: on macOS 15 (arm64),
    sin(pi/4) came out as -0.7071 and sin(5*pi/4) as +0.7071 in this model, on the CPU and
    the GPU. Weeks 2, 3, 4 and 7 and coordinates at multiples of 7.5 degrees produce such
    angles, and the output probabilities were then off by up to 0.99. Core ML's cos is
    correct at these arguments; with sin replaced, the model matches PyTorch to 0.00001.
    """
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as ct_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

    @register_torch_op(override=True)
    def sin(context, node):
        x = ct_ops._get_inputs(context, node, expected=1)[0]
        shifted = mb.sub(x=x, y=np.float32(np.pi / 2))
        context.add(mb.cos(x=shifted, name=node.name))


def run_coreml(path, inputs):
    """Run a fixed-batch Core ML model on ``inputs`` (macOS only)."""
    import coremltools as ct
    mlmodel = ct.models.MLModel(path)
    batch = int(mlmodel.get_spec().description.input[0].type.multiArrayType.shape[0])
    outputs = []
    for start in range(0, len(inputs), batch):
        chunk = inputs[start:start + batch]
        n = len(chunk)
        if n < batch:
            chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch - n, axis=0)])
        outputs.append(np.asarray(mlmodel.predict({INPUT_NAME: chunk})[OUTPUT_NAME])[:n])
    return np.concatenate(outputs)


def _coreml_weights_as_fp16_pass():
    """Register and return the name of the graph pass that stores weights as float16.

    Every float32 weight matrix with at least ``FP16_MIN_ELEMENTS`` elements, the same
    tensors as in the ONNX FP16 model, is stored as float16 and cast back to float32 when the model is loaded
    (``constexpr_cast``), so the model is half the size but computes in float32, like the
    ONNX FP16 model. Measured on macOS: 0.0025 from the PyTorch model, against 0.018 when
    the model computes in float16.
    """
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
    from coremltools.converters.mil.mil.passes.helper import block_context_manager
    from coremltools.converters.mil.mil.passes.pass_registry import register_pass

    @register_pass(namespace='common')
    class geomodel_weights_as_fp16(AbstractGraphPass):
        def apply(self, prog):
            for function in prog.functions.values():
                self._convert(function)

        @block_context_manager
        def _convert(self, block):
            for op in list(block.operations):
                value = op.outputs[0].val if op.op_type == 'const' else None
                if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                        or value.ndim < 2 or value.size < FP16_MIN_ELEMENTS
                        or np.abs(value).max() > np.finfo(np.float16).max):
                    continue
                stored = mb.constexpr_cast(source_val=value.astype(np.float16), output_dtype='fp32',
                                           before_op=op, name=op.name + '_fp16')
                block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0],
                                                   new_var=stored, no_check_var_types=True)
                block.remove_ops([op])

    return 'common::geomodel_weights_as_fp16'


def coreml_structure_problem(path, fp16):
    """Return what is wrong with the operations of a saved Core ML model, or None.

    Checks what can be checked without running the model, so it also works off macOS:
    no sin operation is left (see ``_register_coreml_ops``), and the FP16 model stores
    weights as float16.
    """
    import coremltools as ct
    counts = {}
    for function in ct.utils.load_spec(path).mlProgram.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                counts[operation.type] = counts.get(operation.type, 0) + 1
    if counts.get('sin'):
        return 'contains a sin operation, which Core ML evaluates wrongly at some angles'
    if fp16 and not counts.get('constexpr_cast'):
        return 'has no weights stored as float16'
    return None


def export_coreml(module, model, inputs, results, output_dir, precisions, batch_size):
    """Export Core ML ML Programs (.mlpackage) for iOS 16 / macOS 13 and later.

    The input shape is fixed at ``[batch_size, 3]``. Both models compute in float32;
    ``model_fp16.mlpackage`` stores its weights as float16 (see
    ``_coreml_weights_as_fp16_pass``).

    coremltools converts on any OS, but Core ML only runs on macOS: there the models are
    compared with the PyTorch model like every other format, elsewhere only their
    operations are checked.
    """
    import coremltools as ct

    _register_coreml_ops()
    coreml_dir = os.path.join(output_dir, 'coreml')
    os.makedirs(coreml_dir, exist_ok=True)
    wrapper = module.ExportWrapper(model).eval()
    example = torch.from_numpy(inputs[:1]).repeat(batch_size, 1)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)

    fp16_pipeline = ct.PassPipeline.DEFAULT
    fp16_pipeline.append_pass(_coreml_weights_as_fp16_pass())
    variants = [
        ('fp32', 'Core ML', 'model.mlpackage', {}, 'coreml_fp32'),
        ('fp16', 'Core ML FP16', 'model_fp16.mlpackage', {'pass_pipeline': fp16_pipeline},
         'coreml_fp16'),
    ]
    for precision, name, file_name, options, tolerance in variants:
        if precision not in precisions:
            continue
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name=INPUT_NAME, shape=(batch_size, 3), dtype=np.float32)],
            outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float32)],
            convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=getattr(ct.target, COREML_DEPLOYMENT_TARGET),
            skip_model_load=True,  # loading compiles the model, which only macOS can do
            **options,
        )
        mlmodel.short_description = ('BirdNET+ Geomodel: latitude, longitude and week in, '
                                     'per-class occurrence probabilities out.')
        mlmodel.license = 'Apache-2.0 (see MODEL_LICENSE.txt)'
        mlmodel.input_description[INPUT_NAME] = (
            f'float32 [{batch_size}, 3]: latitude (degrees), longitude (degrees), week (1-48)')
        mlmodel.output_description[OUTPUT_NAME] = 'Per-class probabilities (see labels file)'
        mlmodel.user_defined_metadata.update({'n_species': str(model.n_species)})
        path = os.path.join(coreml_dir, file_name)
        mlmodel.save(path)
        print(f'{name} model saved to {path}')
        problem = coreml_structure_problem(path, precision == 'fp16')
        if problem:
            results.fail(name, problem)
        elif sys.platform == 'darwin':
            try:
                results.check(name, path, run_coreml(path, inputs), tolerance)
            except Exception as error:
                results.fail(name, f'does not run: {str(error)[:200]}')
        else:
            results.not_run(name, path, 'Core ML models only run on macOS; check them there '
                                        'with test_geomodel.py.')
    if 'int8' in precisions and 'fp32' not in precisions and 'fp16' not in precisions:
        print('  ⚠️  Skipping Core ML — there is no INT8 Core ML export for the geomodel.')


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description='Convert a BirdNET+ Geomodel checkpoint to TorchScript, ONNX, LiteRT, TensorFlow and Core ML.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-m', '--model-name', default=DEFAULT_MODEL_NAME,
                        help='Model directory containing model.py: a name in models/ or a path')
    parser.add_argument('--checkpoint', default=None, metavar='PT',
                        help=f'Geomodel checkpoint to convert (default: {DEFAULT_CHECKPOINT_FILE} '
                             f'in the model directory)')
    parser.add_argument('-o', '--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Directory for converted model output')
    parser.add_argument('-f', '--formats', nargs='+', choices=ALL_FORMATS, default=ALL_FORMATS,
                        metavar='FMT', help=f'Output format(s). Choose from: {", ".join(ALL_FORMATS)}')
    parser.add_argument('-p', '--precisions', nargs='+', choices=ALL_PRECISIONS, default=ALL_PRECISIONS,
                        metavar='PREC', help=f'Precision variant(s). Choose from: {", ".join(ALL_PRECISIONS)}')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size baked into the LiteRT, SavedModel and Core ML models')
    parser.add_argument('--torchscript-sigmoid', action='store_true',
                        help='Make the TorchScript model return probabilities like the other '
                             'formats. By default it returns raw logits, as in the geomodel releases.')
    parser.add_argument('--class-pruning', default=None, metavar='XLSX',
                        help='Excel file of class labels. Rows with selected=1 are kept, using the ix '
                             f'column. If omitted, {DEFAULT_CLASS_PRUNING_FILE} in the model directory '
                             'is used when --checkpoint is not given.')
    parser.add_argument('--no-class-pruning', action='store_true',
                        help='Export every class of the checkpoint.')
    parser.add_argument('--taxonomy', default=DEFAULT_TAXONOMY_FILE, metavar='CSV',
                        help='BirdNET taxonomy used to write labels.csv and for --map-labels')
    parser.add_argument('--map-labels', default=None, metavar='LABELS_TXT',
                        help='Match a geomodel labels.txt to the taxonomy, write a class pruning '
                             'sheet to --sheet-out and exit without converting.')
    parser.add_argument('--sheet-out', default=None, metavar='XLSX',
                        help='Where --map-labels writes the class pruning sheet (default: next to '
                             'the labels file, as <name>_pruning.xlsx)')
    parser.add_argument('--experimental-mode', action='store_true',
                        help='Also export the INT8 LiteRT model, which changes the top prediction '
                             'for several percent of inputs.')
    parser.add_argument('--list-models', action='store_true',
                        help='List the model directories in models/ and exit')
    return parser


def _map_labels(args):
    """Run --map-labels: write the class pruning sheet and print what matched."""
    sheet = build_pruning_sheet(args.map_labels, args.taxonomy)
    out = args.sheet_out or os.path.splitext(args.map_labels)[0] + '_pruning.xlsx'
    sheet.to_excel(out, index=False)
    counts = sheet['matched_by'].where(~sheet['matched_by'].str.contains('lumped'), 'lumped').value_counts()
    print(f'Wrote {len(sheet)} rows to {out}: {int(sheet["selected"].sum())} selected')
    for reason, count in counts.items():
        print(f'  {reason}: {count}')
    dropped = sheet[sheet['selected'] == 0]
    for _, row in dropped.head(50).iterrows():
        print(f'  not selected: {row["source_code"]} {row["source_scientific_name"]} ({row["matched_by"]})')
    if len(dropped) > 50:
        print(f'  ... and {len(dropped) - 50} more')
    to_review = sheet[(sheet['review'] != '') & (sheet['selected'] == 1)]
    if len(to_review):
        print(f'\n⚠️  {len(to_review)} selected row(s) need a look (column "review"); '
              f'set selected to 0 for classes that should not be exported:')
        for _, row in to_review.iterrows():
            print(f'  {row["source_code"]} {row["source_scientific_name"]} '
                  f'({row["source_common_name"]}) -> {row["birdnet_id"]}: {row["review"]}')


def main():
    args = _build_parser().parse_args()
    if args.list_models:
        for name in _list_models(os.path.dirname(os.path.abspath(__file__))):
            print(f'  {name}')
        return
    if args.map_labels:
        _map_labels(args)
        return
    if args.batch_size < 1:
        sys.exit('Error: --batch-size must be at least 1')
    if args.class_pruning and args.no_class_pruning:
        sys.exit('Error: --class-pruning and --no-class-pruning cannot be used together')

    model_dir = resolve_model_dir(args.model_name)
    # A checkpoint given with --checkpoint is another model: it gets its own export folder
    export_name = (os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint
                   else os.path.basename(model_dir))
    output_dir = os.path.join(args.output_dir, export_name)
    module = load_model_module(model_dir)
    checkpoint_path = args.checkpoint or os.path.join(model_dir, DEFAULT_CHECKPOINT_FILE)
    if not os.path.isfile(checkpoint_path):
        sys.exit(f'Error: checkpoint not found: {checkpoint_path}')
    with open(checkpoint_path, 'rb') as f:
        if f.read(40).startswith(b'version https://git-lfs'):
            sys.exit(f'Error: {checkpoint_path} is a Git LFS pointer; run `git lfs pull` to fetch it')
    formats, precisions = set(args.formats), set(args.precisions)
    float_precision = bool({'fp32', 'fp16'} & precisions)
    if not ({'checkpoint', 'torchscript'} & formats
            or ({'onnx', 'coreml'} & formats and float_precision)
            or ('tflite' in formats and (float_precision or args.experimental_mode))
            or ('tf' in formats and 'fp32' in precisions)):
        sys.exit('Error: nothing to export; the selected formats have no model in the selected '
                 'precisions (INT8 exists for LiteRT only, with --experimental-mode)')
    if os.path.commonpath([os.path.abspath(checkpoint_path), os.path.abspath(output_dir)]) == os.path.abspath(output_dir):
        sys.exit(f'Error: {checkpoint_path} is inside the export directory {output_dir}, whose model '
                 f'files this run replaces; copy the checkpoint elsewhere or choose another -o')
    model, checkpoint = module.load_checkpoint(checkpoint_path)
    n_source = model.n_species

    # The sheet in the model directory describes the checkpoint in the model directory
    class_pruning = args.class_pruning
    default_sheet = os.path.join(model_dir, DEFAULT_CLASS_PRUNING_FILE)
    if (class_pruning is None and not args.no_class_pruning and args.checkpoint is None
            and os.path.isfile(default_sheet)):
        class_pruning = default_sheet

    # Build and check every label table before anything is written
    taxonomy_labels, plain_labels, label_ids = None, None, None
    if class_pruning:
        kept_ixs = load_class_ixs_from_excel(class_pruning)
        if len(set(kept_ixs)) != len(kept_ixs) or min(kept_ixs) < 0 or max(kept_ixs) >= n_source:
            sys.exit(f'Error: {class_pruning} selects an ix twice or outside 0..{n_source - 1}, '
                     f'the outputs of {checkpoint_path}')
        check_sheet_matches_checkpoint(class_pruning, kept_ixs, checkpoint, checkpoint_path)
        pruned_labels = read_pruned_labels(class_pruning)
        taxonomy_labels = build_taxonomy_labels(pruned_labels, args.taxonomy)
        if taxonomy_labels is not None:
            label_ids = list(taxonomy_labels['birdnet_id'])
            plain_labels = list(zip(taxonomy_labels['birdnet_id'], taxonomy_labels['scientific_name'],
                                    taxonomy_labels['common_name']))
        else:
            print(f'⚠️  {class_pruning} has no birdnet_id column; labels.csv is not written')
    else:
        kept_ixs = list(range(n_source))
    if plain_labels is None:
        # No sheet with BirdNET IDs: the checkpoint's vocabulary names the outputs, and a
        # labels.txt next to the checkpoint adds scientific and common names
        sidecar = None if class_pruning else find_sidecar_labels(checkpoint_path)
        vocab = checkpoint.get('species_vocab') or {}
        names = ([str(vocab['idx_to_species'][i]) for i in kept_ixs]
                 if 'idx_to_species' in vocab else None)
        if names:
            taxonomy_labels = taxonomy_labels_for_ids(names, args.taxonomy)
        if taxonomy_labels is not None:  # the vocabulary already holds BirdNET IDs
            label_ids = names
            plain_labels = list(zip(taxonomy_labels['birdnet_id'], taxonomy_labels['scientific_name'],
                                    taxonomy_labels['common_name']))
        elif sidecar:
            source = read_source_labels(sidecar)
            if len(source) != n_source:
                sys.exit(f'Error: {sidecar} has {len(source)} rows but the model has {n_source} classes')
            if names and list(source['code']) != names:
                row = next(i for i, (a, b) in enumerate(zip(source['code'], names)) if a != b)
                sys.exit(f'Error: {sidecar} does not belong to {checkpoint_path}: row {row} is '
                         f'{source["code"][row]}, the checkpoint has {names[row]}')
            plain_labels = list(source.itertuples(index=False, name=None))
        elif names:
            plain_labels = [(name,) for name in names]
        else:
            print('⚠️  No label source found; the export has no label file')

    inputs = reference_inputs()
    if kept_ixs != list(range(n_source)):
        with torch.no_grad():
            unpruned = module.ExportWrapper(model).eval()(torch.from_numpy(inputs)).numpy()
        module.prune_species(model, kept_ixs)
        with torch.no_grad():
            pruned = module.ExportWrapper(model).eval()(torch.from_numpy(inputs)).numpy()
        if pruned.shape != (len(inputs), len(kept_ixs)) or not np.allclose(
                pruned, unpruned[:, kept_ixs], rtol=0, atol=1e-6):
            sys.exit('Error: the pruned model does not return the selected outputs of the '
                     'checkpoint; prune_species in the model definition does not fit this model')

    clear_model_files(output_dir)
    write_label_files(output_dir, taxonomy_labels, plain_labels)
    write_terms_files(output_dir)

    print(f'Model directory:    {model_dir}')
    print(f'Checkpoint:         {checkpoint_path}')
    print(f'Output directory:   {output_dir}')
    print(f'Output format(s):   {", ".join(sorted(formats))}')
    print(f'Precision(s):       {", ".join(sorted(precisions))}')
    print(f'Batch size:         {args.batch_size} (LiteRT, SavedModel and Core ML)')
    print(f'Classes:            {model.n_species} of {n_source}'
          + (f' selected in {class_pruning}' if class_pruning else ' (class pruning disabled)'))
    print(f'Experimental mode:  {"enabled" if args.experimental_mode else "disabled"}')
    print()

    with torch.no_grad():
        reference = module.ExportWrapper(model).eval()(torch.from_numpy(inputs)).numpy()
    results = Results(reference)

    if 'checkpoint' in formats:
        path = export_checkpoint(model, checkpoint, label_ids, kept_ixs, output_dir, plain_labels)
        reloaded, _ = module.load_checkpoint(path)
        with torch.no_grad():
            output = module.ExportWrapper(reloaded).eval()(torch.from_numpy(inputs)).numpy()
        results.check('Checkpoint', path, output, 'fp32')
    if 'torchscript' in formats:
        export_torchscript(module, model, inputs, results, output_dir, args.torchscript_sigmoid)
    if 'onnx' in formats:
        export_onnx(module, model, inputs, results, output_dir, precisions)
    if 'tflite' in formats or 'tf' in formats:
        export_tf(module, model, inputs, results, output_dir, formats, precisions,
                  args.batch_size, args.experimental_mode)
    if 'coreml' in formats:
        export_coreml(module, model, inputs, results, output_dir, precisions, args.batch_size)

    if not results.rows:
        sys.exit('Error: no model was exported; the selected formats have no model in the '
                 'selected precisions')
    if not results.summary():
        sys.exit('\nError: at least one export differs from the PyTorch model by more than its tolerance')
    print('\nDone.')


if __name__ == '__main__':
    main()
