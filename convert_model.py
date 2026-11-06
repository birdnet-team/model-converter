import argparse
import os
import re
import sys
import numpy as np
import importlib.util
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
import torch
import litert_torch
import onnx
from onnx import helper, numpy_helper
from fix_conv2dsame import fix_conv2dsame_padding
from fix_tflite_dynamic_input import fix_all_for_dynamic_batch, verify_fix


DEFAULT_MODEL_NAME = '05_BiophonyNET_26.10.1' # 01_BirdNET+_V3.0-preview3_Global_11K 03_BirdNET+_V3.0-preview4_Global_12K_EffNetB0 05_BiophonyNET_26.10.1

ALL_FORMATS = ['tf', 'tflite', 'torchscript', 'onnx']
ALL_PRECISIONS = ['fp32', 'fp16', 'int8']

# Class pruning sheet used by default when present in the model directory
DEFAULT_CLASS_PRUNING_FILE = 'labels_pruning.xlsx'

# BirdNET taxonomy used to write labels.csv for kept classes that have a birdnet_id
DEFAULT_TAXONOMY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'BirdNET_Taxonomy_v0.5-Nov2026.csv'
)
TAXONOMY_LABEL_COLUMNS = [
    'birdnet_id', 'scientific_name', 'common_name', 'taxon_group', 'inat_id', 'ebird_code',
]
# Translations are common_name_<lang> columns, e.g. common_name_de, common_name_es_MX, common_name_zh-CN
TRANSLATION_COLUMN = re.compile(r'common_name_[a-z]{2}([_-][A-Za-z]{2})?')
NOT_TRANSLATION_COLUMNS = {'common_name_alt', 'common_name_aliases'}


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'export'
)


# Module-level state (set by _init_config, read by conversion functions)
modelDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', DEFAULT_MODEL_NAME) + '/'

outputDir = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_MODEL_NAME) + '/'
batchSize = 1
segmentDuration = 3.0
precisions = set(ALL_PRECISIONS)  # default: all precisions
experimentalMode = False  # default: experimental mode off
headWeights = None  # default: use model_module.headWeights
model_module = None

classIxsSelection = None  # set from the class pruning sheet; None means keep every class

def _init_config(model_name, output_dir, batch_size, segment_duration, precision_set=None, experimental_mode=False, head_weights=None):
    """Initialise module-level config and load the model definition module."""
    global modelDir, outputDir, batchSize, segmentDuration, model_module
    global precisions, experimentalMode, headWeights

    # A bare name is looked up in models/; anything with a directory part is used as the path it is
    model_arg = model_name.rstrip('/\\')
    if os.path.dirname(model_arg):
        model_path = os.path.abspath(model_arg)
    else:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_arg)

    modelDir = model_path + '/'
    outputDir = os.path.join(output_dir.rstrip('/'), os.path.basename(model_path)) + '/'

    batchSize = batch_size
    segmentDuration = segment_duration
    precisions = precision_set if precision_set else set(ALL_PRECISIONS)
    experimentalMode = experimental_mode

    model_module_path = os.path.join(modelDir, 'model.py')
    if not os.path.isfile(model_module_path):
        sys.exit(f'Error: model definition not found at {model_module_path}')

    module_name = 'model_module'
    spec = importlib.util.spec_from_file_location(module_name, model_module_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = model_module
    spec.loader.exec_module(model_module)


    # Set headWeights after model_module is loaded so we can fall back to its default
    headWeights = head_weights if head_weights is not None else list(model_module.headWeights)

    # Validate and normalize head weights
    if len(headWeights) != 3:
        sys.exit(f'Error: --head-weights must have exactly 3 values, got {len(headWeights)}')
    if any(w < 0.0 for w in headWeights):
        sys.exit(f'Error: --head-weights must be non-negative, got {headWeights}')
    if all(w == 0.0 for w in headWeights):
        sys.exit(f'Error: --head-weights must not all be zero')

    weight_sum = sum(headWeights)
    if abs(weight_sum - 1.0) > 1e-6:
        normalized = [w / weight_sum for w in headWeights]
        print(f'⚠️  Head weights sum to {weight_sum:.4f}, normalizing: '
              f'{headWeights} → {[round(w, 6) for w in normalized]}')
        headWeights = normalized


def _is_experimental():
    """Check whether INT8 export is allowed (requires experimental mode)."""
    return experimentalMode


def createTfPreprocessingModel():
    """Export the mel-spectrogram preprocessing layer as a TF SavedModel and TFLite.

    Builds a Keras model around the model_module.Preprocessing layer, saves it
    as a TF SavedModel, then converts to TFLite.  Supports dynamic input length
    when allow_dynamic_input_length is True (default).

    Output: ``{outputDir}/tf/preprocessing/`` (SavedModel + model.tflite)
    """
    allow_dynamic_input_length = True
    batch_size = batchSize


    
    preprocessing = model_module.Preprocessing()

    
    if allow_dynamic_input_length:
        inputs = keras.Input(shape=(None,), batch_size=batch_size, name='inputs') # dynamic input length
    else:
        segmentDurationInSamples = int(segmentDuration * model_module.sampleRate)
        dtype = np.float32
        sampleVec = np.random.rand(segmentDurationInSamples).astype(dtype)
        input_shape = sampleVec.shape
        inputs = keras.Input(shape=input_shape, batch_size=batch_size, name='inputs') # fixed input length

    x = preprocessing(inputs)

    model = keras.Model(inputs, x)
    model.summary()

    # Save the model as a SavedModel and convert from that
    dstDir = outputDir + 'tf/preprocessing/'
    os.makedirs(dstDir, exist_ok=True)
    model.export(dstDir)
    print(f'Saved model to {dstDir}')

    # Export the model to a TensorFlow Lite file
    converter = tf.lite.TFLiteConverter.from_keras_model(model) # Use keras model

    converter.experimental_new_converter = True
    converter.experimental_enable_resource_variables = True

    #converter.allow_custom_ops = True # ?

    tflite_model = converter.convert()

    dstPath = dstDir + 'model.tflite'
    with open(dstPath, 'wb') as f:
        f.write(tflite_model)
    print('Exported preprocessing tflite model to', dstPath)



def prune_to_selected_classes(model, class_ixs, n_classes):
    """Shrink every class-sized output layer to class_ixs. Shared layers are left alone.

    Output column k is the original class class_ixs[k], in list order. Heads already
    replaced by 1x1 dummies (weight 0) are skipped because their width is not n_classes.
    """
    def _sliced_linear(layer, index):
        sliced = torch.nn.Linear(
            layer.in_features, index.numel(), bias=layer.bias is not None
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            sliced.weight.copy_(layer.weight.index_select(0, index))
            if layer.bias is not None:
                sliced.bias.copy_(layer.bias.index_select(0, index))
        return sliced

    def _sliced_conv1d(layer, index):
        sliced = torch.nn.Conv1d(
            layer.in_channels, index.numel(),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            sliced.weight.copy_(layer.weight.index_select(0, index))
            if layer.bias is not None:
                sliced.bias.copy_(layer.bias.index_select(0, index))
        return sliced

    def _sliced_batchnorm1d(layer, index):
        sliced = torch.nn.BatchNorm1d(
            index.numel(), eps=layer.eps, momentum=layer.momentum,
            affine=layer.affine, track_running_stats=layer.track_running_stats,
        )
        ref = layer.weight if layer.affine else layer.running_mean
        sliced = sliced.to(device=ref.device, dtype=ref.dtype)
        with torch.no_grad():
            if layer.affine:
                sliced.weight.copy_(layer.weight.index_select(0, index))
                sliced.bias.copy_(layer.bias.index_select(0, index))
            if layer.track_running_stats:
                sliced.running_mean.copy_(layer.running_mean.index_select(0, index))
                sliced.running_var.copy_(layer.running_var.index_select(0, index))
                sliced.num_batches_tracked.copy_(layer.num_batches_tracked)
        return sliced

    if not class_ixs:
        return model
    class_ixs = [int(i) for i in class_ixs]
    if len(class_ixs) != len(set(class_ixs)):
        raise ValueError('classIxsSelection contains duplicates')
    if any(i < 0 or i >= n_classes for i in class_ixs):
        raise ValueError(f'classIxsSelection is outside 0..{n_classes - 1}')

    index = torch.tensor(class_ixs, dtype=torch.long)
    replaced = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear) and child.out_features == n_classes:
                setattr(parent, name, _sliced_linear(child, index.to(child.weight.device)))
            elif isinstance(child, torch.nn.Conv1d) and child.out_channels == n_classes and child.groups == 1:
                setattr(parent, name, _sliced_conv1d(child, index.to(child.weight.device)))
            elif isinstance(child, torch.nn.BatchNorm1d) and child.num_features == n_classes:
                ref = child.weight if child.affine else child.running_mean
                setattr(parent, name, _sliced_batchnorm1d(child, index.to(ref.device)))
            else:
                continue
            replaced.append(f'{child.__class__.__name__} {name}')

    if not replaced:
        raise RuntimeError('no class-sized layers found to prune')
    print(f'Pruned classifier to {len(class_ixs)}/{n_classes} classes: {", ".join(replaced)}')
    return model


def loadPytorchModelAndWeights(includePreprocessing=True):

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Load model
    model = model_module.MultiHead02(
        model_module.backbone, 
        model_module.nClasses, 
        pretrained=False, 
        includePreprocessing=includePreprocessing, 
        device=device, 
        headWeights=headWeights
        )


    ## Load weights

    # Get the latest checkpoint file in the model's directory
    checkpointFiles = [f for f in os.listdir(modelDir) if f.endswith('.pth.tar')]
    if not checkpointFiles:
        sys.exit(f'Error: No checkpoint files found in {modelDir}')
    checkpointFiles.sort()  # Sort to get the latest checkpoint (assuming naming convention includes epoch or timestamp)
    checkpointPath = os.path.join(modelDir, checkpointFiles[-1])
    
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.prune_unused_weights()

    # Prune model to selected classes using class indices in classIxsSelection
    if classIxsSelection:
        prune_to_selected_classes(model, classIxsSelection, model_module.nClasses)

    model.eval()

    return model

def get_expected_t_dims():
    """Discover temporal dimension sizes that change with input duration.

    Runs the full model at two durations (1.0 s and ``segmentDuration``) and
    collects every intermediate tensor shape.  Any dimension that differs
    between the two runs is considered time-dependent and returned so that
    ``fix_tflite_dynamic_input`` can patch static RESHAPE/SLICE tensors.

    Returns:
        list[int]: Sorted list of temporal dimension sizes at ``segmentDuration``.
    """
    model = loadPytorchModelAndWeights(includePreprocessing=True)
    device = next(model.parameters()).device
    
    def get_shapes_for_duration(duration):
        frames = int(duration * model_module.sampleRate)
        input_dummy = torch.rand(1, frames).to(device)
        shapes = []
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes.append(tuple(output.shape))
        handles = []
        for name, module in model.named_modules():
            handles.append(module.register_forward_hook(hook_fn))
        with torch.no_grad():
            _ = model(input_dummy)
        for h in handles:
            h.remove()
        return shapes, frames

    # Compare an arbitrary alternate duration (1.0s) against current segmentDuration
    shapes_1, _ = get_shapes_for_duration(1.0)
    shapes_curr, frames_curr = get_shapes_for_duration(segmentDuration)
    
    t_dims = set()
    for s1, sc in zip(shapes_1, shapes_curr):
        if s1 != sc:
            if len(s1) == len(sc):
                for d1, dc in zip(s1, sc):
                    if d1 != dc:
                        t_dims.add(dc)
                        
    # We also add the input seq length explicitly just in case
    t_dims.add(frames_curr)
    if 1 in t_dims:
        t_dims.remove(1)
        
    return sorted(list(t_dims))


def createTfMainModel(saveKerasModel=False):
    """Convert the classifier to a TF SavedModel and TFLite (spectrogram input).

    Steps:
        1. Convert the PyTorch classifier to a static TF SavedModel via litert_torch.
        2. Re-wrap the SavedModel with a dynamic temporal dimension.
        3. Export to TFLite (fp32, and optionally fp16/int8) and fix static ops.

    Args:
        saveKerasModel: If True, also save a ``.keras`` file.

    Output: ``{outputDir}/tf/main/`` (SavedModel + TFLite variants)
    """
    expected_t_dims = get_expected_t_dims()
    model = loadPytorchModelAndWeights(includePreprocessing=False)

    imageWidth = int(model_module.imageWidthResizeFactor * segmentDuration * model_module.sampleRate / model_module.fftHopSizeInSamples)


    inputShape = (batchSize, 3, model_module.imageHeight, imageWidth)
    dtype = np.float32
    testInputNp = np.random.rand(inputShape[0],inputShape[1],inputShape[2],inputShape[3]).astype(dtype)
    testInputTorch = torch.from_numpy(testInputNp)
    sample_inputs = (testInputTorch,)

    print(f"Sample input shape: {testInputTorch.shape}")

    dstDir = outputDir + 'tf/main/'
    os.makedirs(dstDir, exist_ok=True)

    staticDir = dstDir + '_static/'
    os.makedirs(staticDir, exist_ok=True)


    ## Step 1: Convert with static shapes via litert_torch

    print("Step 1: Converting PyTorch model to static TF SavedModel...")

    # # Define a simple wrapper to strip out the embeddings so the resulting
    # # TFLite model keeps the same single-output signature as before.
    # class _SingleOutputWrapper(torch.nn.Module):
    #     def __init__(self, core_model):
    #         super().__init__()
    #         self.core_model = core_model

    #     def forward(self, x):
    #         preds, embs = self.core_model(x)
    #         return preds

    # single_output_model = _SingleOutputWrapper(model)
    # single_output_model.eval()

    edge_model = litert_torch.convert(
        model,
        #single_output_model,
        sample_inputs,
        _saved_model_dir=staticDir,
    )

    print("  Static conversion complete.")



    ## Step 2: Re-wrap the SavedModel with dynamic input signatures

    print("Step 2: Re-wrapping SavedModel with dynamic input signatures...")

    static_model = tf.saved_model.load(staticDir)
    static_fn = static_model.signatures['serving_default']

    input_keys = list(static_fn.structured_input_signature[1].keys())
    print(f"  Static signature input keys: {input_keys}")

    output_keys = list(static_fn.structured_outputs.keys())
    print(f"  Static signature output keys: {output_keys}")

    class DynamicWrapper(tf.Module):
        def __init__(self, static_model, static_fn):
            super().__init__()
            self._static_model = static_model
            self._fn = static_fn

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[batchSize, 3, model_module.imageHeight, None], dtype=tf.float32)
        ])
        def serve(self, x):
            result = self._fn(**{input_keys[0]: x})
            #return result[output_keys[0]]
            #return result
            # Map the generic litert_torch output names to explicit dict keys
            return {
                'predictions': result['output_0'],
                'embeddings': result['output_1']
            }


    wrapper = DynamicWrapper(static_model, static_fn)

    # Trace with concrete input (may fail for StableHLO/XLA ops from litert_torch)
    test_input_tf = tf.constant(testInputNp)
    try:
        _ = wrapper.serve(test_input_tf)
        print("  Dynamic wrapper test passed.")
    except Exception as e:
        print(f"  Warning: Dynamic wrapper test skipped (XLA ops can't run eagerly): {type(e).__name__}")
        print(f"  This is expected — TFLite MLIR converter will lower these ops correctly.")

    # Save with dynamic signature
    tf.saved_model.save(
        wrapper,
        dstDir,
        signatures={'serving_default': wrapper.serve}
    )

    print(f"  Dynamic SavedModel saved to {dstDir}")

    # Verify
    loaded = tf.saved_model.load(dstDir)
    sig = loaded.signatures['serving_default']
    input_specs = sig.structured_input_signature
    if input_specs and len(input_specs) > 1 and input_specs[1]:
        for name, spec in input_specs[1].items():
            dynamic_dims = [i for i, s in enumerate(spec.shape) if s is None]
            print(f"  SavedModel input '{name}': shape={spec.shape} "
                  f"{'✅ dynamic: ' + str(dynamic_dims) if dynamic_dims else '⚠️ static'}")


    ## Step 3: Convert dynamic SavedModel to TFLite and fix static Reshape ops

    def _convert_and_fix_tflite(saved_model_dir, tflite_path,
                                optimize=False, representative_dataset=None,
                                supported_types=None):
        """Convert SavedModel to TFLite, then fix static Reshape for dynamic batch."""
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if supported_types:
            converter.target_spec.supported_types = supported_types
        if representative_dataset:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  Exported TFLite model to {tflite_path}")

        print(f"  Fixing static ops for dynamic batch...")
        audio_length = int(segmentDuration * model_module.sampleRate)
        n_fixed = fix_all_for_dynamic_batch(tflite_path, image_width=imageWidth, audio_length=audio_length, expected_t_dims=expected_t_dims)
        if n_fixed > 0:
            verify_fix(tflite_path, model_module.imageHeight, imageWidth)

    print("Step 3: Converting and fixing TFLite models...")

    if 'fp32' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model.tflite')
    if 'fp16' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model_fp16.tflite',
                                optimize=True, supported_types=[tf.float16])
    if 'int8' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model_int8.tflite', optimize=True) # Dynamic range quantization (weights only)


    ## Cleanup
    import shutil
    if os.path.exists(staticDir):
        shutil.rmtree(staticDir)
        print(f"Cleaned up {staticDir}")


    ## Save as keras model
    if saveKerasModel:
        dstPath = dstDir + 'model.keras'
        saved_model = tf.keras.layers.TFSMLayer(dstDir, call_endpoint='serving_default')
        tf.keras.models.save_model(saved_model, dstPath, overwrite=True)
        print('Exported keras model to', dstPath)




def combineTfModels():
    """Chain the preprocessing and main TF SavedModels into an end-to-end model.

    Loads the previously exported preprocessing and main SavedModels, wraps them
    in a single ``tf.Module`` that accepts raw audio ``(batch, samples)`` and
    returns predictions, exports as a TF SavedModel and TFLite variants.

    Output: ``{outputDir}/tf/combined/`` (SavedModel + TFLite variants)
    """
    expected_t_dims = get_expected_t_dims()
    imageWidth = int(model_module.imageWidthResizeFactor * segmentDuration * model_module.sampleRate / model_module.fftHopSizeInSamples)

    # --------------------------------------------------------------------------

    class CombinedModel(tf.Module):
        def __init__(self, pre, main):
            super().__init__()
            self.pre = pre
            self.main = main

        @tf.function(input_signature=[tf.TensorSpec(shape=[batchSize, None], dtype=tf.float32)])
        def __call__(self, x):
            pre_out = self.pre.signatures['serving_default'](x)['output_0']
            # reshape pre_out to match input shape of main model: (B, 224, W, 3) --> (B, 3, 224, W)
            pre_out = tf.transpose(pre_out, perm=[0, 3, 1, 2])
            #main_out = self.main.signatures['serving_default'](pre_out)['output_0']
            main_out = self.main.signatures['serving_default'](pre_out)
            #return main_out
            return {
                'predictions': main_out['predictions'],
                'embeddings': main_out['embeddings']
            }


    path_pre = outputDir + 'tf/preprocessing/'
    path_main = outputDir + 'tf/main/'

    pre = tf.saved_model.load(path_pre)
    main = tf.saved_model.load(path_main)

    ## Export the combined model as a SavedModel
    combined_model = CombinedModel(pre, main)
    dstDir = outputDir + 'tf/combined/'
    tf.saved_model.save(
        combined_model, dstDir, 
        signatures=combined_model.__call__.get_concrete_function()
    )
    print(f'Combined model saved to {dstDir}')

    ## Convert to TFLite and fix static Reshape ops for dynamic batch

    def _convert_and_fix_tflite(saved_model_dir, tflite_path,
                                optimize=False, representative_dataset=None,
                                supported_types=None):
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if supported_types:
            converter.target_spec.supported_types = supported_types
        if representative_dataset:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f'  Exported combined TFLite model to {tflite_path}')

        print(f'  Fixing static ops for dynamic batch...')
        audio_length = int(segmentDuration * model_module.sampleRate)
        n_fixed = fix_all_for_dynamic_batch(tflite_path, image_width=imageWidth, audio_length=audio_length, expected_t_dims=expected_t_dims)
        if n_fixed > 0:
            verify_fix(tflite_path, model_module.imageHeight, imageWidth)


    if 'fp32' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model.tflite')
    if 'fp16' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model_fp16.tflite',
                                optimize=True, supported_types=[tf.float16])
    if 'int8' in precisions:
        
        if _is_experimental():
            _convert_and_fix_tflite(dstDir, dstDir + 'model_int8.tflite', optimize=True) # Dynamic range quantization (weights only)
        else:
            print("  ⚠️  Skipping TFLite INT8 (combined) — this is only supported in experimental mode. Use --experimental-mode to enable.")



class _Fp32StageWrapper(torch.nn.Module):
    """Adapter that accepts a float16 tensor, runs a module in float32,
    and returns float32 output.

    Used to keep EfficientNetV2 stage 5 in float32 within an otherwise
    float16 backbone.  Some stage-5 expansion conv outputs exceed the fp16
    maximum (~65 504) even in float32 inference (e.g. 74 222 for block 5.13),
    so fp16 computation would saturate to Inf → NaN cascade.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x.float())


def _fold_bn_for_fp16(model: torch.nn.Module) -> int:
    """Fold large-variance BN channels into weight/bias so they are fp16-safe.

    Some EfficientNetV2 stage-5 BN layers have running_var up to 8.7 M.  In
    fp16 that becomes Inf (> 65504), which breaks BN normalisation:
    ``(x − mean) / sqrt(Inf) → 0``  →  output collapses to beta for those
    channels  →  activation drift across residual connections  →  fp16 Conv
    overflow  →  ``Inf / Inf = NaN`` cascade.

    Fix: absorb (mean, var) into (weight, bias) so the effective
    running_mean becomes 0 and running_var becomes 1 for every unsafe channel.
    The transformation is mathematically exact::

        original:  weight * (x − mean) / sqrt(var + eps) + bias
        after:     weight_new * x + bias_new
          where  weight_new = weight / sqrt(var + eps)
                 bias_new   = bias − weight * mean / sqrt(var + eps)
                 running_mean = 0,  running_var = 1 − eps
    """
    FP16_MAX = 65504.0
    n_folded = 0
    for name, m in model.named_modules():
        if not isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.BatchNorm3d)):
            continue
        if m.running_var is None or not m.track_running_stats:
            continue
        if m.weight is None or m.bias is None:
            continue
        unsafe = m.running_var > FP16_MAX
        if not unsafe.any():
            continue
        n_ch = int(unsafe.sum().item())
        print(f'  fold_bn_fp16: {name} — {n_ch} channel(s) with var > {FP16_MAX:.0f}')
        eps = m.eps
        with torch.no_grad():
            std   = torch.sqrt(m.running_var[unsafe] + eps)
            gamma = m.weight.data[unsafe].clone()
            beta  = m.bias.data[unsafe].clone()
            mean  = m.running_mean[unsafe].clone()
            m.weight.data[unsafe]   = gamma / std
            m.bias.data[unsafe]     = beta - gamma * mean / std
            m.running_mean[unsafe]  = 0.0
            m.running_var[unsafe]   = 1.0 - eps   # sqrt(1-eps+eps) = 1.0
        n_folded += n_ch
    return n_folded


def _find_preprocessing_node_names(fp32_model) -> list:
    """Return the names of every ONNX node that belongs to the preprocessing
    sub-graph (everything before the first backbone Conv).

    Strategy: BFS backward from the activation input of the first Conv whose
    weight has shape [out_ch, 3, kH, kW]  (the EfficientNetV2 stem that takes
    the 3-channel normalised spectrogram).  Every node visited along the way
    is a preprocessing node.
    """
    graph = fp32_model.graph

    # output_name → node that produces it
    producer = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    initializer_names = {init.name for init in graph.initializer}
    input_names       = {inp.name  for inp  in graph.input}

    # Find the backbone stem: first Conv with weight shape [out_ch, 3, kH, kW]
    init_shapes = {init.name: list(init.dims) for init in graph.initializer}
    stem_input = None
    for node in graph.node:
        if node.op_type == 'Conv' and len(node.input) >= 2:
            w = node.input[1]
            if w in init_shapes:
                shape = init_shapes[w]
                if len(shape) == 4 and shape[1] == 3:
                    stem_input = node.input[0]
                    break

    if stem_input is None:
        print('  _find_preprocessing_node_names: stem Conv not found, falling back to op_block_list')
        return []

    # BFS backward from stem_input; collect non-empty node names
    preproc_names = []
    visited = set()
    queue   = [stem_input]
    while queue:
        name = queue.pop()
        if name in visited or name in initializer_names or name in input_names:
            continue
        visited.add(name)
        if name in producer:
            node = producer[name]
            if node.name:          # skip unnamed nodes (they stay fp32 by default)
                preproc_names.append(node.name)
            for inp in node.input:
                if inp:
                    queue.append(inp)

    return preproc_names


def createTorchScriptModel():
    """Export the full model (with preprocessing) to TorchScript.

    Produces one ``.pt`` file per requested precision:
        - fp32: ``torch.jit.script``
        - fp16: ``model.half()`` then ``torch.jit.script``

    INT8 is not supported for TorchScript — ``quantize_dynamic`` only covers
    Linear/Conv1d layers, leaving the EfficientNet Conv2d backbone untouched
    (460 MB vs 517 MB fp32).

    Output: ``{outputDir}/torchscript/model[_fp16].pt``
    """
    # Load model and weights
    model = loadPytorchModelAndWeights(includePreprocessing=True)
    
    dstDir = outputDir + 'torchscript/'
    os.makedirs(dstDir, exist_ok=True)

    if 'fp32' in precisions:
        
        ## Export CPU version
        model = model.to('cpu')
        model.eval()

        scripted_module = torch.jit.script(model)
        scripted_module.save(dstDir + 'model.pt')
        print('TorchScript model saved to', dstDir + 'model.pt')

    if 'fp16' in precisions:

        if _is_experimental():

            import copy
            from typing import Tuple

            # STFT (inside PreprocessingLayerTorch) does not support float16:
            # calling model.half() converts the spectrogram transform buffers too,
            # making STFT produce ComplexHalf tensors that downstream ops can't handle.
            # Fix: keep preprocessing in float32; convert only the backbone+head to
            # float16; insert an explicit .half() cast in between.

            # Build a backbone copy whose preprocess is replaced with Identity so
            # it can be converted to float16 without touching the STFT layer.
            fp16_backbone = copy.deepcopy(model)
            fp16_backbone.preprocess = torch.nn.Identity()
            # Fold large-variance BN channels BEFORE converting to fp16.
            # Several EfficientNetV2 stage-5 BN layers have running_var up to 8.7 M;
            # converting them to fp16 gives Inf which cascades into NaN.
            n_folded = _fold_bn_for_fp16(fp16_backbone)
            print(f'  TS FP16: folded {n_folded} BN channel(s) with var > 65504')
            fp16_backbone = fp16_backbone.half()

            # Stage 5 of EfficientNetV2 MUST stay in float32.
            # The expansion convolutions in blocks 5.12–5.14 have large weights
            # (max ≈ 4.4) and the residual stream reaches > 800 in float32 by
            # stage 5.  The resulting dot products exceed the fp16 maximum
            # (e.g. 74 222 for block 5.13) → fp16 output is Inf → PyTorch BN
            # with Inf running_var on Inf input gives Inf/Inf = NaN cascade.
            #
            # Fix: wrap stage 5 with _Fp32StageWrapper so it accepts fp16 input
            # from stages 1–4, casts to fp32 internally, and returns fp32.  All
            # downstream modules (conv_head, bn2, act2, classifiers) also stay
            # in fp32 (reverted by .float() after the global .half()).
            enc = fp16_backbone.encoder
            stage5 = enc.blocks[5]
            stage5.float()                                   # revert to fp32
            enc.blocks[5] = _Fp32StageWrapper(stage5)        # fp16-in / fp32-out
            # Revert post-stage-5 encoder modules and all classifiers to fp32
            for attr in ('conv_head', 'bn2', 'act2', 'norm'):
                m = getattr(enc, attr, None)
                if m is not None:
                    m.float()
            for attr in ('global_pool', 'head', 'fc1', 'fc2',
                        'att_block', 'att_block2', 'dropouts'):
                m = getattr(fp16_backbone, attr, None)
                if m is not None:
                    m.float()

            # Revert the avg_kernel buffer used in pooling back to float32
            if hasattr(fp16_backbone, '_avg_kernel'):
                fp16_backbone._avg_kernel = fp16_backbone._avg_kernel.float()


            class _TS_FP16Wrapper(torch.nn.Module):
                """Accepts float32 audio; runs preprocessing in float32, then casts
                spectrogram to float16 for backbone stages 1–4, then returns to
                float32 for stage 5 and all classifiers. Returns float32 outputs."""
                def __init__(self, preprocess: torch.nn.Module,
                            backbone: torch.nn.Module) -> None:
                    super().__init__()
                    self.preprocess = preprocess   # float32 — preserves STFT
                    self.backbone = backbone       # mixed: fp16 (st1-4) + fp32 (st5+)

                def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    spec = self.preprocess(x)          # float32 audio → float32 spec
                    #emb, pred = self.backbone(spec.half())  # fp16 entry; fp32 exit
                    #return emb, pred
                    pred, emb = self.backbone(spec.half())  # <-- Flipped order
                    return pred, emb # <-- Flipped order

            wrapper = _TS_FP16Wrapper(model.preprocess, fp16_backbone)
            scripted_fp16 = torch.jit.script(wrapper)
            scripted_fp16.save(dstDir + 'model_fp16.pt')
            print('TorchScript FP16 model saved to', dstDir + 'model_fp16.pt')
        else:
            print("  ⚠️  Skipping TorchScript FP16 — this is only supported in experimental mode. Use --experimental-mode to enable.")



def patch_onnx_resize_to_dynamic(model_path, output_path,
                                  imageWidthResizeFactor=1.5,
                                  fftHopSizeInSamples=512):
    """Patch the ONNX model to replace baked-in Resize output sizes with
    a dynamic computation based on the input 'samples' dimension.
    
    The dynamo exporter produces a Resize node whose sizes input is:
        Concat([batch, channels], [224, 281])
    where [224, 281] is a constant initializer.
    
    We replace [224, 281] with [imageHeight, dynamic_imageWidth] where:
        dynamic_imageWidth = (imageWidthResizeFactor_num * samples) 
                             // (imageWidthResizeFactor_den * fftHopSizeInSamples)
    
    imageWidthResizeFactor is a float (e.g. 1.5) which is converted to a
    rational fraction (num/den) so that the computation stays in int64.
    """
    from fractions import Fraction
    frac = Fraction(imageWidthResizeFactor).limit_denominator(1000)
    imageWidthResizeFactor_num = frac.numerator
    imageWidthResizeFactor_den = frac.denominator
    print(f"imageWidthResizeFactor={imageWidthResizeFactor} -> {imageWidthResizeFactor_num}/{imageWidthResizeFactor_den}")
    
    model = onnx.load(model_path)
    graph = model.graph
    
    # Build initializer lookup
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)
    
    # Build output -> node map
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node
    
    # Compute expected baked-in values
    dummy_samples = int(segmentDuration * model_module.sampleRate)
    expected_imageWidth = int(imageWidthResizeFactor_num * dummy_samples // (imageWidthResizeFactor_den * fftHopSizeInSamples))
    expected_imageHeight = model_module.imageHeight
    
    print(f"Looking for baked-in Resize sizes: [{expected_imageHeight}, {expected_imageWidth}]")
    
    # Strategy: Find the initializer containing [224, 281] by searching all initializers,
    # then trace forward to find which Concat node uses it, then verify that Concat
    # feeds into a Resize node.
    
    target_init_name = None
    target_concat_node = None
    target_resize_node = None
    concat_input_idx = None
    
    for init_name, init_val in initializers.items():
        flat = init_val.flatten().tolist()
        if len(flat) >= 2 and expected_imageHeight in [int(v) for v in flat] and expected_imageWidth in [int(v) for v in flat]:
            print(f"Found candidate initializer '{init_name}': {init_val.flatten()}")
            
            # Find which node(s) consume this initializer
            for node in graph.node:
                if init_name in node.input:
                    if node.op_type == 'Concat':
                        # Verify this Concat feeds into a Resize
                        concat_output = node.output[0]
                        for resize_candidate in graph.node:
                            if resize_candidate.op_type == 'Resize' and concat_output in resize_candidate.input:
                                target_init_name = init_name
                                target_concat_node = node
                                target_resize_node = resize_candidate
                                concat_input_idx = list(node.input).index(init_name)
                                print(f"  -> feeds into Concat '{node.name}' at input[{concat_input_idx}]")
                                print(f"  -> which feeds into Resize '{resize_candidate.name}'")
                                break
                    elif node.op_type == 'Resize':
                        # Direct connection (legacy exporter style)
                        target_init_name = init_name
                        target_resize_node = node
                        print(f"  -> directly feeds into Resize '{node.name}'")
                        break
                if target_resize_node:
                    break
        if target_resize_node:
            break
    
    if target_resize_node is None:
        print("WARNING: Could not find Resize node with baked-in sizes. Model may already be dynamic.")
        onnx.save(model, output_path, save_as_external_data=False)
        return
    
    # ========================================================================
    # Build the dynamic imageWidth computation subgraph
    # ========================================================================
    
    model_input_name = graph.input[0].name  # 'input'
    prefix = "dynamic_resize_patch"
    
    new_nodes = []
    new_initializers = []
    
    # 1. Get input shape: Shape(input) -> [batch, samples]
    input_shape_name = f"{prefix}/input_shape"
    new_nodes.append(helper.make_node("Shape", [model_input_name], [input_shape_name]))
    
    # 2. Extract samples dimension: Gather(input_shape, 1) -> samples (scalar)
    samples_dim_name = f"{prefix}/samples_dim"
    gather_idx_name = f"{prefix}/gather_idx"
    new_initializers.append(numpy_helper.from_array(
        np.array(1, dtype=np.int64), name=gather_idx_name))
    new_nodes.append(helper.make_node("Gather",
        [input_shape_name, gather_idx_name], [samples_dim_name], axis=0))
    
    # 3. Reshape samples to [1] for arithmetic
    samples_1d_name = f"{prefix}/samples_1d"
    one_shape_name = f"{prefix}/one_shape"
    new_initializers.append(numpy_helper.from_array(
        np.array([1], dtype=np.int64), name=one_shape_name))
    new_nodes.append(helper.make_node("Reshape",
        [samples_dim_name, one_shape_name], [samples_1d_name]))
    
    # 4. numerator = imageWidthResizeFactor_num * samples
    numerator_name = f"{prefix}/numerator"
    num_factor_name = f"{prefix}/num_factor"
    new_initializers.append(numpy_helper.from_array(
        np.array([imageWidthResizeFactor_num], dtype=np.int64), name=num_factor_name))
    new_nodes.append(helper.make_node("Mul",
        [samples_1d_name, num_factor_name], [numerator_name]))
    
    # 5. denominator = imageWidthResizeFactor_den * fftHopSizeInSamples
    denom_val = imageWidthResizeFactor_den * fftHopSizeInSamples
    denom_name = f"{prefix}/denom"
    new_initializers.append(numpy_helper.from_array(
        np.array([denom_val], dtype=np.int64), name=denom_name))
    
    # 6. imageWidth = numerator // denominator (integer floor division)
    image_width_name = f"{prefix}/image_width"
    new_nodes.append(helper.make_node("Div",
        [numerator_name, denom_name], [image_width_name]))
    
    # 7. Build replacement: [imageHeight, imageWidth]
    image_height_name = f"{prefix}/image_height"
    new_initializers.append(numpy_helper.from_array(
        np.array([expected_imageHeight], dtype=np.int64), name=image_height_name))
    
    dynamic_hw_name = f"{prefix}/dynamic_hw"
    new_nodes.append(helper.make_node("Concat",
        [image_height_name, image_width_name], [dynamic_hw_name], axis=0))
    
    # ========================================================================
    # Wire it up
    # ========================================================================
    
    if target_concat_node is not None:
        # Dynamo style: replace the input to the Concat node
        target_concat_node.input[concat_input_idx] = dynamic_hw_name
        print(f"  Replaced Concat input[{concat_input_idx}] '{target_init_name}' -> '{dynamic_hw_name}'")
    else:
        # Legacy style: replace the input to the Resize node directly
        for i, inp_name in enumerate(target_resize_node.input):
            if inp_name == target_init_name:
                target_resize_node.input[i] = dynamic_hw_name
                print(f"  Replaced Resize input[{i}] '{target_init_name}' -> '{dynamic_hw_name}'")
                break
    
    # Remove old initializer
    for init in list(graph.initializer):
        if init.name == target_init_name:
            graph.initializer.remove(init)
            print(f"  Removed old initializer '{target_init_name}'")
            break
    
    # Insert new nodes before the target (Concat or Resize)
    target_node = target_concat_node if target_concat_node else target_resize_node
    insert_idx = list(graph.node).index(target_node)
    for i, node in enumerate(new_nodes):
        graph.node.insert(insert_idx + i, node)
    
    for init in new_initializers:
        graph.initializer.append(init)
    
    # Validate
    print("\nValidating patched model...")
    try:
        onnx.checker.check_model(model)
        print("  ✅ Model is valid")
    except Exception as e:
        print(f"  ⚠️  Validation warning: {e}")
    
    # Save
    onnx.save(model, output_path, save_as_external_data=False)
    print(f"\nPatched model saved to {output_path}")



def createOnnxModel():
    """Export the full model (with preprocessing) to ONNX.

    Uses ``torch.onnx.export`` with ``dynamo=True`` and dynamic shapes for
    batch and temporal axes, then post-processes the graph:
        1. Patch Resize nodes to compute output sizes dynamically.
        2. Fix Conv2dSame nodes with baked static padding.
        3. Optionally produce fp16 (onnxconverter-common) and int8
           (onnxruntime dynamic quantization) variants.

    Output: ``{outputDir}/onnx/model[_fp16|_int8].onnx``
    """
    # Load model and weights
    model = loadPytorchModelAndWeights(includePreprocessing=True)

    # Pre-fold dangerous BN variances so they don't blow up the ONNX FP16 conversion Cast
    _fold_bn_for_fp16(model)

    # Create dummy input (but model should be able to handle dynamic batch size and input length, so this is just for export)
    batch_size = 4
    inputBatch = torch.rand(batch_size, int(segmentDuration*model_module.sampleRate)).to(next(model.parameters()).device)

    dstDir = outputDir + 'onnx/'
    os.makedirs(dstDir, exist_ok=True)

    torch.onnx.export(
        model,
        inputBatch, 
        dstDir+'temp.onnx', 
        verbose=True, 
        input_names=['input'],
        #output_names=['embeddings', 'predictions'],
        output_names=['predictions', 'embeddings'],
        opset_version=18,
        dynamo=True,
        dynamic_shapes={'input': {0: 'batch_size', 1: 'samples'}}
    )

    # Load model and save without external data format
    onnxModel = onnx.load(dstDir+'temp.onnx')
    onnx.save(onnxModel, dstDir+'model_static_resize.onnx', save_as_external_data=False)

    print('ONNX model (static resize) exported to', dstDir+'model_static_resize.onnx')

    # Patch Resize node to be dynamic
    patch_onnx_resize_to_dynamic(
        dstDir+'model_static_resize.onnx',
        dstDir+'model.onnx',
        imageWidthResizeFactor=model_module.imageWidthResizeFactor,
        fftHopSizeInSamples=model_module.fftHopSizeInSamples
    )

    print('ONNX model (dynamic resize) exported to', dstDir+'model.onnx')

    # Fix Conv2dSame baked padding for dynamic input sizes
    onnx_path = dstDir + 'model.onnx'
    onnx_model = onnx.load(onnx_path)
    n_fixed = fix_conv2dsame_padding(onnx_model)
    if n_fixed > 0:
        onnx.save(onnx_model, onnx_path, save_as_external_data=False)
        print(f'Fixed {n_fixed} Conv2dSame node(s) with baked padding in {onnx_path}')
    else:
        print('No Conv2dSame padding issues found (model may not use Conv2dSame)')

    # Cleanup temp files
    if os.path.exists(dstDir+'temp.onnx'):
        os.remove(dstDir+'temp.onnx')
    if os.path.exists(dstDir+'temp.onnx.data'):
        os.remove(dstDir+'temp.onnx.data')
    if os.path.exists(dstDir+'model_static_resize.onnx'):
        os.remove(dstDir+'model_static_resize.onnx')

    # FP16 variant — post-process the fp32 ONNX graph with onnxconverter_common.
    #
    # Direct re-export via dynamo hangs: the STFT's complex symbolic-shape
    # analysis (Eq / inequality guards over s27, s77) causes the torch.export
    # tracing loop to never terminate.
    #
    # Post-processing pitfall: any graph output that is also consumed as a node
    # input becomes type-mismatched after keep_io_types=True inserts a Cast —
    # the Cast output overwrites the original name for all consumers.  Step 1
    # decouples these dual-role tensors before conversion.
    if 'fp16' in precisions:
        from onnxconverter_common import float16

        fp32_model = onnx.load(onnx_path)

        # ── Step 1: decouple dual-role output tensors ─────────────────────
        # Any graph output whose name also appears as a node input is dual-role.
        # For each such tensor, insert an Identity node and redirect the graph
        # output to the Identity's output, leaving the original name purely
        # internal so it is converted to fp16 without type-mismatch.
        internal_inputs = set()
        for node in fp32_model.graph.node:
            for inp in node.input:
                if inp:
                    internal_inputs.add(inp)

        for out in list(fp32_model.graph.output):
            if out.name in internal_inputs:
                new_name = out.name + '_out'
                identity = helper.make_node('Identity', [out.name], [new_name],
                                            name=f'_fp16_decoupler_{out.name}')
                fp32_model.graph.node.append(identity)
                out.name = new_name
                print(f'  FP16 prep: decoupled graph output → {new_name}')

        # ── Step 2: shape inference so the converter has full type context ─
        # Do NOT clear value_info before this: the converter needs it to
        # identify fp32/fp16 boundaries and insert Cast nodes correctly.
        try:
            fp32_model = onnx.shape_inference.infer_shapes(fp32_model)
        except Exception as e:
            print(f'  Warning: shape inference failed ({e}), proceeding without it')

        # ── Step 3: convert backbone to fp16, keep ALL preprocessing in fp32 ──
        #
        # node_block_list keeps every preprocessing node in fp32 by name.
        # BFS backward from the backbone stem Conv (first Conv with a 3-channel
        # weight) discovers all ~82 preprocessing nodes.  op_block_list is
        # also passed as a belt-and-suspenders safety net — both exclusion
        # lists are additive.  Using only op_block_list is insufficient because
        # it acts on op types globally and misses the mel filterbank MatMul.
        preproc_nodes = _find_preprocessing_node_names(fp32_model)
        if preproc_nodes:
            print(f'  ONNX FP16: blocking {len(preproc_nodes)} preprocessing nodes in fp32')
        else:
            print('  ONNX FP16: stem Conv not found; using op_block_list only')
        fp16_model = float16.convert_float_to_float16(
            fp32_model,
            keep_io_types=True,
            op_block_list=['STFT', 'Log', 'Pow', 'Exp', 'GlobalAveragePool', 'ReduceMean', 'AveragePool', 'ConstantOfShape'],
            node_block_list=preproc_nodes or None,
        )

        fp16_path = dstDir + 'model_fp16.onnx'
        onnx.save(fp16_model, fp16_path, save_as_external_data=False)
        print(f'ONNX FP16 model exported to {fp16_path}')


    # INT8 variant (dynamic quantization)
    if 'int8' in precisions:
        if _is_experimental():
            from onnxruntime.quantization import quantize_dynamic, QuantType
            model_for_quant = onnx.load(onnx_path)
            # Clear stale shape annotations left by Resize/Conv2dSame patches, then
            # re-derive them via shape inference so quantize_dynamic has correct
            # type/shape context for every intermediate tensor.  Without this,
            # missing value_info causes quantize_dynamic to misclassify tensor types.
            del model_for_quant.graph.value_info[:]
            try:
                model_for_quant = onnx.shape_inference.infer_shapes(model_for_quant)
            except Exception as e:
                print(f'  Warning: shape inference failed ({e}), proceeding without it')
            # Exclude preprocessing nodes from quantization — same reason as FP16:
            # the mel filterbank is a MatMul whose weights must not be quantized to
            # INT8 (it corrupts the spectrogram and collapses predictions).
            preproc_nodes = _find_preprocessing_node_names(model_for_quant)
            if preproc_nodes:
                print(f'  ONNX INT8: excluding {len(preproc_nodes)} preprocessing nodes from quantization')
            else:
                print('  ONNX INT8: stem Conv not found; quantizing all nodes')
            temp_quant_path = dstDir + 'model_pre_quant.onnx'
            onnx.save(model_for_quant, temp_quant_path, save_as_external_data=False)
            int8_path = dstDir + 'model_int8.onnx'
            quantize_dynamic(
                temp_quant_path, int8_path,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=preproc_nodes or None,
            )
            os.remove(temp_quant_path)
            print(f'ONNX INT8 (dynamic quantization) model exported to {int8_path}')
        else:
            print("  ⚠️  Skipping ONNX INT8 — this is only supported in experimental mode. Use --experimental-mode to enable.")

###########################################################################################
###########################################################################################
###########################################################################################


def load_class_ixs_from_excel(path):
    """Return ix values for rows whose selected column is 1, in sheet order."""
    if not os.path.isfile(path):
        print(f'⚠️  Class pruning file not found: {path}')
        sys.exit(1)

    import pandas as pd
    try:
        df = pd.read_excel(path)
    except ImportError:
        sys.exit('Error: openpyxl is required to read --class-pruning Excel files')

    missing = [name for name in ('ix', 'selected') if name not in df.columns]
    if missing:
        sys.exit(f'Error: {path} is missing column(s): {", ".join(missing)}')

    selected = pd.to_numeric(df['selected'], errors='coerce')
    ixs = pd.to_numeric(df.loc[selected == 1, 'ix'], errors='coerce')
    if ixs.isna().any():
        sys.exit(f'Error: {path} has a non-numeric ix in a selected row')
    class_ixs = [int(i) for i in ixs.tolist()]
    if not class_ixs:
        sys.exit(f'Error: {path} has no rows with selected == 1')
    return class_ixs


def read_pruned_labels(src_path):
    """Return the selected rows with ix renumbered to the exported output order."""
    import pandas as pd
    df = pd.read_excel(src_path)
    selected = pd.to_numeric(df['selected'], errors='coerce')
    labels = df.loc[selected == 1].copy()
    labels['ix'] = range(len(labels))
    return labels


def build_taxonomy_labels(pruned_labels, taxonomy_path):
    """Return one row per model output with names from the BirdNET taxonomy.

    pruned_labels are the kept rows in output order; their birdnet_id column links each
    output to the taxonomy. Returns None when the sheet has no birdnet_id column, and
    exits if an id is missing, repeated, or unknown.
    """
    import pandas as pd
    if 'birdnet_id' not in pruned_labels.columns:
        return None
    if not os.path.isfile(taxonomy_path):
        sys.exit(f'Error: BirdNET taxonomy not found: {taxonomy_path}')
    with open(taxonomy_path, encoding='utf-8', errors='replace') as f:
        if f.readline().startswith('version https://git-lfs'):
            sys.exit(f'Error: {taxonomy_path} is a Git LFS pointer; run `git lfs pull` to fetch it')

    ids = pruned_labels['birdnet_id'].fillna('').astype(str).str.strip().reset_index(drop=True)
    if (ids == '').any():
        sys.exit(f'Error: {(ids == "").sum()} kept class(es) have no birdnet_id')
    repeated = ids[ids.duplicated()].unique()
    if len(repeated):
        sys.exit(f'Error: birdnet_id used by more than one kept class: {", ".join(repeated[:10])}')

    taxonomy = pd.read_csv(taxonomy_path, dtype=str, keep_default_na=False)
    missing = [c for c in TAXONOMY_LABEL_COLUMNS if c not in taxonomy.columns]
    if missing:
        sys.exit(f'Error: {taxonomy_path} is missing column(s): {", ".join(missing)}')
    if not taxonomy['birdnet_id'].is_unique:
        sys.exit(f'Error: {taxonomy_path} has repeated birdnet_id values')
    unknown = ids[~ids.isin(taxonomy['birdnet_id'])]
    if len(unknown):
        sys.exit(f'Error: {len(unknown)} birdnet_id(s) not in {taxonomy_path}: '
                 f'{", ".join(unknown[:10])}')

    translations = [c for c in taxonomy.columns if TRANSLATION_COLUMN.fullmatch(c)]
    unrecognised = [c for c in taxonomy.columns if c.startswith('common_name_')
                    and c not in translations and c not in NOT_TRANSLATION_COLUMNS]
    if unrecognised:
        print(f'⚠️  Not a recognised translation column, left out of labels.csv: {", ".join(unrecognised)}')
    labels = taxonomy.set_index('birdnet_id', drop=False).loc[ids, TAXONOMY_LABEL_COLUMNS + translations]
    labels.insert(0, 'ix', range(len(labels)))
    return labels


def write_label_files(output_dir, pruned_labels, taxonomy_labels):
    """Write this export's label files and delete any that an earlier run left behind.

    A table that is None means this run has no such file; an existing copy is removed so
    the export directory never holds labels for a different set of classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, table in (('labels_pruned.xlsx', pruned_labels), ('labels.csv', taxonomy_labels)):
        path = os.path.join(output_dir, name)
        if table is not None:
            if name.endswith('.xlsx'):
                table.to_excel(path, index=False)
            else:
                table.to_csv(path, index=False)
            print(f'Wrote {len(table)} labels to {path}')
        elif os.path.exists(path):
            os.remove(path)
            print(f'Removed {path} left by an earlier run')


def _list_models(base_dir):
    """Return subdirectory names under models/ that contain model.py."""
    models_root = os.path.join(base_dir, 'models')
    if not os.path.isdir(models_root):
        return []
    found = []
    for name in sorted(os.listdir(models_root)):
        candidate = os.path.join(models_root, name, 'model.py')
        if os.path.isfile(candidate):
            found.append(name)
    return found


def _build_parser():
    """Build the argparse parser for the conversion CLI."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    available_models = _list_models(base_dir)
    models_help = ', '.join(available_models) if available_models else '(none found)'

    parser = argparse.ArgumentParser(
        description='Convert BirdNET V3.0 PyTorch models to other inference formats.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'examples:\n'
            '  %(prog)s                              # convert default model, all formats, all precisions\n'
            '  %(prog)s -f onnx torchscript           # ONNX + TorchScript only\n'
            '  %(prog)s -p fp32                        # FP32 only (skip FP16/INT8)\n'
            '  %(prog)s -f onnx -p fp16               # ONNX FP16 only\n'
            '  %(prog)s -m my_model                     # use a different model\n'
            '  %(prog)s -o output/                     # custom output directory\n'
            '  %(prog)s --batch-size 4                 # set TFLite batch size\n'
            '  %(prog)s --list-models                  # show available models\n'
            '  %(prog)s --head-weights 1.0 0.0 0.0     # use only Head 1 for predictions and reduce model size\n'
            '  %(prog)s --class-pruning labels.xlsx    # keep only rows with selected=1\n'
            '  %(prog)s --no-class-pruning             # export every class, ignoring labels_pruning.xlsx\n'
            '  %(prog)s --taxonomy taxonomy.csv        # BirdNET taxonomy for labels.csv\n'
            '  %(prog)s --experimental-mode            # enable full INT8 export for ONNX and TFLite\n'
        ),
    )
    parser.add_argument(
        '-m', '--model-name',
        default=DEFAULT_MODEL_NAME,
        help=(
            'Model directory containing model.py and a PyTorch checkpoint: '
            'a name in models/ or a path (default: %(default)s)'
        ),
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for converted model output (default: %(default)s)',
    )
    parser.add_argument(
        '-f', '--formats',
        nargs='+',
        choices=ALL_FORMATS,
        default=ALL_FORMATS,
        metavar='FMT',
        help=(
            'Output format(s) to produce. '
            f'Choose from: {", ".join(ALL_FORMATS)} (default: all)'
        ),
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size baked into TFLite models (default: %(default)s)',
    )
    parser.add_argument(
        '-p', '--precisions',
        nargs='+',
        choices=ALL_PRECISIONS,
        default=ALL_PRECISIONS,
        metavar='PREC',
        help=(
            'Precision variant(s) to produce. '
            f'Choose from: {", ".join(ALL_PRECISIONS)} (default: all)'
        ),
    )
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=3.0,
        help=(
            'Minimum audio segment duration in seconds used during tracing '
            '(default: %(default)s)'
        ),
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models in the models/ directory and exit',
    )
    parser.add_argument(
        '--head-weights',
        type=float,
        nargs=3,
        default=None,
        metavar='W',
        help=(
            'Weights for the three classifier heads, e.g. --head-weights 1.0 0.0 0.0 '
            'to use only Head 1. (default: from model definition, typically [0.4, 0.35, 0.25])'
        ),
    )
    pruning = parser.add_mutually_exclusive_group()
    pruning.add_argument(
        '--class-pruning',
        default=None,
        metavar='XLSX',
        help=(
            'Excel file of class labels. Rows with selected=1 are kept, using the ix column. '
            f'If omitted, {DEFAULT_CLASS_PRUNING_FILE} in the model directory is used when '
            'present; otherwise all classes are exported.'
        ),
    )
    pruning.add_argument(
        '--no-class-pruning',
        action='store_true',
        help=f'Export every class, even if the model directory has {DEFAULT_CLASS_PRUNING_FILE}.',
    )
    parser.add_argument(
        '--taxonomy',
        default=DEFAULT_TAXONOMY_FILE,
        metavar='CSV',
        help=(
            'BirdNET taxonomy used to write labels.csv when the class pruning sheet has a '
            f'birdnet_id column (default: {os.path.basename(DEFAULT_TAXONOMY_FILE)} in the '
            'repository root)'
        ),
    )
    parser.add_argument(
        '--experimental-mode',
        action='store_true',
        default=False,
        help=(
            'Enable experimental features such as full INT8 quantization for ONNX '
            'and TFLite. Without this flag, full INT8 variants are skipped. '
            '(default: disabled)'
        ),
    )
    return parser



def main():
    """Entry point: parse CLI arguments, configure globals, and run conversions."""
    global classIxsSelection
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_models:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models = _list_models(base_dir)
        if models:
            print('Available models:')
            for m in models:
                print(f'  {m}')
        else:
            print('No models found in models/.')
        return

    _init_config(
        args.model_name,
        args.output_dir,
        args.batch_size,
        args.segment_duration,
        precision_set=set(args.precisions),
        experimental_mode=args.experimental_mode,
        head_weights=args.head_weights,
    )

    formats = set(args.formats)
    class_pruning = args.class_pruning
    default_pruning = os.path.join(modelDir, DEFAULT_CLASS_PRUNING_FILE)
    if class_pruning is None and not args.no_class_pruning and os.path.isfile(default_pruning):
        class_pruning = default_pruning
    classIxsSelection = load_class_ixs_from_excel(class_pruning) if class_pruning else None
    # Build and check every label table before writing, so a failed check writes nothing.
    # labels.csv replaces labels_pruned.xlsx; the latter ships only for sheets without birdnet_id.
    pruned_labels = read_pruned_labels(class_pruning) if classIxsSelection else None
    taxonomy_labels = (build_taxonomy_labels(pruned_labels, args.taxonomy)
                       if pruned_labels is not None else None)
    write_label_files(outputDir, pruned_labels if taxonomy_labels is None else None, taxonomy_labels)

    print(f'Model directory:    {modelDir}')
    print(f'Output directory:   {outputDir}')
    print(f'Output format(s):   {", ".join(sorted(formats))}')
    print(f'Precision(s):       {", ".join(sorted(precisions))}')
    print(f'Batch size:         {batchSize}')
    print(f'Segment duration:   {segmentDuration}s')
    print(f'Head weights:       {headWeights}')
    print(f'Class pruning:      {len(classIxsSelection) if classIxsSelection else "disabled"}'
          + (f' classes from {class_pruning}' if classIxsSelection else ''))
    print(f'Taxonomy labels:    '
          + (f'labels.csv from {args.taxonomy}' if taxonomy_labels is not None
             else 'skipped (label sheet has no birdnet_id column)' if classIxsSelection
             else 'skipped (class pruning disabled)'))
    print(f'Experimental mode:  {"enabled" if experimentalMode else "disabled"}')
    print()

    if 'tf' in formats or 'tflite' in formats:
        createTfPreprocessingModel()
        createTfMainModel(saveKerasModel=False)
        combineTfModels()

    if 'torchscript' in formats:
        createTorchScriptModel()

    if 'onnx' in formats:
        createOnnxModel()

    print('\nDone.')

if __name__ == '__main__':
    main()