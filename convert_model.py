import argparse
import os
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


ALL_FORMATS = ['tf', 'tflite', 'torchscript', 'onnx']
ALL_PRECISIONS = ['fp32', 'fp16', 'int8']
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models', '01_BirdNET+_V3.0-preview3_Global_11K'
)

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'export'
)

DEFAULT_CALIBRATION_AUDIO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'example', 'soundscape.wav'
)

# Module-level state (set by _init_config, read by conversion functions)
modelDir = DEFAULT_MODEL_DIR + '/'
outputDir = DEFAULT_OUTPUT_DIR + '/'
batchSize = 1
segmentDuration = 3.0
precisions = set(ALL_PRECISIONS)  # default: all precisions
calibrationAudioPath = DEFAULT_CALIBRATION_AUDIO
model_module = None


def _init_config(model_dir, output_dir, batch_size, segment_duration,
                 precision_set=None, calibration_audio=None):
    """Initialise module-level config and load the model definition module."""
    global modelDir, outputDir, batchSize, segmentDuration, model_module
    global precisions, calibrationAudioPath

    modelDir = model_dir.rstrip('/') + '/'
    outputDir = output_dir.rstrip('/') + '/'
    batchSize = batch_size
    segmentDuration = segment_duration
    precisions = precision_set if precision_set else set(ALL_PRECISIONS)
    calibrationAudioPath = calibration_audio or DEFAULT_CALIBRATION_AUDIO

    model_module_path = os.path.join(modelDir, 'pytorch', 'model.py')
    if not os.path.isfile(model_module_path):
        sys.exit(f'Error: model definition not found at {model_module_path}')

    module_name = 'model_module'
    spec = importlib.util.spec_from_file_location(module_name, model_module_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = model_module
    spec.loader.exec_module(model_module)


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

    
def loadPytorchModelAndWeights():
    """Load the MultiHead01 classifier (without preprocessing) and its checkpoint weights.

    Returns:
        torch.nn.Module: The model in eval mode on CPU.
    """
    # Load model
    model = model_module.MultiHead01(model_module.backbone, model_module.nClasses, pretrained=False)

    # Load weights
    checkpointPath = modelDir + 'pytorch//Checkpoint35.pth.tar'
    checkpoint = torch.load(checkpointPath, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
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
    model = loadPytorchModelAndWeightsIncludingPreprocessingLayer()
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

    For INT8 calibration the preprocessing model is loaded to generate realistic
    spectrogram data from ``calibrationAudioPath``.

    Args:
        saveKerasModel: If True, also save a ``.keras`` file.

    Output: ``{outputDir}/tf/main/`` (SavedModel + TFLite variants)
    """
    expected_t_dims = get_expected_t_dims()
    model = loadPytorchModelAndWeights()

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

    edge_model = litert_torch.convert(
        model,
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
            return result[output_keys[0]]

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

    def _make_representative_dataset():
        """Generate calibration data for INT8 quantization of the main model.

        The main model expects spectrogram input (batch, 3, height, width).
        We run the preprocessing on calibration audio to produce realistic
        spectrogram data, falling back to random if unavailable.
        """
        import librosa

        # Try to produce real spectrograms from calibration audio
        spectrograms = []
        if os.path.isfile(calibrationAudioPath):
            audio, _ = librosa.load(calibrationAudioPath, sr=model_module.sampleRate,
                                    dtype=np.float32)
            print(f"  INT8 calibration audio: {calibrationAudioPath} "
                  f"({len(audio)/model_module.sampleRate:.1f}s)")
            segment_samples = int(segmentDuration * model_module.sampleRate)
            step = segment_samples // 2

            # Load the TF preprocessing model to convert audio → spectrogram
            pre_path = outputDir + 'tf/preprocessing/'
            if os.path.isdir(pre_path):
                pre_model = tf.saved_model.load(pre_path)
                pre_fn = pre_model.signatures['serving_default']
                pre_out_key = list(pre_fn.structured_outputs.keys())[0]
                for start in range(0, max(1, len(audio) - segment_samples + 1), step):
                    chunk = audio[start:start + segment_samples]
                    if len(chunk) < segment_samples:
                        chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
                    chunk_tf = tf.constant(chunk.reshape(batchSize, -1), dtype=tf.float32)
                    spec = pre_fn(chunk_tf)[pre_out_key].numpy()
                    # spec shape: (B, H, W, 3) → transpose to (B, 3, H, W)
                    spec = np.transpose(spec, (0, 3, 1, 2))
                    spectrograms.append(spec.astype(np.float32))
            else:
                print(f"  Warning: preprocessing model not found at {pre_path}, "
                      "using random spectrogram data for calibration")
        else:
            print(f"  INT8 calibration: {calibrationAudioPath} not found, "
                  "using random spectrogram data")

        def representative_dataset():
            if spectrograms:
                for spec in spectrograms:
                    yield [spec]
            else:
                rng = np.random.default_rng(seed=42)
                for _ in range(100):
                    yield [rng.standard_normal(inputShape).astype(np.float32)]

        return representative_dataset

    print("Step 3: Converting and fixing TFLite models...")

    if 'fp32' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model.tflite')
    if 'fp16' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model_fp16.tflite',
                                optimize=True, supported_types=[tf.float16])
    if 'int8' in precisions:
        rep_data = _make_representative_dataset()
        _convert_and_fix_tflite(dstDir, dstDir + 'model_int8.tflite',
                                optimize=True, representative_dataset=rep_data)


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
            main_out = self.main.signatures['serving_default'](pre_out)['output_0']
            return main_out


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

    def _make_combined_representative_dataset():
        """Generate calibration data for combined INT8 model (raw audio input)."""
        import librosa
        audio = None
        if os.path.isfile(calibrationAudioPath):
            audio, _ = librosa.load(calibrationAudioPath, sr=model_module.sampleRate,
                                    dtype=np.float32)
            print(f"  INT8 calibration audio: {calibrationAudioPath} "
                  f"({len(audio)/model_module.sampleRate:.1f}s)")
        else:
            print(f"  INT8 calibration: {calibrationAudioPath} not found, using random data")

        segment_samples = int(segmentDuration * model_module.sampleRate)

        def representative_dataset():
            if audio is not None:
                step = segment_samples // 2
                for start in range(0, max(1, len(audio) - segment_samples + 1), step):
                    chunk = audio[start:start + segment_samples]
                    if len(chunk) < segment_samples:
                        chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
                    yield [chunk.reshape(batchSize, -1).astype(np.float32)]
            else:
                rng = np.random.default_rng(seed=42)
                for _ in range(100):
                    yield [rng.standard_normal((batchSize, segment_samples)).astype(np.float32)]

        return representative_dataset

    if 'fp32' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model.tflite')
    if 'fp16' in precisions:
        _convert_and_fix_tflite(dstDir, dstDir + 'model_fp16.tflite',
                                optimize=True, supported_types=[tf.float16])
    if 'int8' in precisions:
        rep_data = _make_combined_representative_dataset()
        _convert_and_fix_tflite(dstDir, dstDir + 'model_int8.tflite',
                                optimize=True, representative_dataset=rep_data)


def loadPytorchModelAndWeightsIncludingPreprocessingLayer():
    """Load the full model (preprocessing + classifier) and its checkpoint weights.

    Uses ``strict=False`` because the preprocessing layer parameters are not
    stored in the checkpoint — they are initialised from layer defaults.

    Returns:
        torch.nn.Module: The model in eval mode on GPU (if available) or CPU.
    """
    # Load model
    model = model_module.MultiHead01WithPreprocessing(model_module.backbone, model_module.nClasses, pretrained=False)

    # Load weights
    checkpointPath = modelDir + 'pytorch//Checkpoint35.pth.tar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=True)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False) # strict=False to ignore missing keys for preprocessing parameters
    model.to(device)
    model.eval()

    return model


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
    model = loadPytorchModelAndWeightsIncludingPreprocessingLayer()
    
    dstDir = outputDir + 'torchscript/'
    os.makedirs(dstDir, exist_ok=True)

    if 'fp32' in precisions:
        scripted_module = torch.jit.script(model)
        scripted_module.save(dstDir + 'model.pt')
        print('TorchScript model saved to', dstDir + 'model.pt')

    if 'fp16' in precisions:
        model_fp16 = model.half()
        scripted_fp16 = torch.jit.script(model_fp16)
        scripted_fp16.save(dstDir + 'model_fp16.pt')
        print('TorchScript FP16 model saved to', dstDir + 'model_fp16.pt')



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
    model = loadPytorchModelAndWeightsIncludingPreprocessingLayer()

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
        output_names=['embeddings', 'predictions'],
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

    # FP16 variant
    if 'fp16' in precisions:
        from onnxconverter_common import float16
        from onnx import TensorProto
        fp32_model = onnx.load(onnx_path)
        del fp32_model.graph.value_info[:]
        # Convert entire graph to fp16, then manually wrap I/O with Cast nodes
        # so the model accepts float32 input and returns float32 output.
        # (keep_io_types=True is unreliable for ops like Gemm whose outputs
        # are also consumed internally — it breaks type homogeneity.)
        fp16_model = float16.convert_float_to_float16(fp32_model)
        graph = fp16_model.graph

        # Wrap each float16 input: keep graph input as float32, insert
        # Cast(float32→float16), rename all downstream consumers to the cast output.
        for inp in graph.input:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                old_name = inp.name
                cast_out = old_name + '_cast_fp16'
                cast_node = helper.make_node('Cast', [old_name], [cast_out],
                                             to=TensorProto.FLOAT16)
                for node in graph.node:
                    for i, n in enumerate(node.input):
                        if n == old_name:
                            node.input[i] = cast_out
                graph.node.insert(0, cast_node)
                inp.type.tensor_type.elem_type = TensorProto.FLOAT

        # Wrap each float16 output: add Cast(float16→float32) that reads
        # from the original fp16 tensor and writes to a *new* name, then
        # point the graph output entry at that new name.  This keeps the
        # fp16 tensor intact for any internal consumers (e.g. Gemm reading
        # 'embeddings' while 'embeddings' is also a graph output).
        for out in graph.output:
            if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
                fp16_name = out.name
                fp32_name = fp16_name + '_fp32'
                cast_node = helper.make_node('Cast', [fp16_name], [fp32_name],
                                             to=TensorProto.FLOAT)
                graph.node.append(cast_node)
                out.name = fp32_name
                out.type.tensor_type.elem_type = TensorProto.FLOAT

        fp16_path = dstDir + 'model_fp16.onnx'
        onnx.save(fp16_model, fp16_path, save_as_external_data=False)
        print(f'ONNX FP16 model exported to {fp16_path}')

    # INT8 variant (dynamic quantization)
    if 'int8' in precisions:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        # Clear stale shape annotations that conflict with our Resize/Conv2dSame
        # patches — shape inference inside quantize_dynamic will re-derive them.
        model_for_quant = onnx.load(onnx_path)
        del model_for_quant.graph.value_info[:]
        temp_quant_path = dstDir + 'model_pre_quant.onnx'
        onnx.save(model_for_quant, temp_quant_path, save_as_external_data=False)
        int8_path = dstDir + 'model_int8.onnx'
        quantize_dynamic(
            temp_quant_path, int8_path,
            weight_type=QuantType.QInt8,
        )
        os.remove(temp_quant_path)
        print(f'ONNX INT8 (dynamic quantization) model exported to {int8_path}')

###########################################################################################
###########################################################################################
###########################################################################################


def _list_models(base_dir):
    """Return subdirectory names under models/ that contain pytorch/model.py."""
    models_root = os.path.join(base_dir, 'models')
    if not os.path.isdir(models_root):
        return []
    found = []
    for name in sorted(os.listdir(models_root)):
        candidate = os.path.join(models_root, name, 'pytorch', 'model.py')
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
            '  %(prog)s --calibration-audio file.wav   # custom INT8 calibration audio\n'
            '  %(prog)s -m models/my_model             # use a different model directory\n'
            '  %(prog)s -o output/                     # custom output directory\n'
            '  %(prog)s --batch-size 4                 # set TFLite batch size\n'
            '  %(prog)s --list-models                  # show available models\n'
        ),
    )
    parser.add_argument(
        '-m', '--model-dir',
        default=DEFAULT_MODEL_DIR,
        help=(
            'Path to the model directory containing pytorch/model.py and '
            'a PyTorch checkpoint (default: %(default)s)'
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
        '--calibration-audio',
        default=DEFAULT_CALIBRATION_AUDIO,
        help=(
            'Audio file for INT8 calibration data '
            '(default: %(default)s)'
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
    return parser


def main():
    """Entry point: parse CLI arguments, configure globals, and run conversions."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_models:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models = _list_models(base_dir)
        if models:
            print('Available models:')
            for m in models:
                print(f'  models/{m}')
        else:
            print('No models found in models/.')
        return

    _init_config(args.model_dir, args.output_dir, args.batch_size, args.segment_duration,
                 precision_set=set(args.precisions), calibration_audio=args.calibration_audio)

    formats = set(args.formats)

    print(f'Model directory:    {modelDir}')
    print(f'Output directory:   {outputDir}')
    print(f'Output format(s):   {", ".join(sorted(formats))}')
    print(f'Precision(s):       {", ".join(sorted(precisions))}')
    print(f'Batch size:         {batchSize}')
    print(f'Segment duration:   {segmentDuration}s')
    if 'int8' in precisions:
        print(f'Calibration audio:  {calibrationAudioPath}')
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