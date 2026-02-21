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
from fix_tflite_reshape import fix_reshape_for_dynamic_batch, verify_fix


# Adjust this path to point to the directory containing the PyTorch checkpoint and model definition (model.py)
modelDir = os.path.dirname(os.path.abspath(__file__)) + '/models/01_BirdNET+_V3.0-preview3_Global_11K/'

batchSize = 1
# Use smallest duration that should be supported by litert
segmentDuration = 2.0 #3.0 # 3.0 5.0 


# Load model definitions from model.py
model_module_path = modelDir + 'pytorch/model.py'
module_name = 'model_module'
spec = importlib.util.spec_from_file_location(module_name, model_module_path)
model_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = model_module
spec.loader.exec_module(model_module)


def createTfPreprocessingModel():

    allow_dynamic_input_length = True
    batch_size = None #1 # None


    
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
    dstDir = modelDir + 'tf/preprocessing/'
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


    # Works but leads to quite hight differences (Max absolute difference: 8.590895)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    dstPath = dstDir + 'model_optimized.tflite'
    with open(dstPath, 'wb') as f:
        f.write(tflite_model)
    print('Exported optimized preprocessing tflite model to', dstPath)

    
def loadPytorchModelAndWeights():

    # Load model
    model = model_module.MultiHead01(model_module.backbone, model_module.nClasses, pretrained=False)

    # Load weights
    checkpointPath = modelDir + 'pytorch//Checkpoint35.pth.tar'
    checkpoint = torch.load(checkpointPath, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    return model


def createTfMainModel(saveKerasModel=False):

    model = loadPytorchModelAndWeights()

    imageWidth = int(model_module.imageWidthResizeFactor * segmentDuration * model_module.sampleRate / model_module.fftHopSizeInSamples)


    inputShape = (batchSize, 3, model_module.imageHeight, imageWidth)
    dtype = np.float32
    testInputNp = np.random.rand(inputShape[0],inputShape[1],inputShape[2],inputShape[3]).astype(dtype)
    testInputTorch = torch.from_numpy(testInputNp)
    sample_inputs = (testInputTorch,)

    print(f"Sample input shape: {testInputTorch.shape}")

    dstDir = modelDir + 'tf/main/'
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
            tf.TensorSpec(shape=[None, 3, model_module.imageHeight, None], dtype=tf.float32)
        ])
        def serve(self, x):
            result = self._fn(**{input_keys[0]: x})
            return result[output_keys[0]]

    wrapper = DynamicWrapper(static_model, static_fn)

    # Trace with concrete input
    test_input_tf = tf.constant(testInputNp)
    _ = wrapper.serve(test_input_tf)

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

    def _convert_and_fix_tflite(saved_model_dir, tflite_path, optimize=False):
        """Convert SavedModel to TFLite, then fix static Reshape for dynamic batch."""
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        label = 'optimized ' if optimize else ''
        print(f"  Exported {label}TFLite model to {tflite_path}")

        print(f"  Fixing static Reshape nodes for dynamic batch...")
        n_fixed = fix_reshape_for_dynamic_batch(tflite_path)
        if n_fixed > 0:
            verify_fix(tflite_path, model_module.imageHeight, imageWidth)

    print("Step 3: Converting and fixing TFLite models...")
    _convert_and_fix_tflite(dstDir, dstDir + 'model.tflite', optimize=False)
    _convert_and_fix_tflite(dstDir, dstDir + 'model_optimized.tflite', optimize=True)


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

    imageWidth = int(model_module.imageWidthResizeFactor * segmentDuration * model_module.sampleRate / model_module.fftHopSizeInSamples)

    ## Wrap both models in a new tf.Model that runs them sequentially
    class CombinedModel(tf.Module):
        def __init__(self, pre, main):
            super().__init__()
            self.pre = pre
            self.main = main

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
        def __call__(self, x):
            pre_out = self.pre.signatures['serving_default'](x)['output_0']
            # reshape pre_out to match input shape of main model: (B, 224, W, 3) --> (B, 3, 224, W)
            pre_out = tf.transpose(pre_out, perm=[0, 3, 1, 2])
            main_out = self.main.signatures['serving_default'](pre_out)['output_0']
            return main_out


    path_pre = modelDir + 'tf/preprocessing/'
    path_main = modelDir + 'tf/main/'

    pre = tf.saved_model.load(path_pre)
    main = tf.saved_model.load(path_main)

    ## Export the combined model as a SavedModel
    combined_model = CombinedModel(pre, main)
    dstDir = modelDir + 'tf/combined/'
    tf.saved_model.save(
        combined_model, dstDir, 
        signatures=combined_model.__call__.get_concrete_function()
    )
    print(f'Combined model saved to {dstDir}')

    ## Convert to TFLite and fix static Reshape ops for dynamic batch

    def _convert_and_fix_tflite(saved_model_dir, tflite_path, optimize=False):
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        label = 'optimized ' if optimize else ''
        print(f'  Exported combined {label}TFLite model to {tflite_path}')

        print(f'  Fixing static Reshape nodes for dynamic batch...')
        n_fixed = fix_reshape_for_dynamic_batch(tflite_path)
        if n_fixed > 0:
            verify_fix(tflite_path, model_module.imageHeight, imageWidth)

    _convert_and_fix_tflite(dstDir, dstDir + 'model.tflite', optimize=False)
    _convert_and_fix_tflite(dstDir, dstDir + 'model_optimized.tflite', optimize=True)


def loadPytorchModelAndWeightsIncludingPreprocessingLayer():

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

    # Load model and weights
    model = loadPytorchModelAndWeightsIncludingPreprocessingLayer()
    
    scripted_module = torch.jit.script(model)
    dstDir = modelDir + 'torchscript/'
    os.makedirs(dstDir, exist_ok=True)
    scripted_module.save(dstDir + 'model.pt')
    print('TorchScript model saved to', dstDir + 'model.pt')



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

    # Load model and weights
    model = loadPytorchModelAndWeightsIncludingPreprocessingLayer()

    # Create dummy input (but model should be able to handle dynamic batch size and input length, so this is just for export)
    batch_size = 4
    inputBatch = torch.rand(batch_size, int(segmentDuration*model_module.sampleRate)).to(next(model.parameters()).device)

    dstDir = modelDir + 'onnx/'
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

###########################################################################################
###########################################################################################
###########################################################################################


if __name__ == '__main__':

    createTfPreprocessingModel()
    createTfMainModel(saveKerasModel=False)
    combineTfModels()

    createTorchScriptModel()
    createOnnxModel()