
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


modelDir = os.path.dirname(os.path.abspath(__file__)) + '/models/01_BirdNET+_V3.0-preview3_Global_11K/'

segmentDuration = 3.0
batchSize = 1


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


    ## Conversion

    dstDir = modelDir + 'tf/main/'

    edge_model = litert_torch.convert(
        model, 
        sample_inputs, 
        _saved_model_dir=dstDir
    )

    # Export TFLite model
    dstPath = dstDir + 'model.tflite'
    edge_model.export(dstPath)
    print('Exported main tflite model to', dstPath)


    ## Get optimized TFLite model

    # # Use litert_torch to create optimized TFLite model
    # edge_model = litert_torch.convert(
    #     model, 
    #     sample_inputs, 
    #     _ai_edge_converter_flags={'optimizations': [tf.lite.Optimize.DEFAULT]},
    #     _saved_model_dir=dstDir
    # )
    # dstPath = dstDir + 'model_optimized.tflite'
    # edge_model.export(dstPath)
    # print('Exported optimized main tflite model to', dstPath)

    # Use saved model and TFLiteConverter to create optimized TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(dstDir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    dstPath = dstDir + 'model_optimized.tflite'
    with open(dstPath, 'wb') as f: f.write(tflite_model)
    print('Exported optimized main tflite model to', dstPath)


    ## Save as keras model
    if saveKerasModel:
        dstPath = dstDir + 'model.keras'
        saved_model = tf.keras.layers.TFSMLayer(dstDir, call_endpoint='serving_default') # Load the saved model
        tf.keras.models.save_model(saved_model, dstPath, overwrite=True)
        print('Exported keras model to', dstPath)

   
def combineTfModels():

    ## Wrap both models in a new tf.Model that runs them sequentially
    class CombinedModel(tf.Module):
        def __init__(self, pre, main):
            super().__init__()
            self.pre = pre
            self.main = main

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 96000], dtype=tf.float32)])
        def __call__(self, x):
            pre_out = self.pre.signatures['serving_default'](x)['output_0']
            # reshape pre_out to match input shape of main model: (1, 224, 281, 3) --> (1, 3, 224, 281)
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

    ## Convert the combined model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(dstDir)
    tflite_model = converter.convert()
    dstPath = dstDir + 'model.tflite'
    with open(dstPath, 'wb') as f: f.write(tflite_model)
    print('Exported combined tflite model to', dstPath)

    ## Get optimized TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(dstDir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    dstPath = dstDir + 'model_optimized.tflite'
    with open(dstPath, 'wb') as f: f.write(tflite_model)
    print('Exported combined optimized tflite model to', dstPath)


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


def createOnnxModel():

    model = loadPytorchModelAndWeightsIncludingPreprocessingLayer()

    # Create dummy input
    batch_size = 4
    inputBatch = torch.rand(batch_size, int(segmentDuration*model_module.sampleRate)).to(next(model.parameters()).device)

    dstDir = modelDir + 'onnx/'
    os.makedirs(dstDir, exist_ok=True)

    torch.onnx.export(
        model,  # Use original model, not scripted_module
        inputBatch, 
        dstDir+'temp.onnx', 
        verbose=True, 
        input_names=['input'],  # needed ?
        #output_names=['output'],  # needed ?
        output_names=['embeddings', 'predictions'],  # needed ?
        dynamo=True,
        dynamic_shapes={'input': {0: 'batch_size', 1: 'samples'}}
    )

    # Load model and save wihtout external data format
    onnxModel = onnx.load(dstDir+'temp.onnx')
    onnx.save(onnxModel, dstDir+'model.onnx', save_as_external_data=False)
    os.remove(dstDir+'temp.onnx')
    os.remove(dstDir+'temp.onnx.data')
    
    print('ONNX model exported successfully to', dstDir+'model.onnx')

###########################################################################################
###########################################################################################
###########################################################################################


# createTfPreprocessingModel()
# createTfMainModel(saveKerasModel=True)
# combineTfModels()

createOnnxModel()