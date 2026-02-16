
import os
import sys
import numpy as np
import importlib.util
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
import torch
import litert_torch


modelDir = os.path.dirname(os.path.abspath(__file__)) + '/models/01_BirdNET+_V3.0-preview3_Global_11K/'

backbone = 'tf_efficientnetv2_s_in21k'
nClasses = 11560 # 1225
sampleRate = 32000
fftSizeInSamples = 2048
fftHopSizeInSamples = 512
nMelBands = 128
melStartFreq = 40.0
melEndFreq = 15000.0
imageHeight = 224
imageWidthResizeFactor = 1.5

segmentDuration = 3.0
batchSize = 1


def createTfPreprocessingModel():

    allow_dynamic_input_length = True
    batch_size = None #1 # None

    class Preprocessing(Layer):

        def __init__(self, name=None, **kwargs):

            self.sampleRate=sampleRate
            self.fftSizeInSamples=fftSizeInSamples
            self.fftHopSizeInSamples=fftHopSizeInSamples
            self.nMelBands=nMelBands
            self.melStartFreq=melStartFreq
            self.melEndFreq=melEndFreq

            nLowFreqsInPixelToCutMax=2
            nHighFreqsInPixelToCutMax=4
            self.nLowFreqsInPixelToCut = int(nLowFreqsInPixelToCutMax/2.0)
            self.nHighFreqsInPixelToCut = int(nHighFreqsInPixelToCutMax/2.0)


            self.specImageMean = [0.5, 0.4, 0.3]
            self.specImageStd = [0.5, 0.3, 0.1]

            self.imageHeight = imageHeight
            self.imageWidthResizeFactor = imageWidthResizeFactor

            self.topDb = 100

            # Get mel basis matrix from file (created with librosa)
            melBasis = np.load(modelDir + 'pytorch/melBasis.npy')
            self.melBasisTf = tf.convert_to_tensor(melBasis.T, dtype=tf.float32)

            super(Preprocessing, self).__init__(name=name, **kwargs)

        def call(self, inputs):

            # For dynamic input shapes, we can't compute output shape statically
            # Return the lambda without output_shape - TensorFlow will infer it at runtime
            return keras.layers.Lambda(self._preprocess)(inputs)
        
        def _preprocess(self, inputs):

            ### FFT

            # Get image width based on input length dynamically
            # Use tf.shape for dynamic dimensions
            num_samples = tf.shape(inputs)[1]
            duration = tf.cast(num_samples, tf.float32) / self.sampleRate
            imageWidth = tf.cast(self.imageWidthResizeFactor * duration * self.sampleRate / self.fftHopSizeInSamples, tf.int32)

            sampleVecTf = inputs

            # Original number of spectrogram frames (used to reconstruct spec after padding and centering)
            # Use tf.shape for dynamic dimensions, but keep static shape access for fixed dimensions
            num_samples_static = inputs.shape[1]
            if num_samples_static is not None:
                # Static shape known at graph construction time
                nFramesOrg = -(-num_samples_static // self.fftHopSizeInSamples)
            else:
                # Dynamic shape - compute at runtime
                nFramesOrg = tf.cast(tf.math.ceil(tf.cast(num_samples, tf.float32) / self.fftHopSizeInSamples), tf.int32)
            
            pad_amount = 2 * (self.fftSizeInSamples - self.fftHopSizeInSamples)
            # Pad along the sample dimension (axis=1)
            sampleVecTf = tf.pad(sampleVecTf, [[0, 0], [pad_amount // 2, pad_amount // 2]], 'REFLECT')

            specTf = tf.math.abs(tf.signal.stft(sampleVecTf, frame_length=self.fftSizeInSamples, frame_step=self.fftHopSizeInSamples, fft_length=self.fftSizeInSamples, window_fn=tf.signal.hann_window, pad_end=False))**2.0

            # Remove padding (make 2nd dimension smaller) ... --> nFramesOrg
            # Use dynamic slicing for dynamic shapes
            if num_samples_static is not None:
                specTf = specTf[:, 1:nFramesOrg+1, :]
            else:
                specTf = tf.slice(specTf, [0, 1, 0], [-1, nFramesOrg, -1])

            ### Mel spectrogram

            # Apply mel filterbank
            melSpecTf = tf.tensordot(specTf, self.melBasisTf, axes=1)

            ### Scale to dB
            ref=1.0
            amin=1e-10

            melSpecTf = 10.0 * (tf.math.log(tf.math.maximum(melSpecTf, amin)) / tf.math.log(10.0))
            ref_value = tf.math.abs(ref)
            melSpecTf -= 10.0 * tf.math.log(tf.math.maximum(ref_value, amin)) / tf.math.log(10.0)
            melSpecTf = tf.math.maximum(melSpecTf, tf.math.reduce_max(melSpecTf) - self.topDb)

            # [bs, n_frames, n_mels] [bs, n_mels, n_frames]
            melSpecTf = tf.transpose(melSpecTf, perm=[0, 2, 1])


            ### Cut low and high frequencies - use tf.slice instead of advanced indexing
            # melSpecTf = melSpecTf[:, self.nLowFreqsInPixelToCut:-self.nHighFreqsInPixelToCut, :]
            mel_height = melSpecTf.shape[1]
            start_idx = self.nLowFreqsInPixelToCut
            size = mel_height - self.nLowFreqsInPixelToCut - self.nHighFreqsInPixelToCut
            melSpecTf = tf.slice(melSpecTf, [0, start_idx, 0], [-1, size, -1])


            # Flip spectrum vertically (only for better visialization, low freq. at bottom)
            melSpecTf = tf.reverse(melSpecTf, axis=[1])

            # Normalize values between 0.0 and 1.0 (per sample, not per batch)
            epsilon = 1e-10  # Small value to prevent division by zero
            min_val = tf.reduce_min(melSpecTf, axis=[1, 2], keepdims=True)  # Shape: [batch, 1, 1]
            max_val = tf.reduce_max(melSpecTf, axis=[1, 2], keepdims=True)  # Shape: [batch, 1, 1]
            melSpecTf = (melSpecTf - min_val) / (max_val - min_val + epsilon)

            # Invert colors
            melSpecTf = 1.0 - melSpecTf

            ## Resize

            # Add channels dimensions (for resizing)
            melSpecTf = tf.expand_dims(melSpecTf, axis=-1)

            # tf.image.ResizeMethod: area, bicubic, bilinear, gaussian, lanczos3, lanczos5, mitchellcubic, nearest
            #InterpolationMethod = 'bicubic' # Original, but requires Flex delegate
            InterpolationMethod = 'bilinear' # Use bilinear (or nearest) instead of bicubic to avoid Flex delegate requirement
            antialias = False

            # Resize
            melSpecTf = tf.image.resize(melSpecTf, [self.imageHeight, imageWidth], method=InterpolationMethod, antialias=antialias)

            # Remove channels dimension
            melSpecTf = tf.squeeze(melSpecTf, axis=-1)

            # Format now is [bs, height, width] in float32 with values between 0.0 and 1.0

            # [bs, h, w] --> [bs, 3, h, w]
            melSpecTf = tf.stack([melSpecTf, melSpecTf, melSpecTf], axis=1)

            mean = tf.convert_to_tensor(self.specImageMean, dtype=tf.float32)
            std = tf.convert_to_tensor(self.specImageStd, dtype=tf.float32)
            mean = tf.reshape(mean, (1, -1, 1, 1))
            std = tf.reshape(std, (1, -1, 1, 1))
            melSpecTf = tf.math.divide(tf.math.subtract(melSpecTf, mean), std)

            melSpecTf = tf.transpose(melSpecTf, perm=[0, 2, 3, 1]) # bs,c,h,w --> bs,h,w,c
            
            return melSpecTf

    
    preprocessing = Preprocessing()

    
    if allow_dynamic_input_length:
        inputs = keras.Input(shape=(None,), batch_size=batch_size, name='inputs') # dynamic input length
    else:
        segmentDurationInSamples = int(segmentDuration * sampleRate)
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

    # Load model definition from model.py
    model_module_path = modelDir + 'pytorch/model.py'
    module_name = 'model_module'
    spec = importlib.util.spec_from_file_location(module_name, model_module_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = model_module
    spec.loader.exec_module(model_module)

    # Load model
    model = model_module.MultiHead01(backbone, nClasses, pretrained=False)

    # Load weights
    checkpointPath = modelDir + 'pytorch//Checkpoint35.pth.tar'
    checkpoint = torch.load(checkpointPath, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval() # ?

    return model


def createTfMainModel(saveKerasModel=False):

    model = loadPytorchModelAndWeights()

    imageWidth = int(imageWidthResizeFactor * segmentDuration * sampleRate / fftHopSizeInSamples)
    inputShape = (batchSize, 3, imageHeight, imageWidth)
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




###########################################################################################
###########################################################################################
###########################################################################################


createTfPreprocessingModel()
createTfMainModel(saveKerasModel=True)
combineTfModels()