import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer

import torchaudio

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

nLowFreqsInPixelToCutMax=2
nHighFreqsInPixelToCutMax=4

specImageMean = [0.5, 0.4, 0.3]
specImageStd = [0.5, 0.3, 0.1]



def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):


        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    # NNN
    # def nonlinear_transform(self, x):
    #     if self.activation == 'linear':
    #         return x
    #     elif self.activation == 'sigmoid':
    #         return torch.sigmoid(x)
    def nonlinear_transform(self, x: torch.Tensor) -> torch.Tensor:
        # Add explicit type annotation and ensure we always return a Tensor
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            # Add a default case to ensure TorchScript knows a Tensor is always returned
            return x
        
class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        #print('x.shape', x.shape) # torch.Size([8, 1280, 7, 18]) --> batchsize, channels, height, width
        x = torch.nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1,1)).pow(1.0 / self.p)
        #print('self.p', self.p)
        #x = x.view(x.shape[0], x.shape[1]) # old way (bakes batch dim during export)
        x = x.flatten(start_dim=1) # new way (dynamic batch dim)
        #x = x.view(x.shape[0], -1)
        return x


# Approximates tf preprocessing
class PreprocessingLayerTorch(nn.Module):
    def __init__(
            self,
            fftSizeInSamples=fftSizeInSamples,
            fftHopSizeInSamples=fftHopSizeInSamples,
            melBasisPath='melBasis.npy',
            ref=1.0,
            amin=1e-10,
            top_db=100.0,
            imageHeight=imageHeight,
            imageWidthResizeFactor=imageWidthResizeFactor,
            device='cpu'
            ):

        super().__init__()

        self.torchSpecTransform = torchaudio.transforms.Spectrogram(
            n_fft=fftSizeInSamples, 
            hop_length=fftHopSizeInSamples, 
            power=2.0
        )
        
        melBasisPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), melBasisPath) # Hack
        self.melBasis = torch.from_numpy(np.load(melBasisPath).T).to(device) # Could be problematic for torchscript export
        self.amin = amin
        self.ref_value = torch.abs(torch.tensor(ref, device=device)) # Ensure ref_value is on the same device
        self.ref_value_log10 = torch.log10(torch.clamp(self.ref_value, min=self.amin))
        self.top_db = top_db

        self.nLowFreqsInPixelToCut = int(nLowFreqsInPixelToCutMax/2.0)
        self.nHighFreqsInPixelToCut = int(nHighFreqsInPixelToCutMax/2.0)

        self.imageHeight = int(imageHeight)
        self.imageWidthResizeFactor = imageWidthResizeFactor
        self.fftHopSizeInSamples = fftHopSizeInSamples
        
        self.epsilon = 1e-10
        self.ten = torch.tensor(10.0, device=device)
        
        # Replace torchvision.transforms.Normalize with manual normalization parameters for onnx export
        self.register_buffer('mean', torch.tensor(specImageMean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(specImageStd, dtype=torch.float32).view(1, 3, 1, 1))


    def forward(self, sampleVec):
        
        ## FFT
        spec = self.torchSpecTransform(sampleVec)
        
        ## Mel spectrogram
        melSpec = torch.tensordot(spec, self.melBasis, dims=([1], [0])).transpose(1, 2)
        
        ## Scale to dB
        melSpec.clamp_(min=self.amin).log10_().mul_(self.ten).sub_(self.ref_value_log10)
        melSpec.clamp_(min=melSpec.amax(dim=(1,2), keepdim=True) - self.top_db)
        
        ### Post-processing
        
        ## Cut low and high frequencies and flip vertically (only for better visialization, low freq. at bottom)
        melSpec = melSpec[:, self.nLowFreqsInPixelToCut:-self.nHighFreqsInPixelToCut, :].flip(dims=[1])
        

        ## Normalize between 0.0 and 1.0 per sample (not batch) and invert colors
        batch_size = melSpec.size(0)
        flattened = melSpec.reshape(batch_size, -1)
        min_vals, max_vals = torch.aminmax(flattened, dim=1, keepdim=True)
        min_vals = min_vals.view(batch_size, 1, 1)
        max_vals = max_vals.view(batch_size, 1, 1)
        melSpec = (max_vals - melSpec).div_(max_vals - min_vals + self.epsilon)
        

        ## Resize
        melSpec = melSpec.unsqueeze(1) # Add channel dimension: (bs, nFreqs, nFrames) --> (bs, 1, nFreqs, nFrames)

        imageWidth = int(self.imageWidthResizeFactor * sampleVec.shape[1] // self.fftHopSizeInSamples)

        melSpec = torch.nn.functional.interpolate(
            melSpec, 
            size=(self.imageHeight, imageWidth), 
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        
        # # Compute imageWidth dynamically — use scale_factor instead of size
        # # This avoids baking in a constant output size; ONNX Resize will use
        # # scales relative to the input, which adapts to any input length.
        # scale_h = self.imageHeight / melSpec.shape[2]  # target_h / current_h
        # scale_w = self.imageWidthResizeFactor * sampleVec.shape[1] / (self.fftHopSizeInSamples * melSpec.shape[3])

        # melSpec = torch.nn.functional.interpolate(
        #     melSpec,
        #     scale_factor=(scale_h, scale_w),
        #     mode='bilinear',
        #     align_corners=False,
        #     antialias=False
        # )


        # # Compute imageWidth dynamically using integer tensor arithmetic
        # # so that dynamo export preserves it as a dynamic computation
        # imageWidth = torch.div(
        #     torch.tensor(self.imageWidthResizeFactor, dtype=torch.float32, device=sampleVec.device) * sampleVec.shape[1],
        #     self.fftHopSizeInSamples,
        #     rounding_mode='trunc'
        # )
        # imageWidth = imageWidth.to(torch.int64)

        # # Use size= with a list of ints for interpolate
        # # .item() converts tensor to Python int — during dynamo tracing this
        # # is captured as a dynamic symbolic expression, NOT constant-folded
        # melSpec = torch.nn.functional.interpolate(
        #     melSpec, 
        #     size=(self.imageHeight, imageWidth.item()), 
        #     mode='bilinear',
        #     align_corners=False,
        #     antialias=False
        # )




        ## Expand and return normalized
        melSpec = melSpec.expand(-1, 3, -1, -1) # Expand to 3 channels: (bs, 1, nFreqs, nFrames) --> (bs, 3, nFreqs, nFrames)
        return (melSpec - self.mean) / self.std  # Manual normalization for onnx export


class MultiHead02(nn.Module):

    def __init__(self, backbone, nClasses, pretrained=False, includePreprocessing=False, device='cpu'):
        super().__init__()

        if includePreprocessing:
            self.preprocess = PreprocessingLayerTorch(device=device)
        else:
            self.preprocess = torch.nn.Identity()


        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')


        # Get number of input features for the classifier
        #dummyInput = torch.randn(1, 3, imageHeight, imageWidth)
        dummyInput = torch.randn(1, 3, 64, 64)
        dummyOutput = self.encoder(dummyInput)
        in_features = dummyOutput.shape[1]
        print('in_features', in_features)
        print('dummyOutput.shape', dummyOutput.shape)

        # print('self.encoder')
        # print(self.encoder)

        # Head 1 (Bc24Cnn)
        self.global_pool = GeM()
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.head = torch.nn.Linear(in_features, nClasses)


        # Head 2 (FED wihout BatchNorm)
        self.fc1 = nn.Linear(in_features, 2048, bias=True)
        self.att_block = AttBlock(2048, nClasses, activation="linear")

        # Head 3 (SED with BatchNorm)
        self.fc2 = nn.Linear(in_features, 2048, bias=True)
        self.att_block2 = AttBlock(2048, nClasses, activation="linear")
        
        
        self.init_weight()

        # Fixed depthwise conv kernel for avg_pool1d (used by _pool1d_avg)
        # persistent=False: not saved/loaded from checkpoints (not a learned parameter)
        self.register_buffer('_avg_kernel', torch.ones(in_features, 1, 3) / 3.0, persistent=False)

    ## 1D pooling — avoiding both SLICE ops and unsupported StableHLO lowerings:
    ## - F.max_pool1d  → tfl.max_pool_2d ✅ (works natively)
    ## - F.avg_pool1d  → vhlo.reduce_window ❌ (not supported by TFLite converter)
    ## - Manual slicing → tfl.slice ❌ (bakes batch=1 during tracing)
    ## Solution: avg_pool1d via depthwise conv1d with fixed 1/3 weights.
    def _pool1d_avg(self, x):
        """avg_pool1d via depthwise conv1d with uniform 1/3 weights. x: [B, C, T]"""
        return F.conv1d(x, self._avg_kernel, padding=1, groups=self._avg_kernel.shape[0])

    def _pool1d_max(self, x):
        """max_pool1d with kernel_size=3, stride=1, padding=1. x: [B, C, T]"""
        return F.max_pool1d(x, kernel_size=3, stride=1, padding=1)


    
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
    
    def forward(self, input):

        #x = input
        #print('input', x.shape) # torch.Size([48, 3, 224, 557])

        # If include_preprocessing is False, this operates as x = input
        x = self.preprocess(input)


        features = self.encoder(x)
        #print('x = self.encoder(x)', x.shape) # torch.Size([48, 1408, 7, 18])

        ## Head 1
        embeddings = self.global_pool(features)
        #print('embeddings.shape', embeddings.shape) # torch.Size([s31, 1408])
        
        #logitsHead1 = sum([self.head(dropout(x)) for dropout in self.dropouts]) / 5
        
        # Use enumeration for TorchScript compatibility
        logitsHead1 = torch.zeros_like(self.head(self.dropouts[0](embeddings)))
        for dropout in self.dropouts:
            logitsHead1 = logitsHead1 + self.head(dropout(embeddings))
        logitsHead1 = logitsHead1 / len(self.dropouts)

        ## Head 2 
        
        #print('self.encoder.forward_features', x.shape)     # torch.Size([48, 1408, 7, 18]) 
        x = torch.mean(features, dim=3)
        #print('torch.mean(x, dim=3)', x.shape)              # torch.Size([48, 1408, 7])     


        ## Custom 1D pooling (for tflite compatibility)
        # x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        # x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x1 = self._pool1d_max(x)
        x2 = self._pool1d_avg(x)

        x = x1 + x2 # ToCheck if concat or using only max or avg is better
        #print('x = x1 + x2', x.shape)  # torch.Size([48, 1408, 7]) 

        x = F.dropout(x, p=0.4, training=self.training)
        #print('01', x.shape) # torch.Size([8, 1408, 7])
        x = x.transpose(1, 2)
        #print('02', x.shape) # torch.Size([8, 7, 1408])
        x = F.relu_(self.fc1(x))
        #print('03', x.shape) # torch.Size([8, 7, 2048])
        x = x.transpose(1, 2)
        #print('04', x.shape) # torch.Size([8, 2048, 7])
        x = F.dropout(x, p=0.4, training=self.training)
        #print('F.dropou', x.shape) # torch.Size([8, 2048, 7])

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        #print('logit', logit.shape) # torch.Size([8, 254])
        segmentwise_output = segmentwise_output.transpose(1, 2)
        #print('segmentwise_output', segmentwise_output.shape) # torch.Size([8, 7, 254])

        logitsHead2 = logit
        logitsHead2PerTime = segmentwise_output

        ## Head 3model-converter/models/01_BirdNET+_V3.0-preview3_Global_11K/pytorch/model_bu4.py

        #print('self.encoder.forward_features', x.shape)     # torch.Size([48, 1408, 7, 18]) 
        x = torch.mean(features, dim=2)
        #print('torch.mean(x, dim=2)', x.shape)              # torch.Size([8, 1408, 18])   


        ## Custom 1D pooling (for tflite compatibility)
        # x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        # x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x1 = self._pool1d_max(x)
        x2 = self._pool1d_avg(x)

        x = x1 + x2


        x = F.dropout(x, p=0.4, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc2(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.4, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block2(x)
        logit = torch.sum(norm_att * self.att_block2.cla(x), dim=2)
        #print('logit', logit.shape) # torch.Size([8, 254])
        segmentwise_output = segmentwise_output.transpose(1, 2)
        #print('segmentwise_output', segmentwise_output.shape) # torch.Size([8, 18, 254])

        logitsHead3 = logit
        logitsHead3PerTime = segmentwise_output


        # output_dict = {
        #     'logitsHead1' : logitsHead1,
        #     'logitsHead2' : logitsHead2,
        #     'logitsHead2PerTime' : logitsHead2PerTime,
        #     'probsHead2Max': torch.sigmoid(torch.max(logitsHead2PerTime, dim=1)[0]),
        #     'logitsHead3' : logitsHead3,
        #     'logitsHead3PerTime' : logitsHead3PerTime,
        #     'probsHead3Max': torch.sigmoid(torch.max(logitsHead3PerTime, dim=1)[0])
        # }

        # return output_dict

        # Single output for TorchScript export
        #return torch.sigmoid((logitsHead1 + logitsHead2 + logitsHead3) / 3.0)

        return torch.sigmoid((logitsHead1 + logitsHead2 + logitsHead3) / 3.0), embeddings

# Tf preprocessing as Keras layer
class Preprocessing(Layer):

    def __init__(self, name=None, **kwargs):

        self.sampleRate=sampleRate
        self.fftSizeInSamples=fftSizeInSamples
        self.fftHopSizeInSamples=fftHopSizeInSamples
        self.nMelBands=nMelBands
        self.melStartFreq=melStartFreq
        self.melEndFreq=melEndFreq


        self.nLowFreqsInPixelToCut = int(nLowFreqsInPixelToCutMax/2.0)
        self.nHighFreqsInPixelToCut = int(nHighFreqsInPixelToCutMax/2.0)


        self.specImageMean = [0.5, 0.4, 0.3]
        self.specImageStd = [0.5, 0.3, 0.1]

        self.imageHeight = imageHeight
        self.imageWidthResizeFactor = imageWidthResizeFactor

        self.topDb = 100

        # Get mel basis matrix from file (created with librosa)
        #melBasis = np.load(modelDir + 'pytorch/melBasis.npy')
        #melBasis = np.load('melBasis.npy')
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        melBasisPath = os.path.join(scriptDir, 'melBasis.npy')
        melBasis = np.load(melBasisPath)
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
        
        # Per sample normalization !!!
        # melSpecTf = tf.math.maximum(melSpecTf, tf.math.reduce_max(melSpecTf) - self.topDb)
        melSpecTf = tf.math.maximum(melSpecTf, tf.math.reduce_max(melSpecTf, axis=[1, 2], keepdims=True) - self.topDb)

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