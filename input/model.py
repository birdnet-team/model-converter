import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm




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

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


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

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        


# multihead
class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        #print('x.shape', x.shape) # torch.Size([8, 1280, 7, 18]) --> batchsize, channels, height, width
        x = torch.nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1,1)).pow(1.0 / self.p)
        #print('self.p', self.p)
        x = x.view(x.shape[0], x.shape[1])
        #x = x.view(x.shape[0], -1)
        return x


# multihead
class MultiHead01(nn.Module):

    def __init__(self, backbone, nClasses, pretrained=True):
        super().__init__()


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


    
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
    
    def forward(self, input):

        x = input
        #print('input', x.shape) # torch.Size([48, 3, 224, 557])


        # Not needed if already 3 channel
        #x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3]) # torch.Size([48, 1, 224, 557]) --> torch.Size([48, 3, 224, 557])
        #print('expand', x.shape) # torch.Size([48, 3, 224, 557])
        
        features = self.encoder(x)
        #print('x = self.encoder(x)', x.shape) # torch.Size([48, 1408, 7, 18])

        ## Head 1
        x = self.global_pool(features)
        logitsHead1 = sum([self.head(dropout(x)) for dropout in self.dropouts]) / 5


        ## Head 2 
        
        #print('self.encoder.forward_features', x.shape)     # torch.Size([48, 1408, 7, 18]) 
        x = torch.mean(features, dim=3)
        #print('torch.mean(x, dim=3)', x.shape)              # torch.Size([48, 1408, 7])     

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
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

        ## Head 3

        #print('self.encoder.forward_features', x.shape)     # torch.Size([48, 1408, 7, 18]) 
        x = torch.mean(features, dim=2)
        #print('torch.mean(x, dim=2)', x.shape)              # torch.Size([8, 1408, 18])   

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
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
        return torch.sigmoid((logitsHead1 + logitsHead2 + logitsHead3) / 3.0)


