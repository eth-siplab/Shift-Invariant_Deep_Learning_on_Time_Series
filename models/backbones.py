import torch
from torch import nn
from .attention import *
from .MMB import *
from .blurpool import *
from .apspool import *

class FCN(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, args=None):
        super(FCN, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
                                         )
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if n_channels == 9: # ucihar
            self.out_len = 13
        elif n_channels == 3: # shar
            self.out_len = 16
        if n_channels == 6: # hhar
            self.out_len = 9
        if n_channels == 1: # ppg
            self.out_len = 22 # was 13
        if n_channels == 4: # ecg
            self.out_len = 122
        if args.dataset == 'clemson':
            self.out_len = 57
        if args.dataset == 'physio':
            self.out_len = 747

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x

class FCN_big(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, args=None):
        super(FCN_big, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 64, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
                                         )
        self.conv_block2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(128, out_channels, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if n_channels == 9: # ucihar
            self.out_len = 13
        elif n_channels == 3: # shar
            self.out_len = 16
        if n_channels == 6: # hhar
            self.out_len = 9
        if n_channels == 1: # ppg
            self.out_len = 22 # was 13
        if n_channels == 4: # ecg
            self.out_len = 122
        if args.dataset == 'clemson':
            self.out_len = 57
        if args.dataset == 'physio':
            self.out_len = 747

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x

class FCN_blur(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, args=None):
        super(FCN_blur, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU()
                                         )
        
        self.blur_pool1 = BlurPool1D(channels=32, filt_size=2, stride=2, pad_off=0)

        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU())
        
        self.blur_pool2 = BlurPool1D(channels=64, filt_size=2, stride=2, pad_off=0)

        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU())

        self.blur_pool3 = BlurPool1D(channels=out_channels, filt_size=2, stride=2, pad_off=1) if n_channels == 9 or args.dataset == 'clemson' else BlurPool1D(channels=out_channels, filt_size=2, stride=2, pad_off=0)

        if n_channels == 9: # ucihar
            self.out_len = 13
        elif n_channels == 3: # shar
            self.out_len = 22
        if n_channels == 6: # hhar
            self.out_len = 9
        if n_channels == 1: # ppg
            self.out_len = 13
        if n_channels == 4: # ecg
            self.out_len = 127
        if args.dataset == 'clemson':
            self.out_len = 57

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.blur_pool1(x)
        x = self.conv_block2(x)
        x = self.blur_pool2(x)
        x = self.conv_block3(x)
        x = self.blur_pool3(x)

        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x

class FCN_aps(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, args=None):
        super(FCN_aps, self).__init__()

        self.backbone = backbone
        self.channels = n_channels

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU()
                                         )
        
        self.aps_pool1 = ApsPool(channels=32, filt_size=2, stride=2, return_poly_indices = False, circular_flag = True, N = None)

        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU())
        
        self.aps_pool2 = ApsPool(channels=64, filt_size=2, stride=2, return_poly_indices = False, circular_flag = True, N = None)

        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU())

        self.aps_pool3 = ApsPool(channels=out_channels, filt_size=2, stride=2) 

        if n_channels == 9: 
            self.out_len = 12
        elif n_channels == 3: # shar
            self.out_len = 22
        if n_channels == 6: # hhar
            self.out_len = 9
        if n_channels == 1: # ppg
            self.out_len = 13
        if n_channels == 4: # ecg
            self.out_len = 127
        if args.dataset == 'clemson':
            self.out_len = 56

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.aps_pool1(x)
        x = self.conv_block2(x)
        x = self.aps_pool2(x)
        x = self.conv_block3(x)
        x = self.aps_pool3(x)
        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x

#################################################
        
class FCN_controller(nn.Module):
    def __init__(self, n_channels, args=None):
        super(FCN_controller, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 4, kernel_size=8, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(4),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
                                         )
        self.conv_block2 = nn.Sequential(nn.Conv1d(4, 16, kernel_size=5, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(16),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, stride=1, bias=False, padding=1),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if args.dataset == 'ieee_big' or args.dataset == 'dalia' or args.dataset == 'ieee_small':
            self.logits = nn.Linear(1 * 416, 1)
        elif args.dataset == 'chapman':
            self.logits = nn.Linear(1 * 2016, 1)
        elif args.dataset == 'physio':
            self.logits = nn.Linear(1 * 12000, 1)
        elif args.dataset == 'ucihar':
            self.logits = nn.Linear(1 * 256, 1)
        elif args.dataset == 'hhar' or args.dataset == 'usc':
            self.logits = nn.Linear(1 * 224, 1)
        elif args.dataset == 'clemson':
            self.logits = nn.Linear(1 * 960, 1)
        elif args.dataset == 'sleep':
            self.logits = nn.Linear(1 * 6016, 1)
        elif args.dataset == 'respTR':
            self.logits = nn.Linear(1 * 64000, 1)
        
    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return (logits.squeeze())

class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True):
        super(DeepConvLSTM, self).__init__()

        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        #import pdb;pdb.set_trace();
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
##############################################################################
class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=2, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    
class UNET_encode(nn.Module):
    def __init__(self, input_dim, layer_n, kernel_size, depth, args, backbone=True):
        super(UNET_encode, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.backbone = backbone
        self.args = args

        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size, 1, 1)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n * 2), self.kernel_size, 2, 2)
        self.layer3 = self.down_layer(int(self.layer_n * 2) + int(self.input_dim), int(self.layer_n * 3),
                                      self.kernel_size, 2, 2)
        self.layer4 = self.down_layer(int(self.layer_n * 3) + int(self.input_dim), int(self.layer_n * 4),
                                      self.kernel_size, 2, 2)

        self.out_act = nn.ReLU()
        self.out_dim = 2048

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        return nn.Sequential(*block)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, channel, length)
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        
        #############Encoder#####################
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x1 = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x1)

        x2 = torch.cat([out_2, pool_x2], 1)
        x3 = self.layer4(x2)
        
        #############Decoder####################
        if self.backbone:
            return out_0, out_1, out_2, x3
        else:
            return None, x3
        
########################
class UNET_reconstruct(nn.Module):
    def __init__(self, layer_n, kernel_size, out_channels, depth):
        super(UNET_reconstruct, self).__init__()
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.cbr_up1 = conbr_block(int(self.layer_n * 7), int(self.layer_n * 3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n * 5), int(self.layer_n * 2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n * 3), self.layer_n, self.kernel_size, 1, 1)
        self.outcov = nn.Conv1d(self.layer_n, out_channels, kernel_size=self.kernel_size, stride=1, padding=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    
    def forward(self, out_0, out_1, out_2, x3):
        #############Decoder####################
        up = self.upsample(x3)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)
        # out1 = torch.tanh(out.squeeze())
        return out.squeeze()
    
##############################################################################
class LSTM(nn.Module):
    def __init__(self, n_channels, n_classes, LSTM_units=128, backbone=True):
        super(LSTM, self).__init__()

        self.backbone = backbone
        self.lstm = nn.LSTM(n_channels, LSTM_units, num_layers=2)
        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(1, 0, 2)
        x, (h, c) = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

class AE(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, outdim=128, backbone=True):
        super(AE, self).__init__()

        self.backbone = backbone
        self.len_sw = len_sw

        self.e1 = nn.Linear(n_channels, 8)
        self.e2 = nn.Linear(8 * len_sw, 2 * len_sw)
        self.e3 = nn.Linear(2 * len_sw, outdim)

        self.d1 = nn.Linear(outdim, 2 * len_sw)
        self.d2 = nn.Linear(2 * len_sw, 8 * len_sw)
        self.d3 = nn.Linear(8, n_channels)

        self.out_dim = outdim

        if backbone == False:
            self.classifier = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        x_d1 = self.d1(x_encoded)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.len_sw, 8)
        x_decoded = self.d3(x_d2)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded

class CNN_AE(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE, self).__init__()

        self.backbone = backbone
        self.n_channels = n_channels

        self.e_conv1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU())
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.dropout = nn.Dropout(0.35)

        self.e_conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU())
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv1 = nn.Sequential(nn.ConvTranspose1d(out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU())

        if n_channels == 9: # ucihar
            self.lin1 = nn.Linear(33, 34)
        elif n_channels == 6: # hhar
            self.lin1 = nn.Identity()
        elif n_channels == 3: # shar
            self.lin1 = nn.Linear(39, 40)

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(n_channels),
                                     nn.ReLU())

        if n_channels == 9: # ucihar
            self.lin2 = nn.Linear(127, 128)
            self.out_dim = 18 * out_channels
        elif n_channels == 6:  # hhar
            self.lin2 = nn.Linear(99, 100)
            self.out_dim = 15 * out_channels
        elif n_channels == 3: # shar
            self.out_dim = 21 * out_channels

        if backbone == False:
            self.classifier = nn.Linear(self.out_dim, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x_encoded, indice3 = self.pool3(self.e_conv3(x))
        x = self.d_conv1(self.unpool1(x_encoded, indice3))
        x = self.lin1(x)
        x = self.d_conv2(self.unpool2(x, indice2))
        x = self.d_conv3(self.unpool1(x, indice1))
        if self.n_channels == 9: # ucihar
            x_decoded = self.lin2(x)
        elif self.n_channels == 6 : # hhar
            x_decoded =self.lin2(x)
        elif self.n_channels == 3: # shar
            x_decoded = x
        x_decoded = x_decoded.permute(0, 2, 1)
        x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded

class Transformer(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
        super(Transformer, self).__init__()

        self.backbone = backbone
        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        if backbone == False:
            self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.transformer(x)
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

class Classifier(nn.Module):
    def __init__(self, bb_dim, n_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Linear(bb_dim, n_classes)

    def forward(self, x):
        out = self.classifier(x)

        return out

class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x

class Reconstructer(nn.Module):
    def __init__(self, model, layer_n, kernel_size, out_channels, depth=1):
        super(Reconstructer, self).__init__()
        if model == 'Auditory':
            self.Reconstructer = UNET_reconstruct(layer_n, kernel_size, out_channels, depth)
        else:
            raise NotImplementedError

    def forward(self, out0, out1, out2, x3):
        x = self.Reconstructer(out0, out1, out2, x3)
        return x    

class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


from functools import wraps


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, DEVICE, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.DEVICE = DEVICE

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        children = [*self.net.children()]
        print('children[self.layer]:', children[self.layer])
        return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = Projector(model='byol', bb_dim=dim, prev_dim=self.projection_hidden_size, dim=self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            x_decoded, representation = self.get_representation(x)
        else:
            _, representation = self.get_representation(x)

        if len(representation.shape) == 3:
            representation = representation.reshape(representation.shape[0], -1)

        projector = self._get_projector(representation)
        projection = projector(representation)
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            return projection, x_decoded, representation
        else:
            return projection, representation

class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.
    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548
    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    """

    def __init__(self, size: int = 2 ** 16):
        super(NNMemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank
        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it
        """

        output, bank = \
            super(NNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = \
            torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours

##################### ResNet-1D ############################

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            # import pdb;pdb.set_trace();
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, backbone=True, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.backbone = backbone
        self.out_dim = 64
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = x
        # first conv
        if self.verbose: print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out_logit = self.dense(out)
        if self.backbone:
            return None, out
        else:
            if self.verbose:
                print('dense', out.shape)
            # out = self.softmax(out)
            if self.verbose:
                print('softmax', out.shape)
            
            return out_logit, out     

class ResNet1D_SSL(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, backbone=True, use_bn=True, use_do=True, verbose=False, covariance_training=False):
        super(ResNet1D_SSL, self).__init__()
        
        self.verbose = verbose
        self.backbone = backbone
        self.out_dim = 64
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.covariance_training = covariance_training

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.final_linear = nn.Linear(200, 1)

        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = x

        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)

        out = self.final_relu(out)

        out = out.mean(-1)
        # out = self.final_linear(out).squeeze()
        out_logit = self.dense(out)
        
        if self.backbone:
            return None, out
        else:
            return out_logit, out 
        
##################### ResNet-1D Blur ############################

class MyConv1dPadSame_blur(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame_blur, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=1, 
            groups=self.groups)
        
        self.blurpool = BlurPool1D(channels=self.out_channels, filt_size=self.kernel_size, stride=self.stride, pad_off=0)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        net = self.blurpool(net)

        return net
        
class MyMaxPool1dPadSame_blur(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, channels, kernel_size):
        super(MyMaxPool1dPadSame_blur, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=1)
        self.max_pool_blur = BlurPool1D(channels=channels, filt_size=self.kernel_size, stride=self.kernel_size, pad_off=0)

    def forward(self, x):
        
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)
        net = self.max_pool_blur(net)
        
        return net
    
class BasicBlock_blur(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock_blur, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame_blur(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame_blur(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame_blur(channels=in_channels, kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x

        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            # import pdb;pdb.set_trace();
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

class ResNet1D_blur(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, backbone=True, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D_blur, self).__init__()
        
        self.verbose = verbose
        self.backbone = backbone
        self.out_dim = 32
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame_blur(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock_blur(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = x
        # first conv
        if self.verbose: print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out_logit = self.dense(out)
        if self.backbone:
            return None, out
        else:
            if self.verbose:
                print('dense', out.shape)
            # out = self.softmax(out)
            if self.verbose:
                print('softmax', out.shape)
            
            return out_logit, out 
        
##################### ResNet-1D APS ############################

class MyConv1dPadSame_aps(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame_aps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=1, 
            groups=self.groups)
        
        self.apspool = ApsPool(channels=self.out_channels, filt_size=self.kernel_size, stride=self.stride, return_poly_indices = False, circular_flag = True, N = None)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        net = self.apspool(net)

        return net
        
class MyMaxPool1dPadSame_aps(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, channels, kernel_size):
        super(MyMaxPool1dPadSame_aps, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=1)
        self.max_pool_blur = BlurPool1D(channels=channels, filt_size=self.kernel_size, stride=self.kernel_size, pad_off=0)
        self.max_pool_aps = ApsPool(channels=channels, filt_size=self.kernel_size, stride=self.kernel_size, return_poly_indices = False, circular_flag = True, N = None)

    def forward(self, x):
        
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0) # Circular padding
        net = self.max_pool(net)
        net = self.max_pool_aps(net)
        
        return net
    
class BasicBlock_aps(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock_aps, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame_aps(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame_aps(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame_aps(channels=in_channels, kernel_size=self.stride)

    def forward(self, x):
        identity = x
        # import pdb;pdb.set_trace();
        # the first conv
        out = x

        if not self.is_first_block:
            if self.use_bn: out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do: out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D_aps(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, backbone=True, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D_aps, self).__init__()
        
        self.verbose = verbose
        self.backbone = backbone
        self.out_dim = 32
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame_aps(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock_aps(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = x
        # first conv
        if self.verbose: print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out_logit = self.dense(out)
        if self.backbone:
            return None, out
        else:
            if self.verbose:
                print('dense', out.shape)
            # out = self.softmax(out)
            if self.verbose:
                print('softmax', out.shape)
            
            return out_logit, out 
        
##################### My ReLU ############################
        
class myReLU(nn.Module):
    def __init__(self):
        super(myReLU, self).__init__()
        
    def forward(self, x):
        return F.relu(x)

class MyConv1dPadSame_aReLU(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame_aReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame_aReLU(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame_aReLU, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock_aReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock_aReLU, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame_aReLU(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame_aReLU(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame_aReLU(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            # import pdb;pdb.set_trace();
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D_aReLU(nn.Module):

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, backbone=True, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D_aReLU, self).__init__()
        
        self.verbose = verbose
        self.backbone = backbone
        self.out_dim = 32
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame_aReLU(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock_aReLU(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = x
        # first conv
        if self.verbose: print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out_logit = self.dense(out)
        if self.backbone:
            return None, out
        else:
            if self.verbose:
                print('dense', out.shape)
            # out = self.softmax(out)
            if self.verbose:
                print('softmax', out.shape)
            
            return out_logit, out 
        
############## TWaveNet model ################
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction

        self.conv_even = lambda x: x[:, ::2, :]
        self.conv_odd = lambda x: x[:, 1::2, :]


    def forward(self, x):
        '''Returns the odd and even part'''
        # if not even, pad the input with last item
        if x.size(1) % 2 != 0:
            x = torch.cat((x, x[:,-1:, :]), dim=1)
        return (self.conv_even(x), self.conv_odd(x))

class LiftingScheme(nn.Module):
    def __init__(self, in_planes, modified=False, size=[], splitting=True, k_size=4, dropout = 0, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = True

        # kernel_size = k_size
        kernel_size = 3
        dilation = 1

        pad = dilation * (kernel_size - 1) // 2 +1
        # pad = k_size // 2 # 2 1 0 0

        self.splitting = splitting
        self.split = Splitting()

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 2

            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation,stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #    nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
             #   nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            if self.modified:
                modules_phi += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                #nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
                modules_psi += [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]

            self.phi = nn.Sequential(*modules_phi)
            self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)
#        self.phi = nn.Sequential(*modules_phi)
#        self.psi = nn.Sequential(*modules_psi)


    def forward(self, x):
        if self.splitting:
            #3  224  112
            #3  112  112
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if not self.modified:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # x_odd = self.ptemp(x_odd)
            # x_odd =self.U(x_odd) #18 65
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c) #         Todo: +  -> * -> sigmod
#            d = x_odd - self.P(x_even)
#            c = x_even + self.U(d)

            # c = x_even + self.seNet_P(x_odd)
            # d = x_odd - self.seNet_P(c)
            return (c, d)
        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # a = self.phi(x_even)
            d = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            c = x_even.mul(torch.exp(self.psi(d))) + self.U(d)
            return (c, d)

class LiftingSchemeLevel(nn.Module):
    def __init__(self, in_planes, share_weights, modified=False, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingSchemeLevel, self).__init__()
        self.level = LiftingScheme(
             in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)


    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (L, H) = self.level(x)  #10 3 224 224

        return (L, H)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv=True):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
#        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.disable_conv = disable_conv
        if not self.disable_conv:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))

class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:

            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingSchemeLevel(in_planes, share_weights,
                                       size=lifting_size, kernel_size=kernel_size,
                                       simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=True)
        else:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=False)

    def forward(self, x):
        (L, H) = self.wavelet(x) #10 9 128
        approx = L
        details = H

        r = None
        if(self.regu_approx + self.regu_details != 0.0):  #regu_details=0.01, regu_approx=0.01

            if self.regu_details:
                rd = self.regu_details * \
                     H.abs().mean()


            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)


            if self.regu_approx == 0.0:
                # Only the details
                r = rd
            elif self.regu_details == 0.0:
                # Only the approximation
                r = rc
            else:
                # Both
                r = rd + rc
        if self.bootleneck:
            return self.bootleneck(approx).permute(0, 2, 1), r, details
        else:
            return approx.permute(0, 2, 1), r, details

class TWaveNet(nn.Module):
    def __init__(self, num_classes, big_input=True, first_conv=9, extend_channel = 32, # extend_channel = 16 for other 48 for sleep
                 number_levels=4, number_level_part=[[1, 0], [1, 0], [1, 0]],
                 lifting_size=[2, 1], kernel_size=4, no_bootleneck=True,
                 classifier="mode2", share_weights=False, simple_lifting=False,
                  regu_details=0.01, regu_approx=0.01, haar_wavelet=False, backbone=False):
        super(TWaveNet, self).__init__()
        self.backbone = backbone
        self.initialization = False
        self.nb_channels_in = first_conv
        self.level_part = number_level_part

        # First convolution
        if first_conv != 3 and first_conv != 6 and first_conv != 22:
            self.first_conv = True
            self.conv1 = nn.Sequential(
                nn.Conv1d(first_conv, extend_channel,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(extend_channel),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(extend_channel, extend_channel,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(extend_channel),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.5),
            )
            in_planes = extend_channel
            out_planes = extend_channel * (number_levels + 1)
        else:
            self.first_conv = False
            in_planes = first_conv
            out_planes = first_conv * (number_levels + 1)


        self.levels = nn.ModuleList()


        for i in range(number_levels):
            # bootleneck = True
            # if no_bootleneck and i == number_levels - 1:
            #     bootleneck = False
            if i == 0:

                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, no_bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelTWaveNet(in_planes,
                                  lifting_size, kernel_size, no_bootleneck,
                                  share_weights, simple_lifting, regu_details, regu_approx)
                    )
            else:
                if haar_wavelet:
                    self.levels.add_module(
                        'level_' + str(i),
                        Haar(in_planes,
                             lifting_size, kernel_size, no_bootleneck,
                             share_weights, simple_lifting, regu_details, regu_approx)
                    )
                else:
                    self.levels.add_module(
                        'level_' + str(i),
                        LevelTWaveNet(in_planes,
                                  lifting_size, kernel_size, no_bootleneck,
                                  share_weights, simple_lifting, regu_details, regu_approx)
                    )
            in_planes *= 1


            out_planes += in_planes * 3

        if no_bootleneck:
            in_planes *= 1

        self.num_planes = out_planes


        if classifier == "mode1":
            self.fc = nn.Linear(out_planes, num_classes)
        elif classifier == "mode2":

            self.fc = nn.Sequential(
                nn.Linear(in_planes*(number_levels + 1), 1024),  # Todo:  extend channels
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(1024, num_classes)

            )
        else:
            raise "Unknown classifier"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                m.bias.data.zero_()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.atten = MultiHeadAttention(n_head=16, d_model=in_planes, d_k=32, d_v=32, dropout=0, share_weight=False, temp=False)
        self.count_levels = 0

    def forward(self, x):
        if self.first_conv:
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = x.permute(0, 2, 1)
        rs = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        input = [x, ]
        for l in self.levels:
            low, r, details = l(input[0])
            if self.level_part[self.count_levels][0]:
                input.append(low)
            else:
                low = low.permute(0, 2, 1)
                det += [self.avgpool(low)]
            if self.level_part[self.count_levels][1]:
                details = details.permute(0, 2, 1)
                input.append(details)
            else:
                det += [self.avgpool(details)]
            del input[0]
            rs += [r]
            self.count_levels = self.count_levels + 1

        for aprox in input:
            aprox = aprox.permute(0, 2, 1)  # b 77 1
            aprox = self.avgpool(aprox)
            det += [aprox]

        self.count_levels = 0
        # We add them inside the all GAP detail coefficients
        x = torch.cat(det, 2) #[b, 77, 8]
        x = x.permute(0, 2, 1)
        q, att = self.atten(x, x, x, mask=None)
        x = q
        b, c, l = x.size()
        x = x.view(-1, c * l)
        #
        # det += [aprox]
        # x = torch.cat(det, 2)
        # b, c, l = x.size()
        # x = x.view(-1, c * l)
        return self.fc(x), rs

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0, share_weight=False, temp=False):
        super().__init__()
        self.share_weight = share_weight
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        if share_weight:
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        else:
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        if temp:
            self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        else:
            self.attention = ScaledDotProductAttention(temperature=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        if  self.share_weight:
            k = self.w_qs(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_qs(v).view(sz_b, len_v, n_head, d_v)
        else:
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class Spectral(nn.Module):
    def __init__(self):
        super(Spectral, self).__init__()

        self.opt_INP_DIM = 1
        self.opt_OUT_DIM = 1
        self.opt_WIDTH = 256
        self.opt_DEPTH = 6
        self.model = self.make_model()

    def make_model(self):
        layers = []
        layers.append(nn.Linear(self.opt_INP_DIM, self.opt_WIDTH))
        layers.append(nn.ReLU())
        for _ in range(self.opt_DEPTH - 2): 
            layers.append(nn.Linear(self.opt_WIDTH, self.opt_WIDTH))
            layers.append(nn.ReLU())
        layers.extend([nn.Linear(self.opt_WIDTH, self.opt_OUT_DIM)])
        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        out = self.model(x)
        return out, None

class FreTS(nn.Module):
    def __init__(self, num_class, feature_size, seq_length, backbone=False):
        super(FreTS, self).__init__()
        self.embed_size = 16 #embed_size 128
        self.hidden_size = 48 #hidden_size 256
        self.class_size = num_class
        self.backbone = backbone
        # self.pre_length = configs.pred_len  --> No prediction
        self.feature_size = feature_size #channels
        self.seq_length = seq_length
        self.channel_independence = '0'
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size * self.feature_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.class_size)
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size, self.class_size)
        # )        

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        # import pdb;pdb.set_trace();
        # x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        # x = x.mean(-1)
        x = self.fc(x.reshape(B, -1))
        return x.squeeze(), None

class multi_rate_2(nn.Module):
    def __init__(self, num_classes, sample_length=200, channels_size=9, n_block=6, mag_ratio=0.3, args=None):
        super(multi_rate_2, self).__init__()

        self.freq = torch.fft.rfftfreq(n=sample_length)
        self.project = ProjectionModule(in_channels=channels_size, freq=self.freq, mag_ratio=mag_ratio, args=args)
        self.second_model = ResNet1D(in_channels=channels_size, base_filters=32, kernel_size=5, stride=2, groups=1, n_block=n_block, n_classes=num_classes, downsample_gap=2, increasefilter_gap=4, backbone=False)
        # self.second_model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)

    def forward(self, x):
        mod_x, res_loss = self.project(x)
        out, _ = self.second_model(mod_x)
        return out, res_loss    

class ProjectionModule(nn.Module):
    def __init__(self, in_channels, freq, mag_ratio=0.3, args=None):
        super(ProjectionModule, self).__init__()
        self.in_channels = in_channels
        self.freq = freq
        self.mag_ratio = mag_ratio
        # Define the weight matrix as a trainable parameter
        self.weight_mag = nn.Parameter(torch.rand(in_channels, 101) + 0.5)  # Sampled between 0.5 and 1.5
        self.weight_angle = nn.Parameter(torch.rand(in_channels, 1) * 2 * math.pi - math.pi)  # Sampled between -pi and pi
        # self.mag_cont = mag_cont(input_size=100, hidden_size=100)
        self.mag_cont = mag_cont(input_size=self.freq.size(0), hidden_size=self.freq.size(0))
        self.angle_cont = angle_cont(input_size=1, hidden_size=10)
        # self.together_cont = together_cont(input_size=self.freq.size(0), hidden_size=self.freq.size(0))

    def calculate_divergence(self, fft_orig, fft_new):
        # normalize to sum 1
        freq_interest_org = fft_orig / torch.sum(fft_orig, dim=1, keepdim=True)
        freq_interest = fft_new / torch.sum(fft_new, dim=1, keepdim=True)
        kl_loss = nn.KLDivLoss(reduction='sum')
        if torch.min(freq_interest) == 0 or torch.min(freq_interest_org) == 0:
            freq_interest = freq_interest + 1e-10
            freq_interest_org = freq_interest_org + 1e-10
        loss = kl_loss(torch.log(freq_interest), freq_interest_org)
        return 0.5*loss
    
    def von_mises_sample(self, mu, kappa):
        # Sample from von Mises distribution
        m = torch.distributions.von_mises.VonMises(loc=mu, concentration=kappa)
        sampled_angles = m.sample()
        return sampled_angles

    # Custom loss function for von Mises distribution
    def von_mises_loss(self, kappa):
        loss = -torch.mean(kappa)
        return loss
    
    def normalize(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = x / torch.std(x, dim=1, keepdim=True)
        return x

    def forward(self, x):
        # import pdb;pdb.set_trace();
        x_fft = torch.fft.rfft(x, n=x.size(1), dim=1, norm='ortho')  
        mag_fft = torch.abs(x_fft)
        angle_fft = torch.angle(x_fft)
        # mod_mag = mag_fft
        mod_mag = torch.empty_like(mag_fft)
        mod_mag = mag_fft.clone()
        for i in range(mag_fft.size(2)):
            # indices = torch.topk(mag_fft[:,:,i], k=100)
            # expand_ind = indices[1].unsqueeze(2)
            # modified_values  = indices[0] + self.mag_ratio * self.mag_cont(indices[0].unsqueeze(2))
            # mod_mag.scatter_(1, expand_ind, modified_values.unsqueeze(2))
            #
            mod_mag[:,:,i] = mag_fft[:,:,i] + self.mag_ratio * self.mag_cont(mag_fft[:,:,i].unsqueeze(2))
            # Ablation without modifying magnitude 
            # mod_mag[:,:,i] = mag_fft[:,:,i] 

        # residual_loss = torch.mean(torch.abs(mag_fft - mod_mag))
        # residual_loss = torch.nn.functional.mse_loss(mag_fft, mod_mag)
        # residual_loss = 0 
        residual_loss = self.calculate_divergence(mag_fft, mod_mag)
        ##
        params = self.angle_cont(angle_fft[:,:,0].unsqueeze(2))
        angles = self.von_mises_sample(params[0], params[1])
        # 
        angle_to_add = (2*torch.pi*self.freq).unsqueeze(1).expand_as(angle_fft)
        mod_angle = angle_fft + angles[:,None,None] * angle_to_add.to(angle_fft.device)
        ## Ablation without modifying the phase
        # mod_angle = angle_fft
        #
        z =  torch.polar(mod_mag, mod_angle) # Go back to polar coordinates
        mixed_samples_time = torch.fft.irfft(z, n=x.size(1), dim=1, norm='ortho')
        # mixed_samples_time = self.normalize(mixed_samples_time)
        return mixed_samples_time, residual_loss

class mag_cont(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mag_cont, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.fc1(x))
        weights = torch.sigmoid(self.fc2(x))  # Ensure outputs are between 0 and 1
        weights = 0.5 + weights  # Shift and scale
        return weights.transpose(1, 2).squeeze()


class angle_cont(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=1):
        super(angle_cont, self).__init__()
        self.learning = True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.von_mises = VonMisesLayer()

    def forward(self, x):
        if self.learning:
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
            mu, kappa = out[:, 0], self.relu(out[:, 1]) + 0.5
        else:
            mu, kappa = self.von_mises(x)
        return mu, kappa

class VonMisesLayer(nn.Module):
    def __init__(self):
        super(VonMisesLayer, self).__init__()
        """
        Compute von Mises distribution parameters for a batch of angle tensors.
        
        Args:
        - x (torch.Tensor): Batch of angle tensors with shape (batch_size, angles)
        
        Returns:
        - mu (torch.Tensor): Mean direction for each element in the batch with shape (batch_size,)
        - kappa (torch.Tensor): Concentration parameter for each element in the batch with shape (batch_size,)
        """

    def forward(self, x):
        # Mean direction (μ)
        sin_mean = torch.sin(x).mean(dim=1)  
        cos_mean = torch.cos(x).mean(dim=1)  
        mu = torch.atan2(sin_mean, cos_mean)  # Compute arctangent of the mean sine/cosine values

        # Resultant length (R)
        sin_sq_mean = torch.sin(x).pow(2).mean(dim=1)  # Compute mean of squared sine values along the angle dimension
        cos_sq_mean = torch.cos(x).pow(2).mean(dim=1)  # Compute mean of squared cosine values along the angle dimension
        R = torch.sqrt(sin_sq_mean + cos_sq_mean)  # Compute square root of the sum of squared means

        # Concentration parameter (κ)
        kappa = torch.log((R + torch.sqrt(R.pow(2) + 1e-8)) / (1 - R + torch.sqrt((1 - R).pow(2) + 1e-8)))

        return mu, kappa
    

