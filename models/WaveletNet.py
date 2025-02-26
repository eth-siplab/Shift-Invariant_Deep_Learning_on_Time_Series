# torch
import torch
import torch.nn as nn
import numpy as np
# built-in
import functools
# project
import eerie

class WaveletNet(torch.nn.Module):
    def __init__(self, use_bias=False, args=None):
        super(WaveletNet, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 12
        n_classes = args.n_class
        in_channels = args.n_feature
        
        if args.dataset == 'physio':
            out_space_dim = 46
        elif args.dataset == 'chapman':
            out_space_dim = 7
        elif args.dataset == 'ucihar':
            out_space_dim = 1
        elif args.dataset == 'hhar':
            out_space_dim = 3
        elif args.dataset == 'clemson':
            out_space_dim = 21
        elif args.dataset == 'sleep':
            out_space_dim = 182
        elif args.dataset == 'ieee_big' or args.dataset == 'dalia':
            out_space_dim = 9
        else:
            out_space_dim = 1

        dp_rate = 0.25

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        # print(h_grid_RdG.grid)

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        # print(h_grid_crop.grid)

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=in_channels,             out_channels=n_channels,     kernel_size=7, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group, in_channels=n_channels,     out_channels=n_channels * 2,  kernel_size=5, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group, in_channels=n_channels * 2, out_channels=n_channels * 4,  kernel_size=5, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c4 = eerie.nn.GConvGG(group, in_channels=n_channels * 4, out_channels=n_channels * 8,  kernel_size=5,  h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c5 = eerie.nn.GConvGG(group, in_channels=n_channels * 8, out_channels=n_channels * 16, kernel_size=3,  h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)

        # Fully connected
        # self.f1 = torch.nn.Linear(in_features=n_channels * 16 * out_space_dim, out_features= n_channels * 8, bias=True)
        self.f1 = torch.nn.Linear(in_features=n_channels * 8 * out_space_dim, out_features= n_channels * 8, bias=True)
        self.f2 = torch.nn.Linear(in_features=n_channels * 8, out_features=n_channels * 4,  bias=True)
        self.f3 = torch.nn.Linear(in_features=n_channels * 4,  out_features=n_classes,      bias=True)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels,      eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels * 2,  eps=eps)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels * 4,  eps=eps)
        self.bn4 = torch.nn.BatchNorm2d(num_features=n_channels * 8,  eps=eps)
        self.bn5 = torch.nn.BatchNorm2d(num_features=n_channels * 16, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1
        # DropOut
        self.dropout = torch.nn.Dropout(p=dp_rate)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        
        x = x.transpose(1, 2)
        # Conv-layers
        # We replace strided convolutions with normal convolutions followed by max pooling.
        # -----
        out = self.c1(x)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn1(torch.relu(out))
        # -----
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        # -----
        out = self.c2(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn2(torch.relu(out))
        # -----
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        # -----
        out = self.c3(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn3(torch.relu(out))
        # -----
        out = self.c4(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn4(torch.relu(out))
        # -----
        # out = self.c5(out)
        # out = self.pool(out, kernel_size=2, stride=2, padding=0)
        # out = self.bn5(torch.relu(out))
        # -----
        # out = self.pool(out, kernel_size=2, stride=2, padding=0)
        # -----
        # Fully connected lyrs
        out = out.view(out.size(0), -1)

        out = self.dropout(self.f1(out))
        out = self.dropout(self.f2(out))
        out = self.f3(out)

        return out, None

class WaveletLoss(torch.nn.Module):
    def __init__(self, weight_loss=10):
        super(WaveletLoss, self).__init__()
        self.weight_loss = weight_loss

    def forward(self, model):
        loss = 0.0
        num_lyrs = 0

        # Go through modules that are instances of GConvs
        for m in model.modules():
            if not isinstance(m, eerie.nn.GConvRdG) and not(isinstance(m, eerie.nn.GConvGG)):
                continue
            if m.weights.shape[-1] == 1:
                continue
            if isinstance(m, eerie.nn.GConvRdG):
                index = -1
            elif isinstance(m, eerie.nn.GConvGG):
                index = (-2, -1)
            loss = loss + torch.mean(torch.sum(m.weights, dim=index)**2)
            num_lyrs += 1

        # Avoid division by 0
        if num_lyrs == 0:
            num_lyrs = 1

        loss = self.weight_loss * loss #/ float(num_lyrs)
        return loss

def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

if __name__ == '__main__':
    # Sanity check
    # print('OneDCNN')
    # model = OneDCNN()
    # num_params(model)
    # model(torch.rand([2, 1, 64000]))

    # Sanity check
    print('WaveletNet')
    model = WaveletNet()
    num_params(model)
    #model(torch.rand([2, 1, 50999]))
