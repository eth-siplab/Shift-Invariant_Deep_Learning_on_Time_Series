from kymatio.torch import Scattering1D
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ScatterWave(nn.Module):
    def __init__(self, args):
        super(ScatterWave, self).__init__()
        if args.len_sw == 200:
            k = 34
            self.J = 4
        elif args.len_sw == 128:
            k = 18
            self.J = 3
        elif args.len_sw == 100:
            k = 18
            self.J = 3
        elif args.len_sw == 6000:
            k = 34
            self.J = 4
        else:
            k = 18
            self.J = 3

        self.shape = args.len_sw * args.n_feature
        self.C = args.n_feature
        self.scattering = Scattering1D(J=self.J, Q=5, shape=self.shape)
        self.NN = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, args.n_class)) # for shift-invariancy


    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        Sx_all = self.scattering(x)
        Sx_all = Sx_all[:,1:,:]
        Sx_all = torch.log(torch.abs(Sx_all) + 1e-6)
        Sx_all = torch.mean(Sx_all, dim=-1) # for shift-invariancy
        x = self.NN(Sx_all) # for shift-invariancy
        # x = self.convNN(Sx_all.unsqueeze(1))
        return x, None