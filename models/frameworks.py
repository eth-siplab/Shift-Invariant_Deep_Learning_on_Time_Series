import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from .backbones import *
from .TC import *
from scipy.stats import norm

class SimCLR(nn.Module):
    def __init__(self, backbone, dim=128):
        super(SimCLR, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='SimCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2,  DACL_training=False, covariance_training=False):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        elif covariance_training:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        if DACL_training:
            # Mix intermediate representations with high values
            lambda1 = ((0.9 - 1) * torch.rand(1) + 1).to(z1.device)
            lambda2 = ((0.9 - 1) * torch.rand(1) + 1).to(z1.device)
            index = torch.randperm(z1.size(0))
            # Permute batch index for mixing

            # Mix the representations
            z1 = lambda1 * z1 + (1 - lambda1) * z1[index]
            z2 = lambda2 * z2 + (1 - lambda2) * z2[index]

        if covariance_training:
            B,L,C = x1.shape
            # shifts = torch.randint(-L//2, L//2 + 1, (B,), dtype=torch.int32)

            # shifted_tensor, shifted_tensor2 = x1.clone(), x2.clone()

            # for i in range(B):
            #     shifted_tensor[i,:,:] = torch.roll(x1[i,:,:], shifts[i].item(), dims=0)
            #     shifted_tensor2[i,:,:] = torch.roll(x2[i,:,:], shifts=(shifts[i],), dims=0)
            
            # _, z1_s, sz1_s = self.encoder(shifted_tensor)
            # _, z2_s, sz2_s = self.encoder(shifted_tensor2)

            # rec_1 = covariance_training(torch.concat((z1, sz1_s), dim=1))
            # rec_2 = covariance_training(torch.concat((z2, sz2_s), dim=1))
            #
            z1_p = self.projector(z1)
            z2_p = self.projector(z2)        

            out1 = covariance_training(z1)
            out2 = covariance_training(z2)
            x1_fft = torch.abs(torch.fft.rfft(x1, dim=1))
            x2_fft = torch.abs(torch.fft.rfft(x2, dim=1))
            kl_loss = nn.KLDivLoss(reduction='sum')
            #
            out1 = (out1/torch.sum(out1, axis=1, keepdim=True)).squeeze()
            x1_fft = (x1_fft[:,1:,:]/torch.sum(x1_fft[:,1:,:], axis=1, keepdim=True)).squeeze()
            out2 = (out2/torch.sum(out2, axis=1, keepdim=True)).squeeze()
            x2_fft = (x2_fft[:,1:,:]/torch.sum(x2_fft[:,1:,:], axis=1, keepdim=True)).squeeze()
            #
            # loss1 = 1.5*kl_loss(torch.log(out1+1e-6), x1_fft)
            # loss2 = 1.5*kl_loss(torch.log(out2+1e-6), x2_fft)
            # Gaussian label
            gaussian_distributions = torch.stack([gaussian_distribution(value) for value in torch.max(x1_fft,dim=1)[1].squeeze()])
            loss1 = nn.BCEWithLogitsLoss()(out1, gaussian_distributions) # maybe -> https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
            loss2 = nn.BCEWithLogitsLoss()(out2, gaussian_distributions)
            #
            # z1 = torch.nn.functional.normalize(z1, dim=1)
            # z1_s = torch.nn.functional.normalize(z1_s, dim=1)

            # z2 = torch.nn.functional.normalize(z2, dim=1)
            # z2_s = torch.nn.functional.normalize(z2_s, dim=1)
            # #            
            # loss1 = 1*(F.mse_loss(z1_s, z1) + F.mse_loss(z2_s, z2))
            # loss2 = 1*(F.mse_loss(rec_1, shifted_tensor) + F.mse_loss(rec_2, shifted_tensor2))

            return z1_p, z2_p, loss1, loss2

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, z1, z2
        else:
            return z1, z2

def gaussian_distribution(value, length=100, sigma=3):
    x = torch.linspace(value - 3*sigma, value + 3*sigma, length).to(value.device)
    gaussian = (1 / (sigma * torch.sqrt(2 * torch.tensor(np.pi).to(value.device)))) * torch.exp(-0.5 * ((x - value) / sigma)**2)
    gaussian = gaussian / gaussian.sum()
    return gaussian

class Auditory(nn.Module):
    def __init__(self, backbone, out_channels, DEVICE):
        super(Auditory, self).__init__()

        self.encoder = backbone
        self.out_channels = out_channels
        self.reconstructer = Reconstructer(model='Auditory', layer_n=32, kernel_size=5, out_channels=out_channels, depth=1)

    def forward(self, x1, x2):
        z1, z1_rec, _, _ = self.encoder(x1)
        z2, z2_rec, _, _ = self.encoder(x2)

        # if len(x3.shape) != 3:
        #     import pdb;pdb.set_trace();
        #     z1 = z1.reshape(z1.shape[0], -1)
        #     z2 = z2.reshape(z2.shape[0], -1)
        return z1, z2, z1_rec, z2_rec


class NNCLR(nn.Module):
    def __init__(self, backbone, dim=128, pred_dim=64):
        super(NNCLR, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='NNCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.predictor = Predictor(model='NNCLR', dim=dim, pred_dim=pred_dim)

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)
        
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, p1, p2, z1.detach(), z2.detach()
        else:
            return p1, p2, z1.detach(), z2.detach()

class BYOL(nn.Module):
    def __init__(
        self,
        DEVICE,
        backbone,
        window_size = 30,
        n_channels = 77,
        hidden_layer = -1,
        projection_size = 64,
        projection_hidden_size = 256,
        moving_average = 0.99,
        use_momentum = True,
    ):
        super().__init__()

        net = backbone
        self.bb_dim = net.out_dim
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, DEVICE=DEVICE, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average)

        self.online_predictor = Predictor(model='byol', dim=projection_size, pred_dim=projection_hidden_size)

        self.to(DEVICE)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, window_size, n_channels, device=DEVICE),
                     torch.randn(2, window_size, n_channels, device=DEVICE))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x1,
        x2,
        DACL_training=False,
        return_embedding = False,
        return_projection = True,
        require_lat = False
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            online_proj_one, x1_decoded, lat1 = self.online_encoder(x1)
            online_proj_two, x2_decoded, lat2 = self.online_encoder(x2)
        else:
            online_proj_one, lat1 = self.online_encoder(x1)
            online_proj_two, lat2 = self.online_encoder(x2)

        if DACL_training:
            # Mix intermediate representations with high values
            lambda1 = ((0.9 - 1) * torch.rand(1) + 1).to(online_proj_one.device)
            lambda2 = ((0.9 - 1) * torch.rand(1) + 1).to(online_proj_one.device)
            index = torch.randperm(online_proj_one.size(0))
            # Permute batch index for mixing
            # Mix the representations
            online_proj_one = lambda1 * online_proj_one + (1 - lambda1) * online_proj_one[index]
            online_proj_two = lambda2 * online_proj_two + (1 - lambda2) * online_proj_two[index]

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
                target_proj_one, _, _ = target_encoder(x1)
                target_proj_two, _, _ = target_encoder(x2)
            else:
                target_proj_one, _ = target_encoder(x1)
                target_proj_two, _ = target_encoder(x2)
            
            if DACL_training:
                # Mix intermediate representations with high values
                lambda1 = ((0.9 - 1) * torch.rand(1) + 1).to(target_proj_one.device)
                lambda2 = ((0.9 - 1) * torch.rand(1) + 1).to(target_proj_one.device)
                # Permute batch index for mixing
                # Mix the representations
                target_proj_one = lambda1 * target_proj_one + (1 - lambda1) * target_proj_one[index]
                target_proj_two = lambda2 * target_proj_two + (1 - lambda2) * target_proj_two[index]
            
            target_proj_one.detach_()
            target_proj_two.detach_()

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            if require_lat:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()
        else:
            if require_lat:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()

class TSTCC(nn.Module):
    def __init__(self, backbone, DEVICE, temp_unit='tsfm', tc_hidden=100):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(TSTCC, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_channels
        self.TC = TC(self.bb_dim, DEVICE, tc_hidden=tc_hidden, temp_unit=temp_unit).to(DEVICE)
        self.projector = Projector(model='TS-TCC', bb_dim=self.bb_dim, prev_dim=None, dim=tc_hidden)

    def forward(self, x1, x2, DACL_training=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
            
        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        if DACL_training:
            # Mix intermediate representations with high values
            lambda1 = ((0.9 - 1) * torch.rand(1) + 1).to(z1.device)
            lambda2 = ((0.9 - 1) * torch.rand(1) + 1).to(z1.device)
            index = torch.randperm(z1.size(0))
            # Permute batch index for mixing

            # Mix the representations
            z1 = lambda1 * z1 + (1 - lambda1) * z1[index]
            z2 = lambda2 * z2 + (1 - lambda2) * z2[index]

        nce1, c_t1 = self.TC(z1, z2)
        nce2, c_t2 = self.TC(z2, z1)

        p1 = self.projector(c_t1)
        p2 = self.projector(c_t2)

        return nce1, nce2, p1, p2