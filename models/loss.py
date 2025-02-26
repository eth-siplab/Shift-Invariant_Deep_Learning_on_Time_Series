import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn
import math

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, dim_mixing=False):
        # l2 normalized features 

        zjs = torch.nn.functional.normalize(zjs, dim=1)
        zis = torch.nn.functional.normalize(zis, dim=1)
        if dim_mixing:
            m = torch.distributions.Beta(2*torch.ones(zjs.shape[0],1), 2*torch.ones(zjs.shape[0],1))
            lamb_from_beta = (m.sample() + 1).to(zjs.device) # For positive extrapolation
            zis = torch.mul(lamb_from_beta,zis) + torch.mul((1-lamb_from_beta),zjs)
            zjs = torch.mul(lamb_from_beta,zjs) + torch.mul((1-lamb_from_beta),zis)
        
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class NTXentLoss_modify(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss_modify, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, x1, x2):
        # l2 normalized features 
        zjs = torch.nn.functional.normalize(zjs, dim=1)
        zis = torch.nn.functional.normalize(zis, dim=1)
        import pdb;pdb.set_trace();
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

class AuditoryLoss(torch.nn.Module):
    def __init__(self, device, batch_size):
        super(AuditoryLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.fourier_mag_sim = self._fourier_mag_sim
        self.fourier_length = 256

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _JS_divergence(self, x, y):
        m = (0.5 * (x + y)).log()
        return 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(m, x.log()) + nn.KLDivLoss(reduction='sum', log_target=True)(m, y.log()))
    
    def _KL_divergence(self, x, y):
        return nn.KLDivLoss(reduction='sum', log_target=True)(x.log(), y.log())

    def _cosine_simililarity(self, x, y):
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        v = cos(x, y)
        return torch.sum(v)
    
    def _dot_simililarity(self, x, y):
        v = torch.tensordot(x, y, dims=2)
        return torch.sum(v)
    
    def _l1loss(self, x, y):
        loss = nn.L1Loss(reduction='sum')
        return loss(x, y)
    
    def _l2loss(self, x, y):
        loss = nn.MSELoss(reduction='sum')
        return loss(x, y)
    
    def _fourier_mag_sim(self, z1_rec, x1):
        z1_fft = torch.abs(torch.fft.rfft(z1_rec, n=self.fourier_length, norm='forward')) + torch.finfo(torch.float32).eps
        x1_fft = torch.abs(torch.fft.rfft(x1, n=self.fourier_length, norm='forward')) + torch.finfo(torch.float32).eps
        #########
        z1_fft = z1_fft[:,:,:]/torch.sum(z1_fft[:,:,:], axis=2, keepdim=True)
        x1_fft = x1_fft[:,:,:]/torch.sum(x1_fft[:,:,:], axis=2, keepdim=True)
        #########
        if torch.any(z1_fft <= 0): import pdb;pdb.set_trace();
        if torch.any(x1_fft <= 0): import pdb;pdb.set_trace();
        mse_loss = nn.MSELoss()
        # loss = kl(torch.log(z1_fft), x1_fft)
        # loss = mse_loss(z1_rec, x1)
        return self._KL_divergence(z1_fft, x1_fft) 

    def _fourier_angle_sim(self, z1_rec, x1):
        z1_fft = torch.fft.rfft(z1_rec, n=self.fourier_length, norm='forward') + torch.finfo(torch.float32).eps
        x1_fft = torch.fft.rfft(x1, n=self.fourier_length, norm='forward') + torch.finfo(torch.float32).eps
        diff_fft = (z1_fft/x1_fft) + torch.finfo(torch.float32).eps
        freq = torch.fft.rfftfreq(n=self.fourier_length)
        normal_diff = torch.exp(-1j*2*torch.pi*freq[None,:]*shifts[:,None]).cuda(3)
        #########
        loss = torch.sum(torch.abs(diff_fft - normal_diff[:,None,:]))
        return loss

    def forward(self, z1, z2, z1_rec, z2_rec, x1, x2):
        # l2 normalized features 
        # z1_norm = torch.nn.functional.normalize(z1, dim=1)
        # z2_norm = torch.nn.functional.normalize(z2, dim=1)
        l3 = self._l1loss(z1, z2)

        x1, z1_rec, z2_rec = x1.transpose(1,2), z1_rec.transpose(1,2), z2_rec.transpose(1,2)
        l1 = self.fourier_mag_sim(z1_rec, x1)
        l2 = self.fourier_mag_sim(z1_rec, z2_rec)
        # l1 = self._l2loss(z1_rec, x1)
        # l2 = self._l2loss(z1_rec, x2)
        if math.isnan(l1 + l2):
            import pdb;pdb.set_trace();
        return (l1 + l2 + l3) / self.batch_size