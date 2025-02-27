import numpy as np
import torch
import scipy
import random
import matplotlib.pyplot as plt


def find_the_phase_of_biggest_component(fftsamples, freq):
    abs_fft = torch.abs(fftsamples)
    index = torch.argmax(abs_fft, dim=1).squeeze()
    phase_fft = torch.angle(fftsamples)
    # Get the phase of the biggest component
    angles = phase_fft[torch.arange(phase_fft.size(0)),1, 0].squeeze()
    dtheta, sign = distance_from_zero_phase(angles)
    return dtheta, sign, 1

def constant_phase_shift(sample, args, DEVICE):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    freq = torch.fft.rfftfreq(n=sample.size(1))
    dtheta, sign, index = find_the_phase_of_biggest_component(fftsamples, freq)
    
    coeff = dtheta / (2*torch.pi*freq[index])  

    if freq.dim() != 1 or coeff.dim() != 1: # if there is a single element in the batch
        coeff = coeff.unsqueeze(0)

    shifter_coeff = torch.exp(-1j*2*torch.pi*freq[None,:]*coeff[:,None])
    shifter_coeff = shifter_coeff.unsqueeze(dim=2).expand(-1, -1, sample.shape[2])
    shifted_fft = shifter_coeff*fftsamples
    #
    mixed_samples_time = torch.fft.irfft(shifted_fft, n=sample.size(1), dim=1, norm='ortho')
    return mixed_samples_time

def distance_from_zero_phase(angles):
    dtheta = angles % (2 * torch.pi)
    dtheta2 = torch.pi - torch.abs(dtheta)
    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    #
    clockwise = dtheta > 0
    sign = torch.where(clockwise, -1, 1)
    return dtheta, sign

def time_shift_one_sample(sample):
    # shift sample n times
    all_shifts = torch.zeros(50, sample.size(0), 4)
    for i in range(50):
        shift = np.random.randint(1, sample.size(0))
        all_shifts[i, :, :] = torch.roll(sample, shift, dims=0)
    return all_shifts

def random_time_shift(samples):
    # For each element in the batch shift randomly
    shifted_samples = torch.empty_like(samples)
    all_shiftes = torch.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        # Get the random shift
        shift = np.random.randint(1, samples.shape[1])
        # Shift the signal using circular
        shifted_samples[i, :, :] = torch.roll(samples[i, :, :], shift, dims=0)
        all_shiftes[i] = shift
    return shifted_samples, all_shiftes

def continous_shift_evaluate(samples, model, DEVICE ,labels):
    # For each element in the batch shift randomly
    shifted_samples = torch.zeros([samples.shape[1], samples.shape[1], samples.shape[2]])
    shifts = torch.zeros(samples.shape[0], samples.shape[1], 4)
    shifts2 = torch.zeros(samples.shape[0], samples.shape[1])
    m = torch.nn.Softmax(dim=1)
    for i in range(samples.shape[0]):
        for k in range(samples.shape[1]):
            shifted_samples[k, :, :] = torch.roll(samples[i, :, :], k, dims=0)
        out, _ = model(shifted_samples.to(DEVICE).float())
        out = out.detach()
        _, predicted = torch.max(out.data, 1)
        shifts[i,:, :] = m(out).detach().cpu()
        shifts2[i,:] = predicted.detach().cpu()
    return shifted_samples, shifts


def frame_transform(sample, fftsamples, ref_frame, args, DEVICE):
    freq = torch.fft.rfftfreq(n=sample.size(1)).to(DEVICE)
    phase_fft = torch.angle(fftsamples)
    # Get the phase of the lowest freq component after DC
    angles = phase_fft[torch.arange(phase_fft.size(0)), 1, 0].squeeze().to(DEVICE)
    # import pdb;pdb.set_trace();
    dtheta, sign = distance_from_zero_phase(angles-ref_frame)
    #
    coeff = dtheta / (2*torch.pi*freq[1])  
    if freq.dim() != 1 or coeff.dim() != 1: # if there is a single element in the batch
        coeff = coeff.unsqueeze(0)

    shifter_coeff = torch.exp(-1j*2*torch.pi*freq[None,:]*coeff[:,None])
    shifter_coeff = shifter_coeff.unsqueeze(dim=2).expand(-1, -1, sample.shape[2])
    shifted_fft = shifter_coeff*(fftsamples.to(DEVICE))

    mixed_samples_time = torch.fft.irfft(shifted_fft, n=sample.size(1), dim=1, norm='ortho')
    return mixed_samples_time