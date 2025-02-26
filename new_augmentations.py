import numpy as np
import torch
import scipy
import random
import matplotlib.pyplot as plt



def gen_new_aug(sample, args, DEVICE):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    mixing_coeff = (0.9 - 1) * torch.rand(1) + 1  
    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * mixing_coeff + (1 - mixing_coeff) * abs_fft[index]
    z =  torch.polar(mixed_abs, phase_fft) # Go back to fft
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_2(sample, args, inds, out, DEVICE, similarities):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args) 
    coeffs = mixing_coeff.squeeze()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    mixed_phase = phase_mix(phase_fft, inds, similarities)
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_3_ablation(sample, args, DEVICE, similarities): # Apply proposed mixup but use random coeffs instead of similarity based
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def gen_new_aug_4_comparison(sample, args, DEVICE): # Apply random phase changes but keep amplitude the same
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def opposite_phase(sample, args, DEVICE, similarities): # Show the importance of phase interpolations
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    coeffs = torch.ones(sample.shape[0])
    coeffs = torch.nn.init.trunc_normal_(coeffs,1,0.1,0.9,1)

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    dtheta, sign = phase_mix_2(phase_fft, inds)
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * -sign
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time

def STAug(sample, args, DEVICE): # Comparison for Spectral and Time Augmentation
    sample = sample.detach().cpu().numpy()
    for i in range(sample.shape[0]): # For each sample in the batch
        for k in range(sample.shape[2]):  # If there is one more than one channel
            current_imf = emd.sift.sift(sample[i,:,k])
            w = np.random.uniform(0, 2, current_imf.shape[1])
            weighted_imfs = current_imf * w[None,:]
            s_prime = np.sum(weighted_imfs,axis=1)
            sample[i,:,k] = s_prime
    return torch.from_numpy(sample).float()


def vanilla_mix_up(sample):
    mixing_coeff = (0.9 - 1) * torch.rand(1) + 1  
    #m = torch.distributions.beta.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    #mixing_coeff = m.sample()   
    # Permute batch index for mixing
    index = torch.randperm(sample.size(0))
    # Mix the data
    mixed_data = mixing_coeff * sample + (1 - mixing_coeff) * sample[index]
    return mixed_data

def vanilla_mix_up_geo(sample):
    mixing_coeff = (0.7 - 1) * torch.rand(1) + 1
    # Permute batch index for mixing
    index = torch.randperm(sample.size(0))
    # Mix the data
    mixed_data = sample**mixing_coeff * sample[index]**(1 - mixing_coeff)
    return mixed_data

def vanilla_mix_up_binary(sample):
    alpha=0.8
    lam = torch.empty(sample.shape).uniform_(alpha, 1)
    mask = torch.empty(sample.shape).bernoulli_(lam)
    x_shuffle = sample[torch.randperm(sample.shape[0])]
    x_mixup = sample * mask + x_shuffle * (1 - mask)
    return x_mixup

def best_mix_up(sample, args, similarities, DEVICE): # Choose coeffs from best, but apply linear (Vanilla mixup) --- Ablation
    index = torch.randperm(sample.size(0))
    coeffs = mixing_coefficient_set_for_each(similarities, index, args) 
    coeffs = coeffs.squeeze()
    # Mix the data
    mixed_data = coeffs[:, None, None] * sample + (1 - coeffs[:, None, None]) * sample[index]
    return mixed_data

def best_mix_up_geo(sample, args, inds, out):
    mixed_samples = torch.empty(sample.shape, dtype=torch.float64)
    for idx, ind in enumerate(inds):
        mixing_coeff = (0.7 - 1) * torch.rand(1) + 1
        mixed_samples[idx,:,:] = sample[idx,:,:]**mixing_coeff *  sample[ind,:,:]**(1 - mixing_coeff)
    return mixed_samples


def mixing_coefficient_set(out):
    mixing_coefficient = torch.ones(out.shape).to(out.device)
    for idx, ind in enumerate(out):
        if ind > 0.7:
            mixing_coefficient[idx] = (0.7 - 1) * torch.rand(1) + 1
        else:
            torch.nn.init.trunc_normal_(mixing_coefficient[idx],0.85,out[idx],0.7,1)
    return mixing_coefficient

def mixing_coefficient_set_for_each(similarities, inds, args):
    threshold = 0.8
    mixing_coefficient = torch.ones(similarities.shape)
    similarities = similarities.cpu()
    distances = torch.gather(similarities,0,inds.unsqueeze(1)).cpu().numpy()
    
    mixing_coefficient = torch.ones(similarities.shape)
    distances[distances>threshold] = (0.7 - 1) * torch.rand(1) + 1
    mixing_coefficient = torch.ones(distances.shape)
    mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,args.mean, args.std, args.low_limit,args.high_limit) 
    # mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,0.9,0.2,0.7,1) --> Example
    distances[distances<=threshold] = mixing_coefficient[distances<=threshold]
    distances = torch.from_numpy(distances)
    return distances


def spec_mix(samples):
    batch_size, alpha = samples.size(0), 1
    indices = torch.randperm(batch_size)
    lam = (0.1 - 0.4) * torch.rand(1) + 0.4
    for i in range(samples.size(2)):
        current_channel = samples[:,:,i]
        current_channel_stft = torch.stft(current_channel,samples.size(1),return_complex=True)
        shuffled_data = current_channel_stft[indices, :, :]
        cut_len = int(lam * current_channel_stft.size(1))
        cut_start = np.random.randint(0, current_channel_stft.size(1) - cut_len + 1)
        current_channel_stft[:, cut_start:cut_start+cut_len, :] = shuffled_data[:, cut_start:cut_start+cut_len, :]
        samples[:,:,i] = torch.istft(current_channel_stft, n_fft=samples.size(1),length=samples.size(1))
    return samples


def cut_mix(data,alpha=2):
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]

    lam = (0.1 - 0.3) * torch.rand(1) + 0.3

    cut_len = int(lam * data.size(1))
    cut_start = np.random.randint(0, data.size(1) - cut_len + 1)

    data[:, cut_start:cut_start+cut_len] = shuffled_data[:, cut_start:cut_start+cut_len]
    return data

def phase_mix(phase_fft, inds, similarities):
    phase_difference = phase_fft - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    sign = torch.where(clockwise, -1, 1)
    coeffs = torch.squeeze(mixing_coefficient_set_for_each_phase(similarities, inds))
    mixed_phase = phase_fft
    mixed_phase = phase_fft + (1-coeffs[:, None, None]) * torch.abs(dtheta) * sign
    return mixed_phase

def phase_mix_2(phase_fft, inds):
    phase_difference = phase_fft - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    locs = torch.where(torch.abs(phase_difference) > torch.pi, -1, 1)
    sign = torch.where(clockwise, -1, 1)
    return dtheta, sign

def mixing_coefficient_set_for_each_phase(similarities, inds):
    threshold = 0.8
    mixing_coefficient = torch.ones(similarities.shape)
    similarities = similarities.cpu()
    distances = torch.gather(similarities,0,inds.unsqueeze(1)).cpu().numpy()
    
    mixing_coefficient = torch.ones(similarities.shape)
    distances[distances>threshold] = (0.9 - 1) * torch.rand(1) + 1
    mixing_coefficient = torch.ones(distances.shape)
    mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,1,0.1,0.9,1) 
    # mixing_coefficient = torch.nn.init.trunc_normal_(mixing_coefficient,0.9,0.2,0.7,1)
    distances[distances<=threshold] = mixing_coefficient[distances<=threshold]
    distances = torch.from_numpy(distances)
    return distances

def check_max_not_selected(max_indices, indices, abs_fft):
    for i in range(len(max_indices)):
        while indices[i] == max_indices[i].item():
            #np.random.shuffle(indices)
            indices = np.random.choice(np.ceil(abs_fft.size(1)/2).astype(int),abs_fft.size(2)) 
    return indices


def shift_me(x, max_len=0):
    batch_size, length, channels = x.size()
    # Generate random shifts for each signal in the batch
    shifts = torch.randint(5, length // 10 + 1, (batch_size,))
    shifted_tensors = torch.empty_like(x)
    for i in range(batch_size):
        #import pdb;pdb.set_trace();        
        # Extract the i-th signal from the batch
        signal = x[i, :, :]
        # Shift the signal using circular padding
        shifted_signal = torch.roll(signal, shifts[i].item(), dims=0)
        # Append the shifted signal to the list
        shifted_tensors[i,:,:] = shifted_signal
    return shifted_tensors, shifts
######################################

def calculate_freq_similarity(args, sample1, sample2):
    fft1 = torch.fft.rfft(sample1, dim=1, norm='ortho')
    fft2 = torch.fft.rfft(sample2, dim=1, norm='ortho')
    mag_fft1, mag_fft2 = torch.abs(fft1), torch.abs(fft2)
    #
    fft1_norm = mag_fft1[:,:,:]/torch.sum(mag_fft1[:,:,:], axis=1, keepdim=True)
    fft2_norm = mag_fft2[:,:,:]/torch.sum(mag_fft2[:,:,:], axis=1, keepdim=True)
    #
    distance = torch.eye(fft1_norm.shape[0])
    for i in range(fft1_norm.shape[0]):
        for j in range(fft1_norm.shape[0]):
            distance[i][j] = torch.nn.KLDivLoss(reduction='sum', log_target=True)(fft1_norm[i,:,:].log(), fft2_norm[j,:,:].log())
    return distance

######################################### For Supervised Learning Paradigm #########################################

def vanilla_mixup_sup(sample, target, alpha=0.3):
    size_of_batch = sample.size(0)
    # Choose quarters of the batch to mix
    indices = torch.randperm(size_of_batch)
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    mixing_coeff = m.sample()  
    # Mix the data
    mixed_data = mixing_coeff * sample + (1 - mixing_coeff) * sample[indices]
    return mixed_data, target, mixing_coeff, target[indices]


def gen_new_aug_3_ablation_sup(sample, args, DEVICE, target, alpha=0.2): 
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    coeffs = m.sample()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs + (1 - coeffs) * abs_fft[inds]

    dtheta, sign = phase_mix_2(phase_fft, inds)
    dtheta2, sign2 = phase_mix_2(phase_fft[inds], torch.linspace(0,63,64,dtype=inds.dtype))
    #mixed_phase = phase_fft if coeffs > 0.5 else phase_fft[inds]
    phase_coeff = (0.9 - 1) * torch.rand(1) + 1
    mixed_phase = phase_fft + (1-phase_coeff) * torch.abs(dtheta) * sign if coeffs > 0.5 else phase_fft[inds] + (1-phase_coeff) * torch.abs(dtheta2) * sign2
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time, target, coeffs, target[inds]

def cutmix_sup(data, target, alpha=1.):
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]

    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    lam = m.sample()

    cut_len = int(lam * data.size(1))
    cut_start = np.random.randint(0, data.size(1) - cut_len + 1)

    data[:, cut_start:cut_start+cut_len] = shuffled_data[:, cut_start:cut_start+cut_len]
    return data, target, lam, target[indices]

def binary_mixup_sup(sample, target, alpha=0.2):
    lam = torch.empty(sample.shape).uniform_(alpha, 1)
    mask = torch.empty(sample.shape).bernoulli_(lam)
    indices = torch.randperm(sample.shape[0])
    x_shuffle = sample[indices]
    x_mixup = sample * mask + x_shuffle * (1 - mask)
    return x_mixup, target, lam, target[indices]


def gen_new_aug_2_sup(sample, args, inds, out, DEVICE, similarities, target):
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    inds = torch.randperm(sample.size(0))
    mixing_coeff = mixing_coefficient_set_for_each(similarities, inds, args) 
    coeffs = mixing_coeff.squeeze()

    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
    mixed_phase = phase_mix(phase_fft, inds, similarities)
    #z =  torch.polar(mixed_abs, torch.angle(fftsamples)) # Go back to fft
    z =  torch.polar(mixed_abs, mixed_phase)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    return mixed_samples_time, target, coeffs, target[inds]


def mag_mixup_sup(sample, args, DEVICE, target, alpha=0.2): 
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    index = torch.randperm(sample.size(0))
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    coeffs = m.sample()
    abs_fft = torch.abs(fftsamples)
    phase_fft, phase_fft2 = torch.angle(fftsamples), torch.angle(fftsamples[index])
    mixed_abs = abs_fft * coeffs + (1 - coeffs) * abs_fft[index] 
    z =  torch.polar(mixed_abs, phase_fft) if coeffs > 0.5 else torch.polar(mixed_abs, phase_fft2)
    mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
    #value = torch.roll(value,5,1)
    return mixed_samples_time, target, coeffs, target[index]

#############################

def find_the_phase_of_biggest_component(fftsamples, freq):
    abs_fft = torch.abs(fftsamples)
    index = torch.argmax(abs_fft, dim=1).squeeze()
    phase_fft = torch.angle(fftsamples)
    # Get the phase of the biggest component
    angles = phase_fft[torch.arange(phase_fft.size(0)),1, 0].squeeze()
    dtheta, sign = distance_from_zero_phase(angles)
    # import pdb;pdb.set_trace();
    # if dtheta.dim() == 1:
    #     return dtheta, sign, 1
    # else:
    #     return dtheta, sign, 1
    return dtheta, sign, 1

def constant_phase_shift(sample, args, DEVICE):
    # import pdb;pdb.set_trace();
    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
    freq = torch.fft.rfftfreq(n=sample.size(1))
    dtheta, sign, index = find_the_phase_of_biggest_component(fftsamples, freq)
    #
    coeff = dtheta / (2*torch.pi*freq[index])  
    # shift_value = sample.shape[1]-torch.ceil(coeff).type(torch.int64)
    # mixed_samples_time = torch.empty_like(sample)
    # for i in range(sample.shape[0]):
    #     # Shift the signal using circular
    #     mixed_samples_time[i, :, :] = torch.roll(sample[i, :, :], shift_value[i].item(), dims=0)
    # return mixed_samples_time
    #
    if freq.dim() != 1 or coeff.dim() != 1: # if there is a single element in the batch
        coeff = coeff.unsqueeze(0)

    shifter_coeff = torch.exp(-1j*2*torch.pi*freq[None,:]*coeff[:,None])
    shifter_coeff = shifter_coeff.unsqueeze(dim=2).expand(-1, -1, sample.shape[2])
    shifted_fft = shifter_coeff*fftsamples
    #
    # dtheta, sign, index = find_the_phase_of_biggest_component(shifted_fft, freq)
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

def distance_from_zero_phase_2(angles):
    # angle_diff = 0.0300 - angles # 0.03 is the zero phase for DaLiA
    angle_diff = 0.0116 - angles # 0.0116 is the zero phase for IEEE
    # angle_diff = 0.0019 - angles # 0.0019 is the zero phase for chapman
    dtheta = angle_diff % (2 * torch.pi)
    # dtheta[dtheta > torch.pi] -= 2 * torch.pi
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
    import pdb;pdb.set_trace();
    return shifted_samples, shifts


def frame_transform(sample, fftsamples, ref_frame, args, DEVICE):
    freq = torch.fft.rfftfreq(n=sample.size(1)).to(DEVICE)
    phase_fft = torch.angle(fftsamples)
    # Get the phase of the lowest freq component after DC
    angles = phase_fft[torch.arange(phase_fft.size(0)), 1, 0].squeeze().to(DEVICE)
    # import pdb;pdb.set_trace();
    # target_phase = 0.0116 + torch.exp(-1j*2*torch.pi*freq[1]*frame_num)
    dtheta, sign = distance_from_zero_phase(angles-ref_frame)
    #
    coeff = dtheta / (2*torch.pi*freq[1])  
    if freq.dim() != 1 or coeff.dim() != 1: # if there is a single element in the batch
        coeff = coeff.unsqueeze(0)

    shifter_coeff = torch.exp(-1j*2*torch.pi*freq[None,:]*coeff[:,None])
    shifter_coeff = shifter_coeff.unsqueeze(dim=2).expand(-1, -1, sample.shape[2])
    shifted_fft = shifter_coeff*(fftsamples.to(DEVICE))
    #
    # shift_value = sample.shape[1]-torch.ceil(coeff).type(torch.int64)
    # mixed_samples_time = torch.empty_like(sample)
    # for i in range(sample.shape[0]):
    #     # Shift the signal using circular
    #     mixed_samples_time[i, :, :] = torch.roll(sample[i, :, :], shift_value[i].item(), dims=0)
    #
    #dtheta, sign, index = find_the_phase_of_biggest_component(shifted_fft)
    mixed_samples_time = torch.fft.irfft(shifted_fft, n=sample.size(1), dim=1, norm='ortho')
    return mixed_samples_time