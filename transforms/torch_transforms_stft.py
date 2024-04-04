import random
import torch
import torchaudio
from torch.utils.data import Dataset

from .transforms_wav import should_apply_transform

class ToSTFT(object):
    """Applies on an audio the short time fourier transform."""

    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['n_fft'] = self.n_fft
        data['hop_length'] = self.hop_length
        data['stft'] = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(samples)
        data['stft_shape'] = data['stft'].shape
        return data

class StretchAudioOnSTFT(object):
    """Stretches an audio on the frequency domain."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = torchaudio.transforms.TimeStretch()(stft.unsqueeze(0), scale).squeeze(0)
        data['stft'] = stft_stretch
        return data

class TimeshiftAudioOnSTFT(object):
    """A simple timeshift on the frequency domain without multiplying with exp."""

    def __init__(self, max_shift=8):
        self.max_shift = max_shift

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        shift = random.randint(-self.max_shift, self.max_shift)
        data['stft'] = torch.roll(stft, shifts=shift, dims=1)
        return data

class AddBackgroundNoiseOnSTFT(Dataset):
    """Adds a random background noise on the frequency domain."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        noise = random.choice(self.bg_dataset)['stft']
        percentage = random.uniform(0, self.max_percentage)
        data['stft'] = data['stft'] * (1 - percentage) + noise * percentage
        return data

class FixSTFTDimension(object):
    """Either pads or truncates in the time axis on the frequency domain, applied after stretching, time shifting etc."""

    def __call__(self, data):
        stft = data['stft']
        t_len = stft.shape[2]
        orig_t_len = data['stft_shape'][2]
        if t_len > orig_t_len:
            stft = stft[:, :, :orig_t_len]
        elif t_len < orig_t_len:
            stft = torch.nn.functional.pad(stft, (0, orig_t_len - t_len))
        data['stft'] = stft
        return data

class ToMelSpectrogramFromSTFT(object):
    """Creates the mel spectrogram from the short time fourier transform of a file. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        mel_basis = torchaudio.transforms.MelScale(n_mels=self.n_mels, sample_rate=sample_rate)(stft)
        s = torch.matmul(mel_basis, torch.abs(stft)**2.0)
        data['mel_spectrogram'] = torchaudio.transforms.AmplitudeToDB()(s)
        return data

class DeleteSTFT(object):
    """Pytorch doesn't like complex numbers, use this transform to remove STFT after computing the mel spectrogram."""

    def __call__(self, data):
        del data['stft']
        return data

class AudioFromSTFT(object):
    """Inverse short time fourier transform."""

    def __call__(self, data):
        stft = data['stft']
        data['istft_samples'] = torchaudio.functional.istft(stft, length=data['samples'].shape[-1], window=torch.hann_window(data['n_fft']))
        return data
