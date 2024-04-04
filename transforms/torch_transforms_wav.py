import random
import torch
import torchaudio
from torch.utils.data import Dataset

def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob

class LoadAudio(object):
    """Loads an audio into a tensor."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path']
        if path:
            waveform, sample_rate = torchaudio.load(path, normalize=True)
        else:
            # silence
            sample_rate = self.sample_rate
            waveform = torch.zeros(int(self.sample_rate), dtype=torch.float32)
        data['samples'] = waveform
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if len(samples) < length:
            samples = torch.nn.functional.pad(samples, (0, length - len(samples)))
        elif len(samples) > length:
            samples = samples[:length]
        data['samples'] = samples
        return data

class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range

    def __call__(self, data):
        if not should_apply_transform():
            return data

        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data

class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0  / (1 + scale)
        with torch.no_grad():
            data['samples'] = torch.nn.functional.interpolate(torch.arange(0, len(samples), speed_fac), torch.arange(0,len(samples)), samples).float()
        return data

class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        with torch.no_grad():
            data['samples'] = torchaudio.transforms.TimeStretch()(data['samples'].unsqueeze(0), scale).squeeze(0)
        return data

class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        max_shift = int(sample_rate * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        data['samples'] = torch.roll(samples, shifts=shift)
        return data

class AddBackgroundNoise(Dataset):
    """Adds a random background noise."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        percentage = random.uniform(0, self.max_percentage)
        data['samples'] = samples * (1 - percentage) + noise * percentage
        return data

class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 tensor."""

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=self.n_mels)
        data['mel_spectrogram'] = torchaudio.transforms.AmplitudeToDB()(mel_spec_transform(samples))
        return data

class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, tensor_name, normalize=None):
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = data[self.tensor_name]
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data
