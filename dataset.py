import os
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset


class KWSDataset(Dataset):
    def __init__(self, dataset_dir: str, split_path: str, transform=None):
        self.samples_links = pd.read_csv(split_path)
        self.samples_links.reset_index()

        self.classes = np.sort(self.samples_links["WORD"].unique())
        self.class_name2id = {self.classes[i]: i for i in range(len(self.classes))}
        self.class_id2name = {id: name for name, id in self.class_name2id.items()}

        self.class_samples_indices = [[] for _ in range(len(self.classes))]
        self.data = []
        cur_idx = 0

        for row_id, row in self.samples_links.iterrows():
            cl = row["WORD"]
            link = row["LINK"]

            path = os.path.join(dataset_dir, link)
            target = self.class_name2id[cl]

            self.data.append((path, target))
            self.class_samples_indices[target].append(row_id)

        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_classes_number(self):
        return len(self.classes)

    def get_class_from_idx(self, idx):
        if idx in self.class_id2name.keys():
            return self.class_id2name[idx]
        return 'unknown'
    
    def get_idx_from_class(self, c):
        if c in self.class_name2id.keys():
            return self.class_name2id[c]
        return -1

    def get_class_indices(self):
        return self.class_samples_indices

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        classes_number = len(self.classes)
        classes_size = np.zeros(classes_number)

        for i in range(classes_number):
            classes_size[i] = len(self.class_samples_indices[i])

        total_size = float(sum(classes_size))
        weight_per_class = total_size / classes_size

        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight


class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data