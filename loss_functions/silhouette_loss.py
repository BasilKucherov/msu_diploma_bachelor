import torch
import random
from torch.utils.data import Sampler
import copy
import numpy as np


class RandomCycleIterator:
    def __init__(self, data_list):
        self.data_list = copy.deepcopy(data_list)
        random.shuffle(self.data_list)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data_list):
            random.shuffle(self.data_list)
            self.index = 0

        value = self.data_list[self.index]
        self.index += 1
        return value


class SilhouetteBatchSampler(Sampler):
    def __init__(self, class_indices, batch_size, n_classes_per_batch):
        self.n_classes_per_batch = n_classes_per_batch
        self.class_indices = copy.deepcopy(class_indices)
        self.class_iterators = [RandomCycleIterator(c) for c in class_indices]
        self.batch_size = batch_size
        self.n_classes = len(class_indices)
        self.n_samples_per_class = self.batch_size // self.n_classes_per_batch
        self.residue = self.batch_size % self.n_classes_per_batch

        n_samples = 0

        for c in class_indices:
            n_samples += len(c)

        self.n_samples = n_samples


    def __iter__(self):
        for i in range(self.__len__()):
            batch_classes = np.random.choice(self.n_classes, self.n_classes_per_batch, replace=False)
            residue = self.residue

            batch = []
            for cl in batch_classes:
                class_sample_number = self.n_samples_per_class + (1 if residue > 0 else 0)
                for i in range(class_sample_number):
                    batch.append(next(self.class_iterators[cl]))
                residue -= 1

            if len(batch) > 0:
                yield batch

    def __len__(self):
        return self.n_samples // self.batch_size


class SilhouetteLoss(torch.nn.Module):
    def __init__(self):
        super(SilhouetteLoss, self).__init__()

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        distances = torch.cdist(features, features)

        mean_intra_distances = []
        for i in range(len(features)):
            intra_distances = torch.masked_select(distances[i], labels == labels[i])
            mean_intra_distance = torch.mean(intra_distances[intra_distances != 0])
            mean_intra_distances.append(mean_intra_distance)
        mean_intra_distances = torch.stack(mean_intra_distances)

        # print(f"\t labels = {labels}")
        # print(f"\t mean_intra_distances = {mean_intra_distances}")

        mean_nearest_distances = []
        for i in range(len(features)):
            nearest_cluster_idx = torch.argmin(torch.where(labels != labels[i],
                                                        distances[i], torch.tensor(float('inf'))))
            nearest_distances = distances[i][labels == labels[nearest_cluster_idx]]
            mean_nearest_distance = torch.mean(nearest_distances[nearest_distances != 0])
            mean_nearest_distances.append(mean_nearest_distance)

        mean_nearest_distances = torch.stack(mean_nearest_distances)

        # print(f"\t mean_nearest_distances = {mean_nearest_distances}")


        silhouette_coeffs = (mean_nearest_distances - mean_intra_distances) / torch.max(mean_intra_distances, mean_nearest_distances)
        # print(f"\t silhouette_coeffs = {silhouette_coeffs}")

        silhouette = torch.mean(silhouette_coeffs)

        # print(f"\t silhouette = {silhouette}")

        return -silhouette
    

class SilhouetteMarginLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(SilhouetteMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        distances = torch.cdist(features, features)

        mean_intra_distances = []
        for i in range(len(features)):
            intra_distances = torch.masked_select(distances[i], labels == labels[i])
            mean_intra_distance = torch.mean(intra_distances[intra_distances != 0])
            mean_intra_distances.append(mean_intra_distance)
        mean_intra_distances = torch.stack(mean_intra_distances)

        # print(f"\t labels = {labels}")
        # print(f"\t mean_intra_distances = {mean_intra_distances}")

        mean_nearest_distances = []
        for i in range(len(features)):
            nearest_cluster_idx = torch.argmin(torch.where(labels != labels[i],
                                                        distances[i], torch.tensor(float('inf'))))
            nearest_distances = distances[i][labels == labels[nearest_cluster_idx]]
            mean_nearest_distance = torch.mean(nearest_distances[nearest_distances != 0])
            mean_nearest_distances.append(mean_nearest_distance)

        mean_nearest_distances = torch.stack(mean_nearest_distances)

        # print(f"\t mean_nearest_distances = {mean_nearest_distances}")


        diff = mean_intra_distances - mean_nearest_distances
        loss = torch.clamp(diff + self.margin, min=0.0)

        return loss.mean()