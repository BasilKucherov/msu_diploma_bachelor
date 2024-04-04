
import copy
import random
import torch
from torch.utils.data import Sampler
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


class TripletBatchSampler(Sampler):
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

''' Triplet Loss Online Hard Mining'''
class TripletLossBatchHard(torch.nn.Module):
    def __init__(self, margin=0.0, loss_agr_policy = "mean"):
        super(TripletLossBatchHard, self).__init__()
        self.margin = margin
        self.loss_agr_policy = loss_agr_policy

        print("[TripletLossHard] I am alive")

    def forward(self, embeddings, labels):
        device = embeddings.device

        dists = torch.cdist(embeddings, embeddings)

        same_identity_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1))
        negative_mask = ~same_identity_mask
        positive_mask = same_identity_mask ^ torch.eye(labels.size(0), dtype=torch.uint8).to(device)

        dists_negative = dists.masked_fill(negative_mask == 0, float("inf"))
            
        furthest_positive = torch.max(dists * positive_mask.to(torch.float), dim=1)[0]
        closest_negative = torch.min(dists_negative, dim=1)[0]

        diff = furthest_positive - closest_negative
        loss = torch.clamp(diff + self.margin, min=0.0)


        if self.loss_agr_policy == "mean":
            return loss.mean(), furthest_positive.mean(), closest_negative.mean()
        else:            
            return loss.max(), furthest_positive.mean(), closest_negative.mean()

''' Triplet Loss Online Kinda Random Sampling'''
def generate_triplets(labels):
    anchors = []
    positives = []
    negatives = []

    unique_labels, unique_labels_cnt = labels.unique(return_counts=True)
    anchor_labels = torch.masked_select(unique_labels, unique_labels_cnt > 1)
    
    for anchor_label in anchor_labels:
        anchor_indices = torch.where(labels == anchor_label)[0]
        
        for anchor_index in anchor_indices:
            positive_index = random.choice(anchor_indices)
            while positive_index == anchor_index:
                positive_index = random.choice(anchor_indices)
            
            negative_label = anchor_label
            while negative_label == anchor_label:
                negative_label = random.choice(unique_labels)
            
            negative_indices = torch.where(labels == negative_label)[0]
            negative_index = random.choice(negative_indices)
            
            anchors.append(anchor_index)
            positives.append(positive_index)
            negatives.append(negative_index)
    
    anchors = torch.tensor(anchors)
    positives = torch.tensor(positives)
    negatives = torch.tensor(negatives)

    return anchors, positives, negatives

class TripletLossBatchRandom(torch.nn.Module):
    def __init__(self, margin=0.0, loss_agr_policy = "mean"):
        super(TripletLossBatchRandom, self).__init__()
        self.margin = margin
        self.loss_agr_policy = loss_agr_policy

        print("[TripletLossBatchRandom] I am alive")

    def forward(self, embeddings, labels):
        device = embeddings.device

        anchor_idxs, positive_idxs, negative_idxs = generate_triplets(labels)

        anchor = embeddings[anchor_idxs, :]
        positive = embeddings[positive_idxs, :]
        negative = embeddings[negative_idxs, :]

        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, 2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, 2)
        loss = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)

        if self.loss_agr_policy == "mean":
            return loss.mean()
        else:            
            return loss.max()
