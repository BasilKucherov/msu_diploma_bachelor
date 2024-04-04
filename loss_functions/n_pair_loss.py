
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


class NPairBatchSampler(Sampler):
    def __init__(self, class_indices, batch_size=None):
        self.class_indices = copy.deepcopy(class_indices)
        self.class_iterators = [RandomCycleIterator(c) for c in class_indices]
        self.classes_iterator = RandomCycleIterator(np.arange(len(class_indices)))
        self.n_classes = len(class_indices)
        self.batch_size = batch_size

        n_samples = 0

        for c in class_indices:
            n_samples += len(c)

        self.n_samples = n_samples

    def __iter__(self):
        for i in range(self.__len__()):
            batch = []

            if self.batch_size is None or self.batch_size > (self.n_classes * 2):
                for cl in range(self.n_classes):
                    for i in range(2):
                        batch.append(next(self.class_iterators[cl]))
            else:
                for i in range(self.batch_size // 2):
                    cl = next(self.classes_iterator)

                    for i in range(2):
                        batch.append(next(self.class_iterators[cl]))

            if len(batch) > 0:
                yield batch

    def __len__(self):
        if self.batch_size is None or self.batch_size > (self.n_classes * 2):
            return self.n_samples // (self.n_classes * 2)
        else:
            return self.n_samples // self.batch_size


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(-target * torch.nn.functional.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(-target * torch.nn.functional.log_softmax(logits, -1), -1))


class NpairLoss(torch.nn.Module):
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, labels):
        anchor_indxs = torch.arange(0, len(labels), 2)
        positive_indxs = torch.arange(1, len(labels), 2)
        labels = labels[::2]

        anchors = embeddings[anchor_indxs]
        positives = embeddings[positive_indxs]

        batch_size = anchors.size(0)
        labels = labels.view(labels.size(0), 1)

        labels = (labels == torch.transpose(labels, 0, 1)).float()
        labels = labels / torch.sum(labels, dim=1, keepdim=True).float()

        logit = torch.matmul(anchors, torch.transpose(positives, 0, 1))
        loss_ce = cross_entropy(logit, labels)
        l2_loss = torch.sum(anchors**2) / batch_size + torch.sum(positives**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss
