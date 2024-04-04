import numpy as np
import torch
from collections import defaultdict

def clustering_metric_fc(features, labels):
    features = torch.tensor(features)
    labels = torch.tensor(labels)

    unique_labels = torch.unique(labels)
    cluster_centers = torch.stack([features[labels == l].mean(dim=0) for l in unique_labels])
    overall_center = features.mean(dim=0)
    intra_class_var = torch.sum(torch.tensor([((features[labels == l] - cluster_centers[i])**2).sum() for i, l in enumerate(unique_labels)]))
    inter_class_var = torch.sum(torch.tensor([((cluster_centers[i] - overall_center)**2).sum() * (labels == l).sum() for i, l in enumerate(unique_labels)]))
    ratio = intra_class_var / inter_class_var

    return ratio.item()


def clustering_metric_hv_one(x1, x2, y1, y2):
    num = torch.norm((x1 - y1) - (x2 - y2))
    den = torch.norm(x1 - y1) + torch.norm(x2 - y2)
    variation = num / den
    return variation

def random_four_elements(groups):
    labels = list(groups.keys())

    while True:

        label = torch.randint(high=len(labels), size=(1,))
        same_label_features = groups[labels[label]]
        if len(same_label_features) < 2:
            continue

        indices = torch.randperm(len(same_label_features))[:2]
        x1, x2 = same_label_features[indices]

        diff_label = torch.randint(high=len(labels), size=(1,))
        while diff_label == label:
            diff_label = torch.randint(high=len(labels), size=(1,))

        diff_label_features = groups[labels[diff_label]]
        if len(diff_label_features) < 2:
            continue

        indices = torch.randperm(len(diff_label_features))[:2]
        y1, y2 = diff_label_features[indices]

        yield x1, x2, y1, y2

def clustering_metric_hv(features, labels, n=100):
    features = torch.tensor(features)
    labels = torch.tensor(labels)

    groups = defaultdict(list)
    for feature, label in zip(features, labels):
        groups[label.item()].append(feature)

    unique_labels = torch.unique(labels)
    valid_groups = 0

    for l in unique_labels:
        groups[l.item()] = torch.stack(groups[l.item()])

        if groups[l.item()].shape[0] > 2:
            valid_groups += 1
    
    if valid_groups < 2:
        return 0


    variation = 0
    max_var = 0
    for i in range(n):
        x1, x2, y1, y2 = next(random_four_elements(groups))
        var = clustering_metric_hv_one(x1, x2, y1, y2)

        variation += var

        if var > max_var:
            max_var = var

    variation /= n

    return variation.item()
