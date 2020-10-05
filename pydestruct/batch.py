import random
from sklearn.cluster import KMeans
import numpy as np
import torch


def batch_iterator_factory(data, batch_size, n_clusters=-1, size_getter=None, shuffle=False):
    if n_clusters > 1:
        if size_getter is None:
            raise RuntimeError("You need to give a function to compute the size of an input!")
        return KMeansBatchIterator(data, batch_size, n_clusters, size_getter, shuffle=shuffle)
    elif size_getter is not None:
        return CustomSizeBatchIterator(data, batch_size, size_getter, shuffle=shuffle)
    else:
        return StandardBatchIterator(data, batch_size, shuffle=shuffle)


def batch_iterator(data, batch_size, size_getter=None):
    start = 0
    c = 0
    n_words_in_batch = 0
    while start < len(data):
        if start + c >= len(data) or (n_words_in_batch != 0 and n_words_in_batch + size_getter(data[start + c]) > batch_size):
            yield data[start:start + c]
            start += c
            c = 0
            n_words_in_batch = 0
        else:
            n_words_in_batch += size_getter(data[start + c])
            c += 1


class StandardBatchIterator:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def n_updates_per_epoch(self):
        return len(list(range(0, len(self.data), self.batch_size)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
        for start in range(0, len(self.data), self.batch_size):
            yield self.data[start : start + self.batch_size]


class CustomSizeBatchIterator:
    def __init__(self, data, batch_size, size_getter, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size_getter = size_getter

    def n_updates_per_epoch(self):
        return sum(1 for _ in self)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
        for x in batch_iterator(self.data, self.batch_size, self.size_getter):
            yield x


class KMeansBatchIterator:
    def __init__(self, data, batch_size, n_clusters, size_getter, shuffle=False):
        data_sizes = np.array([size_getter(data[i]) for i in range(len(data))])
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++").fit(data_sizes.reshape(-1, 1))

        self.clusters = {k: list() for k in range(n_clusters)}
        for i in range(len(data)):
            self.clusters[kmeans.labels_[i]].append(data[i])

        self.batch_sizes = batch_size
        self.shuffle = shuffle
        self.size_getter = size_getter

    def n_updates_per_epoch(self):
        return sum(1 for _ in self)

    def __iter__(self):
        range_fn = torch.randperm if self.shuffle else torch.arange
        for cluster_id in range_fn(len(self.clusters)).tolist():
            if self.shuffle:
                random.shuffle(self.clusters[cluster_id])
            for x in batch_iterator(self.clusters[cluster_id], self.batch_sizes, self.size_getter):
                yield x
