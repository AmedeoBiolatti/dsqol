import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt


def kfolds(x, n, shuffle=True, keep_out=None):
    index = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(index)
    if keep_out is None:
        keep_out = int(0)
    chunks = np.array_split(index, n + keep_out)
    in_chunks = chunks[:n]
    out_chunks = chunks[n:]
    folds = []
    for i in range(n):
        folds += [(np.concatenate([in_chunks[j] for j in range(n) if j != i]), in_chunks[i])]
    return folds


def ring_kfolds(x, n, intersection=1, shuffle=True, keep_out=None):
    index = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(index)
    if keep_out is None:
        keep_out = int(0)
    chunks = np.array_split(index, n + keep_out)
    in_chunks = chunks[:(n)]
    out_chunks = chunks[(n):]
    folds = []
    for i in range(n):
        folds += [(np.concatenate([in_chunks[j] for j in range(n) if (i - j) % n > intersection]), in_chunks[i])]
    return folds


def star_kfolds(x, n, shuffle=True, keep_out=None):
    index = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(index)
    if keep_out is None:
        keep_out = int(0)
    n_blocks = int(n * (n + 1) / 2)
    chunks = np.array_split(index, n_blocks + keep_out)
    in_chunks = chunks[:(n_blocks)]
    out_chunks = chunks[(n_blocks):]
    folds = [[[], []] for _ in range(n)]

    i = 0
    j = 1
    for k in range(n_blocks):
        if len(folds[i][1]) == 0:
            folds[i][1] = in_chunks[k]
        else:
            folds[i][1] = np.concatenate([folds[i][1], in_chunks[k]], 0)
        if len(folds[j][1]) == 0:
            folds[j][1] = in_chunks[k]
        else:
            folds[j][1] = np.concatenate([folds[j][1], in_chunks[k]], 0)
        for h in range(n):
            if h not in [i, j]:
                if len(folds[h][0]) == 0:
                    folds[h][0] = in_chunks[k]
                else:
                    folds[h][0] = np.concatenate([folds[h][0], in_chunks[k]], 0)
                pass
            pass
        j += 1
        if j == n:
            i += 1
            j = i + 1
        if i >= n - 1:
            break
        pass

    return folds


def random_splits(x, n, frac=0.2, shuffle=True, keep_out=None):
    d = x.shape[0]
    index = np.arange(d)
    if shuffle:
        np.random.shuffle(index)
    if keep_out is None:
        keep_out = int(0)
    in_ = index[:int(d * n / (n + keep_out))]
    out_ = index[int(d * n / (n + keep_out)):]
    folds = []
    for i in range(n):
        folds += [model_selection.train_test_split(in_, test_size=frac)]
    return folds


def build_folds_mask(folds):
    indexes = np.unique(np.concatenate([np.concatenate([t, v]) for t, v in folds]))
    mask = np.ones((indexes.shape[0], len(folds)))
    for i, (train_indexes, _) in enumerate(folds):
        mask[train_indexes, i] = 0
    return mask


def plot_folds(folds):
    mask = build_folds_mask(folds)
    plt.matshow(mask, aspect=len(folds) / mask.shape[0])
    pass


x = np.zeros(20)
folds = star_kfolds(x, 5, shuffle=False)
plot_folds(folds)
