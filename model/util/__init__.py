import pandas as pd
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset


def butter_bandpass(lowcut=0.5, highcut=40, fs=None, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, [low, high], btype='bandpass')

    return b, a


def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=None, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


class TorchDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.total_x = x
        self.total_y = y

    def __getitem__(self, index):
        x = torch.tensor(self.total_x[index], dtype=torch.float32)
        y = torch.tensor(self.total_y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.total_y)


def np_to_var(x, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    if not hasattr(x, '__len__'):
        x = [x]

    x = np.asarray(x)

    if dtype is not None:
        x = x.astype(dtype)

    x_tensor = torch.tensor(x, requires_grad=requires_grad, **tensor_kwargs)

    if pin_memory:
        x_tensor = x_tensor.pin_memory()

    return x_tensor


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)

        if targets.shape[-1] != 1:
            at = at.reshape(targets.shape[0], targets.shape[1])
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


# class ImbalancedSampler(torch.utils.data.sampler.Sampler):
#     def __init__(self, dataset, indices=None, num_samples=None):
#         self.indices = list(range(len(dataset))) if indices is None else indices
#         self.num_samples = len(self.indices) if num_samples is None else num_samples
#
#         df = pd.DataFrame()
#         df["label"] = dataset.total_y[:, 0]


