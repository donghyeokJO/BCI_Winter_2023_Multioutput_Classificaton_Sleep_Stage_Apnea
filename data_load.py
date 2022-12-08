import os
import pickle
import config
import random
import torch

import numpy as np

from typing import Tuple, Sequence, Callable
from torchvision import transforms
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config import CHANNEL_LENGTH


def encode_event(events):
    new = []
    for event in events:
        # class 0: non-apnea, class 1: apnea
        # temp = np.zeros(len(config.EVENT_ENCODING_VALUES))
        temp = [0, 0]
        for e in event:
            if e.upper() in config.EVENT_ENCODING:
                # temp[config.EVENT_ENCODING[e.upper()]] = 1
                temp[1] = 1
                break
        new.append(temp)
    new = np.array(new)
    return new


def encode_event_signal(events, stage):
    new = []
    for event, st in zip(events, stage):
        # temp_event = 0
        # temp_signal = np.zeros(len(config.SLEEP_STAGE_ENCODING_VALUES))
        temp = np.zeros(len(config.SLEEP_STAGE_ENCODING_VALUES) + 1)
        for e in event:
            if e.upper() in config.EVENT_ENCODING:
                # temp_event = 1
                temp[0] = 1
                break
        # temp_signal = config.SLEEP_STAGE_ENCODING[st.upper()]
        # if st.upper()
        temp[config.SLEEP_STAGE_ENCODING[st.upper()] + 1] = 1

        new.append(temp)
    new = np.array(new)
    # new = new.reshape(new.shape[0], 1)
    return new


def get_train_test_idx():
    # train_idx = random.sample(config.ALL_SUBJECTS, 80)
    # extra = [idx for idx in config.ALL_SUBJECTS if idx not in train_idx]
    # valid_idx = random.sample(extra, 10)
    # test_idx = [idx for idx in extra if idx not in valid_idx]
    train_idx = config.ALL_SUBJECTS[:80]
    valid_idx = config.ALL_SUBJECTS[80:90]
    test_idx = config.ALL_SUBJECTS[90:]

    return sorted(train_idx), sorted(valid_idx), sorted(test_idx)


def get_train_test_dir():
    all_list = sorted(os.listdir(config.DATA_BASE_PATH))

    train_idx, valid_idx, test_idx = get_train_test_idx()

    train_dir = sorted([all_list[x] for x in train_idx])
    valid_dir = sorted([all_list[x] for x in valid_idx])
    test_dir = sorted([all_list[x] for x in test_idx])

    return train_dir, valid_dir, test_dir


def collate_fn(data):
    total_event = []
    total_signals = np.array([])

    for i, d in enumerate(data):
        if i == 0:
            total_signals = d[0]
            total_event = d[-1]
        else:
            total_signals = np.concatenate((total_signals, d[0]), axis=0)
            total_event = np.concatenate((total_event, d[-1]))

    total_event = MultiLabelBinarizer().fit_transform(np.array(total_event))

    return tuple(zip(total_signals, total_event))


class ISRUCDataset(Dataset):
    def __init__(self, data_dir):
        self.total_x, self.total_y = self.data_loader(data_dir)

    @staticmethod
    def data_loader(data_dir):
        x_, y_ = [], []
        for path in data_dir[:]:
            path = os.path.join(config.DATA_BASE_PATH, path)
            with open(path, 'rb') as f:
                temp = pickle.load(f)
                event = temp['y']['label_1']['event']
                stage = temp['y']['label_1']['stage']
                signals = temp['x']
                temp_signal = np.array([])

                signal_flag = False

                for i, eeg in enumerate(config.EEG):
                    if eeg not in signals:
                        continue

                    if not signal_flag:
                        temp_signal = signals.get(eeg)
                        signal_flag = True

                    else:
                        temp_signal = np.dstack((temp_signal, signals.get(eeg)))

                if len(temp_signal.shape) != 3:
                    print(path)
                    print('signal size 안 맞음')
                    continue

                if temp_signal.shape[2] != CHANNEL_LENGTH:
                    print(path)
                    print("channel size 안 맞음.")
                    continue

                temp_shape = temp_signal.shape
                temp_signal = temp_signal.reshape((temp_shape[0], temp_shape[2], temp_shape[1]))

                # event = encode_event(event)
                event = encode_event_signal(event, stage)
                x_.append(temp_signal)
                y_.append(event)

            f.close()
        x_ = np.concatenate(x_, axis=0)
        y_ = np.concatenate(y_, axis=0)
        return x_, y_

    def __len__(self):
        size = self.total_x.shape[0]
        return size

    def __getitem__(self, idx):
        X = torch.tensor(self.total_x[idx], dtype=torch.float32)
        label = torch.tensor(self.total_y[idx], dtype=torch.int32)
        return X, label

    def get_labels(self):
        return self.total_y


def make_shuffled_data():
    dir_1, dir_2, dir_3 = get_train_test_dir()
    total_dir = dir_1 + dir_2 + dir_3

    x_, y_ = ISRUCDataset.data_loader(total_dir)
    x_, y_ = shuffle(x_, y_, random_state=0)

    train_x, test_x, train_y, test_y = train_test_split(x_, y_, test_size=0.2, random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

    with open('shuffled.pkl', 'wb') as f:
        shuffled = {
            'train_x': train_x,
            'test_x': test_x,
            'valid_x': valid_x,
            'train_y': train_y,
            'test_y': test_y,
            'valid_y': valid_y
        }
        pickle.dump(shuffled, f)

    f.close()


def load_shuffled_data(mode="train"):
    with open('shuffled.pkl', 'rb') as f:
        shuffled = pickle.load(f)
        train_x, test_x, valid_x = shuffled['train_x'], shuffled['test_x'], shuffled['valid_x']
        train_y, test_y, valid_y = shuffled['train_y'], shuffled['test_y'], shuffled['valid_y']

    f.close()

    if mode == "train":
        x = train_x
        y = train_y

    elif mode == "test":
        x = test_x
        y = test_y

    else:
        x = valid_x
        y = valid_y

    return x,y


class ISRUCDataset2(Dataset):
    def __init__(self, mode="train"):
        self.total_x, self.total_y = load_shuffled_data(mode)

    def __len__(self):
        size = self.total_x.shape[0]
        return size

    def __getitem__(self, idx):
        X = torch.tensor(self.total_x[idx], dtype=torch.float32)
        label = torch.tensor(self.total_y[idx], dtype=torch.int32)

        return X, label


if __name__ == "__main__":
    load_shuffled_data(mode="train")
    train_dir, valid_dir, test_dir = get_train_test_dir()
    isruc = ISRUCDataset(data_dir=train_dir)
    dataloader = DataLoader(isruc, batch_size=32, shuffle=True)
    for data in dataloader:
        x, y = data
        print(y)
        exit()
