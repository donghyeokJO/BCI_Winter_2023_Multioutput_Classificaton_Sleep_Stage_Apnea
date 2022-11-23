import os
import torch
import argparse

import numpy as np
import pandas as pd
import scikitplot as sk_plt
import matplotlib.pyplot as plt

from config import EVENT_ENCODING_VALUES, CHANNEL_LENGTH, SLEEP_STAGE_ENCODING_VALUES
from data_load import ISRUCDataset, get_train_test_dir
from model.AttnSleep import Net
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

best_model_path = '/home/brainlab/Workspace/DH/multi_label_sleep/ckpt/AttnSleep/2022-11-22 16:59:20_fold0/best_model.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    # get model arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='type of model to use', default='AttnSleep')
    parser.add_argument('--sampling_rate', help='sampling rate', default=200)
    parser.add_argument('--epochs', help='number of epochs', default=100)
    parser.add_argument('--batch', help='batch size', default=32)
    parser.add_argument('--lr', help='learning rate parameter', default=.00001)
    # parser.add_argument('--lr', help='learning rate parameter', default=.00001)
    # parser.add_argument('--f1', help='first filter', default=8)
    # parser.add_argument('--f2', help='second filter', default=16)

    parser.add_argument('--f1', type=int, default=10)
    parser.add_argument('--f2', type=int, default=20)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--channel_size', type=int, default=CHANNEL_LENGTH)
    parser.add_argument('--rnn_layers', type=float, default=2)
    parser.add_argument('--cnn_dropout_rate', type=float, default=0.25)
    parser.add_argument('--rnn_dropout', type=float, default=0.5)
    parser.add_argument('--classes', type=int, default=len(EVENT_ENCODING_VALUES))
    parser.add_argument('--stage_classes', type=int, default=len(SLEEP_STAGE_ENCODING_VALUES))
    parser.add_argument('--patience', type=int, default=200)

    args = vars(parser.parse_args())

    return args


class Tester:
    def __init__(self, args):
        self.args = args
        self.ckpt = torch.load(best_model_path)
        self.model = self.load_model()
        self.model.to(device)
        self.model.eval()

    def load_model(self):
        model = Net(**self.args)
        model.load_state_dict(self.ckpt['model_state_dict'])
        return model

    def accuracy(self):
        _, _, test_dir = get_train_test_dir()
        apnea_result = []
        stage_result = []

        with torch.no_grad():
            apnea_accuracy = 0
            stage_accuracy = 0

            for dir in test_dir:
                dataset = ISRUCDataset(data_dir=[dir])
                x = torch.tensor(dataset.total_x, dtype=torch.float32).to(device)
                y = torch.tensor(dataset.total_y, dtype=torch.int32).to(device)

                out_apnea, out_stage = self.model(x)
                apnea, stage = y[:, 0], y[:, 1:]

                # apnea
                label = apnea.reshape(apnea.shape[0], 1)
                output = out_apnea > 0.5
                acc = torch.mean(torch.eq(label, output.to(torch.int32)).to(dtype=torch.float32))

                apnea_accuracy += acc

                apnea_result.append({
                    'path': dir,
                    'accuracy': acc
                })

                # stage
                label = torch.argmax(stage, dim=-1)
                output = torch.argmax(out_stage, dim=-1)
                acc = torch.mean(torch.eq(label, output).to(dtype=torch.float32))

                stage_accuracy += acc

                stage_result.append({
                    'path': dir,
                    'accuracy': acc
                })

            apnea_accuracy /= len(test_dir)
            stage_accuracy /= len(test_dir)

        print('Apnea average: {}'.format(apnea_accuracy))
        print('Stage average: {}'.format(stage_accuracy))

        apnea_result = pd.DataFrame(apnea_result)
        stage_result = pd.DataFrame(stage_result)

        apnea_result.to_csv('apnea_result.csv', index=False)
        stage_result.to_csv('stage_result.csv', index=False)

    @staticmethod
    def get_accuracy(preds, label):
        label = label.float()
        if preds.shape[1] == 1:
            label = label.reshape(label.shape[0], 1)
            output = preds > 0.5
            acc = torch.mean(torch.eq(label, output.to(torch.int32)).to(dtype=torch.float32))

        else:
            label = torch.argmax(label, dim=-1)
            output = torch.argmax(preds, dim=-1)
            acc = torch.mean(torch.eq(label, output).to(dtype=torch.float32))

        return acc



if __name__ == '__main__':
    arguments = get_args()
    tester = Tester(arguments)
    tester.accuracy()
