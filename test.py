import os
import torch
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as sk_plt
import matplotlib.pyplot as plt

from config import EVENT_ENCODING_VALUES, CHANNEL_LENGTH, SLEEP_STAGE_ENCODING_VALUES
from data_load import ISRUCDataset, get_train_test_dir, ISRUCDataset2
from model.AttnSleep import Net
from sklearn.metrics import confusion_matrix, f1_score

# best_model_path = '/home/brainlab/Workspace/DH/multi_label_sleep/ckpt/AttnSleep/2022-11-22 16:59:20_fold0/best_model.pth'
best_model_path = '/home/brainlab/Workspace/DH/multi_label_sleep/ckpt/AttnSleep/2022-11-24 20:05:59_fold0/best_model.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    # get model arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='type of model to use', default='AttnSleep')
    parser.add_argument('--sampling_rate', help='sampling rate', default=200)
    parser.add_argument('--epochs', help='number of epochs', default=100)
    parser.add_argument('--batch', help='batch size', default=32)
    parser.add_argument('--lr', help='learning rate parameter', default=.00001)

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

    @staticmethod
    def get_test_loader():
        train_dir, valid_dir, test_dir = get_train_test_dir()
        # dataset = ISRUCDataset(data_dir=test_dir)
        dataset = ISRUCDataset2(mode="test")

        x = torch.tensor(dataset.total_x, dtype=torch.float32).to(device)
        y = torch.tensor(dataset.total_y, dtype=torch.int32).to(device)

        return x, y

    def accuracy(self):
        train_dir, valid_dir, test_dir = get_train_test_dir()
        apnea_result = []
        stage_result = []

        with torch.no_grad():
            apnea_accuracy = []
            stage_accuracy = []
            apnea_f1 = []
            stage_f1 = []

            for dir in test_dir:
                dataset = ISRUCDataset(data_dir=[dir])
                # dataset = ISRUCDataset2(mode="test")
                x = torch.tensor(dataset.total_x, dtype=torch.float32).to(device)
                y = torch.tensor(dataset.total_y, dtype=torch.int32).to(device)

                out_apnea, out_stage = self.model(x)
                apnea, stage = y[:, 0], y[:, 1:]

                # apnea
                label = apnea.reshape(apnea.shape[0], 1)
                output = out_apnea > 0.5
                acc = torch.mean(torch.eq(label, output.to(torch.int32)).to(dtype=torch.float32))
                f1 = f1_score(apnea.tolist(), output.reshape(output.shape[0]).to(torch.int32).cpu().tolist(), average='micro')

                apnea_accuracy.append(acc.tolist())
                apnea_f1.append(f1.tolist())

                apnea_result.append({
                    'path': dir,
                    'accuracy': acc
                })

                # stage
                label = torch.argmax(stage, dim=-1)
                output = torch.argmax(out_stage, dim=-1)

                acc = torch.mean(torch.eq(label, output).to(dtype=torch.float32))
                f1 = f1_score(label.tolist(), output.cpu().tolist(), average='micro')

                stage_accuracy.append(acc.tolist())
                stage_f1.append(f1.tolist())

                stage_result.append({
                    'path': dir,
                    'accuracy': acc
                })

            # apnea_accuracy /= len(test_dir)
            # stage_accuracy /= len(test_dir)
            # apnea_f1 /= len(test_dir)
            # stage_f1 /= len(test_dir)

        apnea_accuracy = np.array(apnea_accuracy)
        stage_accuracy = np.array(stage_accuracy)
        apnea_f1 = np.array(apnea_f1)
        stage_f1 = np.array(stage_f1)

        print('Apnea average: {}'.format(np.mean(apnea_accuracy)))
        print('Stage average: {}'.format(np.mean(stage_accuracy)))
        print('Apnea f1: {}'.format(np.mean(apnea_f1)))
        print('Stage f1: {}'.format(np.mean(stage_f1)))

        # apnea_result = pd.DataFrame(apnea_result)
        # stage_result = pd.DataFrame(stage_result)

        # apnea_result.to_csv('apnea_result.csv', index=False)
        # stage_result.to_csv('stage_result.csv', index=False)

    def get_confusion_matrix(self):
        x, y = self.get_test_loader()
        with torch.no_grad():
            out_apnea, out_stage = self.model(x)
            apnea, stage = y[:, 0], y[:, 1:]

            out_apnea = out_apnea > 0.5
            out_apnea = out_apnea.to(torch.int32)
            out_apnea = out_apnea.reshape(out_apnea.shape[0]).cpu()
            apnea = apnea.cpu()

            apnea_matrix = confusion_matrix(apnea, out_apnea)

            norm_matrix_apnea = apnea_matrix.astype('float') / apnea_matrix.sum(axis=1)[:, np.newaxis]
            sk_plt.metrics.plot_confusion_matrix(apnea, out_apnea, normalize=True)
            plt.savefig('apnea.eps', format='eps')

            out_stage = torch.argmax(out_stage, dim=-1).cpu()
            stage = torch.argmax(stage, dim=-1).cpu()

            stage_matrix = confusion_matrix(stage, out_stage)

            norm_matrix_stage = stage_matrix.astype('float') / stage_matrix.sum(axis=1)[:, np.newaxis]
            sk_plt.metrics.plot_confusion_matrix(stage, out_stage, normalize=True)
            plt.savefig('stage.eps', format='eps')
            plt.show()

            return norm_matrix_apnea, norm_matrix_stage

    def get_hypnograms(self):
        _, _, test_dir = get_train_test_dir()
        # print(test_dir)
        for test in [test_dir[-2]]:
            dataset = ISRUCDataset(data_dir=[test])
            x, y = dataset.total_x, dataset.total_y
            x_values = range(x.shape[0])
            y = y[:, 1:]
            y = np.argmax(y, axis=-1)
            plt.figure(figsize=(20, 5))
            plt.plot(x_values, y)
            plt.yticks(np.arange(0, 5))
            plt.savefig('original.eps', format='eps')

            _, out_y = self.model(torch.tensor(x, dtype=torch.float32).to(device))
            out_y = torch.argmax(out_y, dim=-1)
            out_y = out_y.cpu()
            plt.figure(figsize=(20, 5))
            plt.plot(x_values, out_y, color='green')
            plt.yticks(np.arange(0, 5))
            plt.savefig('predicted.eps', format='eps')

            plt.show()

    def best_tester(self):
        _, _, test_dir = get_train_test_dir()
        for test in test_dir:
            dataset = ISRUCDataset(data_dir=[test])
            x, y = torch.tensor(dataset.total_x, dtype=torch.float32).to(device), torch.tensor(dataset.total_y, dtype=torch.int32).to(device)

            stage = y[:, 1:]
            _, out_stage = self.model(x)

            label = torch.argmax(stage, dim=-1)
            output = torch.argmax(out_stage, dim=-1)

            acc = torch.mean(torch.eq(label, output).to(dtype=torch.float32))
            f1 = f1_score(label.tolist(), output.cpu().tolist(), average='micro')

            print(f"{test}: {acc}")
            print(f"{test}: {f1}")

    def get_stacked_graph(self):
        _, _, test_dir = get_train_test_dir()
        cmap = sns.color_palette("viridis", as_cmap=False, n_colors=5)

        for test in [test_dir[-2]]:
            dataset = ISRUCDataset(data_dir=[test])
            x, y = dataset.total_x, dataset.total_y
            x_values = range(x.shape[0])

            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = y[:, 1:]

            _, out_y = self.model(x)
            outs = []
            for i in range(out_y.shape[1]):
                out = out_y[:, i]
                outs.append(out.cpu().tolist())

            plt.figure(figsize=(20, 5))
            # fig, ax = plt.subplots()
            labels = ["Awake", "N1", "N2", "N3", "REM"]
            plt.stackplot(x_values, outs[0], outs[1], outs[2], outs[3], outs[4], labels=labels, colors=cmap)
            plt.legend(loc="lower center")
            plt.savefig('stacked.eps', format='eps')
            plt.show()



if __name__ == '__main__':
    arguments = get_args()
    tester = Tester(arguments)
    tester.accuracy()
    # tester.get_confusion_matrix()
    # tester.get_hypnograms()
    # tester.best_tester()
    # tester.get_stacked_graph()
