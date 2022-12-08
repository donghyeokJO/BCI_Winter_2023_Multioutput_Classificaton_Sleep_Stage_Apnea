import os
import torch
import argparse

import numpy as np

from config import EVENT_ENCODING_VALUES, CHANNEL_LENGTH, SLEEP_STAGE_ENCODING_VALUES
from data_load import *
from datetime import datetime
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


'''
usage: python -m runner --model [model_name] + (model_arguments)
'''


def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'

    os.system(command)


if __name__ == "__main__":
    clear_console()

    try:
        open('shuffled.pkl', 'rb')
    except:
        make_shuffled_data()

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
    parser.add_argument('--patience', type=int, default=300)

    args = vars(parser.parse_args())

    model = args.get('model')

    now = datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')

    for i in range(10):
        # dynamic importing of model
        mod = __import__(f'model.{model}', fromlist=model)
        model_cls = getattr(mod, model)(**args)

        train_dir, valid_dir, test_dir = get_train_test_dir()

        # train_dataset = ISRUCDataset(data_dir=train_dir)
        # valid_dataset = ISRUCDataset(data_dir=valid_dir)
        # test_dataset = ISRUCDataset(data_dir=test_dir)

        train_dataset = ISRUCDataset2(mode="train")
        valid_dataset = ISRUCDataset2(mode="valid")
        test_dataset = ISRUCDataset2(mode="test")

        train_dataloader = DataLoader(train_dataset, batch_size=args.get('batch'), shuffle=True)
        # train_dataloader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset, labels=train_dataset.total_y[:, 0]), batch_size=args.get('batch'))

        writer_path = f'runs/{model}/{now}_fold{i}'
        tensorboard_writer = SummaryWriter(writer_path)

        ckpt_dir = f'ckpt/{model}/{now}_fold{i}'

        try:
            os.makedirs(ckpt_dir)
        except:
            pass

        model_cls.train_(train_dataloader, valid_dataset, test_dataset, tensorboard_writer, ckpt_dir)

        break

