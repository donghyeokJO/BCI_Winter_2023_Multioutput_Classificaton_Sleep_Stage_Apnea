# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn

from model.util import EarlyStopping
from torchmetrics.functional import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self, f1, f2, d, channel_size, sampling_rate,
                 rnn_layers, cnn_dropout_rate, rnn_dropout, classes, epochs, stage_classes, **kwargs):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.epochs = epochs

        self.rnn_units = f2 * 6
        self.cnn = EEGNet(f1=f1, f2=f2, d=d, channel_size=channel_size,
                          dropout_rate=cnn_dropout_rate, sampling_rate=sampling_rate)
        self.rnn = nn.LSTM(input_size=f2 * 6, hidden_size=f2 * 6, num_layers=rnn_layers,
                           dropout=rnn_dropout, bidirectional=True)

        # Apnea
        self.fc1 = nn.Linear(
            in_features=3600,
            out_features=classes
        )

        self.sigmoid = torch.nn.Sigmoid()

        # Sleep stage
        self.fc2 = nn.Linear(
            in_features=3600,
            out_features=stage_classes
        )

    def forward(self, x):
        b = x.size()[0]

        # EEGNet (Convolution Neural Network)
        cnn_outs = []
        for sample_x in torch.split(x, split_size_or_sections=self.sampling_rate, dim=-1):
            sample_x = sample_x.unsqueeze(dim=1)
            cnn_out = self.cnn(sample_x)
            cnn_out = cnn_out.view([b, -1])
            # print(cnn_out.shape)
            cnn_outs.append(cnn_out)
        cnn_outs = torch.stack(cnn_outs, dim=1)

        # Recurrent Neural Network
        rnn_outs, _ = self.rnn(cnn_outs)
        # Skip-Connected
        rnn_outs = rnn_outs[:, :, :self.rnn_units] + rnn_outs[:, :, self.rnn_units:] + cnn_outs
        rnn_outs = rnn_outs.view([b, -1])

        # apnea
        out1 = self.fc1(rnn_outs)
        out1 = self.sigmoid(out1)

        # sleep stage
        out2 = self.fc2(rnn_outs)
        out2 = torch.softmax(out2, dim=-1)

        return out1, out2


class ZZNet:
    def __init__(self, f1, f2, d, channel_size, sampling_rate,
                 rnn_layers, cnn_dropout_rate, rnn_dropout, classes, epochs, stage_classes, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = kwargs.get('batch')
        self.model_args = {
            'f1': f1, 'f2': f2, 'd': d, 'channel_size': channel_size, 'sampling_rate': sampling_rate,
            'rnn_layers': rnn_layers, 'cnn_dropout_rate': cnn_dropout_rate, 'rnn_dropout': rnn_dropout, 'classes': classes, 'epochs': epochs,
            'stage_classes': stage_classes
        }

        self.model = Net(f1, f2, d, channel_size, sampling_rate, rnn_layers, cnn_dropout_rate, rnn_dropout, classes, epochs, stage_classes)
        self.model.to(device=self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=kwargs.get('lr'))

        self.criterion = torch.nn.BCELoss()
        # self.criterion = torch.nn.MultiLabelSoftMarginLoss()

        self.model_state_dict = self.model.state_dict()
        self.optimizer_state_dict = self.optimizer.state_dict()

        self.early_stopping = EarlyStopping(patience=kwargs.get('patience'))

    def train_(self, train_dataloader: DataLoader, valid_dataset, test_dataset, writer: SummaryWriter, ckpt_dir: str):
        total_iter = 0
        for epoch in range(self.epochs):
            for i, data in enumerate(train_dataloader):
                self.model.train()
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                train_out_apnea, train_out_stage = self.model(train_x)
                # if we include sleep stage
                apnea, stage = train_y[:, 0], train_y[:, 1:]

                # apnea
                t_apnea_accuracy, t_apnea_loss = self.get_accuracy_loss(preds=train_out_apnea, label=apnea)
                # sleep stage
                t_stage_accuracy, t_stage_loss = self.get_accuracy_loss(preds=train_out_stage, label=stage)

                t_loss = t_apnea_loss + t_stage_loss
                self.optimizer.zero_grad()
                t_loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_x, val_y = valid_dataset.total_x, valid_dataset.total_y
                        val_x = torch.tensor(val_x, dtype=torch.float32).to(self.device)
                        val_y = torch.tensor(val_y, dtype=torch.int32).to(self.device)

                        test_x, test_y = test_dataset.total_x, test_dataset.total_y
                        test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
                        test_y = torch.tensor(test_y, dtype=torch.int32).to(self.device)

                        v_out_apnea, v_out_stage = self.model(val_x)
                        test_out_apnea, test_out_stage = self.model(test_x)

                        val_apnea, val_stage = val_y[:, 0], val_y[:, 1:]
                        test_apnea, test_stage = test_y[:, 0], test_y[:, 1:]

                        val_apnea_accuracy, val_apnea_loss = self.get_accuracy_loss(preds=v_out_apnea, label=val_apnea)
                        val_stage_accuracy, val_stage_loss = self.get_accuracy_loss(preds=v_out_stage, label=val_stage)

                        val_loss = val_apnea_loss + val_stage_loss

                        test_apnea_accuracy, test_apnea_loss = self.get_accuracy_loss(preds=test_out_apnea, label=test_apnea)
                        test_stage_accuracy, test_stage_loss = self.get_accuracy_loss(preds=test_out_stage, label=test_stage)

                        test_loss = test_apnea_loss + test_stage_loss

                        writer.add_scalar('train/loss', t_loss.item(), total_iter)
                        writer.add_scalar('train/accuracy/apnea', t_apnea_accuracy, total_iter)
                        writer.add_scalar('train/accuracy/stage', t_stage_accuracy, total_iter)

                        writer.add_scalar('validation/loss', val_loss.item(), total_iter)
                        writer.add_scalar('validation/accuracy/apnea', val_apnea_accuracy, total_iter)
                        writer.add_scalar('validation/accuracy/stage', val_stage_accuracy, total_iter)

                        writer.add_scalar('test/loss', test_loss.item(), total_iter)
                        writer.add_scalar('test/accuracy/apnea', test_apnea_accuracy, total_iter)
                        writer.add_scalar('test/accuracy/stage', test_stage_accuracy, total_iter)

                        print(
                            '[Epoch] : {0:2d}  '
                            '[Iteration] : {1:4d}  '
                            '[Train Apnea Acc] : {2:.4f}  '
                            '[Train Stage Acc] : {3:.4f}  '
                            '[Train Loss] : {4:.4f}    '
                            '[Val Apnea Acc] : {5:.4f}    '
                            '[Val Stage Acc] : {6:.4f}    '
                            '[Val Loss] : {7:.4f}    '
                            '[Test Apnea Acc] : {8:.4f}    '
                            '[Test Stage Acc] : {9:.4f}    '
                            '[Test Loss]: {10:.4f}    '.format(
                                epoch, i, t_apnea_accuracy, t_stage_accuracy, t_loss.item(),
                                val_apnea_accuracy, val_stage_accuracy, val_loss.item(), test_apnea_accuracy, test_stage_accuracy,
                                test_loss.item()
                            )
                        )

                total_iter += 1
            save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch))
            self.save_model(save_dir=save_dir, epoch=epoch)

        save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(self.epochs))
        self.save_model(save_dir=save_dir, epoch=self.epochs)

        writer.close()

    def get_accuracy_loss(self, preds, label):
        label = label.float()
        # label.requires_grad_(True)
        if preds.shape[1] == 1:
            label = label.reshape(label.shape[0], 1)
            loss = self.criterion(preds, label)
            output = preds > 0.5
            acc = torch.mean(torch.eq(label, output.to(torch.int32)).to(dtype=torch.float32))

        else:
            loss = self.criterion(preds, label)
            label = torch.argmax(label, dim=-1)
            output = torch.argmax(preds, dim=-1)
            acc = torch.mean(torch.eq(label, output).to(dtype=torch.float32))
        # acc = []
        #
        # pred = preds > 0.5
        # pred = pred.to(torch.int32)
        #
        # for i, j in zip(pred, label):
        #     acc.append(torch.equal(i, j))
        #
        # acc = torch.mean(torch.Tensor(acc))
        # acc = float(acc)

        # f1 = f1_score(preds, label.argmax(dim=-1), num_classes=self.model_args.get('classes'), average='macro')

        return acc, loss

    def save_model(self, save_dir, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            'parameter': self.model_args
        }, os.path.join(save_dir))


class EEGNet(nn.Module):
    def __init__(self, f1, f2, d, channel_size, dropout_rate, sampling_rate):
        super(EEGNet, self).__init__()
        self.cnn = nn.Sequential()
        half_sampling_rate = sampling_rate // 2

        self.cnn.add_module(
            name='conv_temporal',
            module=nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, half_sampling_rate),
                stride=1,
                bias=False,
                padding=(0, half_sampling_rate // 2)
            )
        )
        self.cnn.add_module(
            name='batch_normalization_1',
            module=nn.BatchNorm2d(f1)
        )
        self.cnn.add_module(
            name='conv_spatial',
            module=nn.Conv2d(
                in_channels=f1,
                out_channels=f1 * d,
                kernel_size=(channel_size, 1),
                stride=1,
                bias=False,
                groups=f1,
                padding=(0, 0),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_2',
            module=nn.BatchNorm2d(f1 * d)
        )
        self.cnn.add_module(
            name='activation1',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_1',
            module=nn.AvgPool2d(
                kernel_size=(1, 4)
            )
        )
        self.cnn.add_module(
            name='dropout_rate1',
            module=nn.Dropout(dropout_rate)
        )
        self.cnn.add_module(
            name='conv_separable_point',
            module=nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f2,
                kernel_size=(1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_3',
            module=nn.BatchNorm2d(f2),
        )
        self.cnn.add_module(
            name='activation2',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_2',
            module=nn.AvgPool2d(
                kernel_size=(1, 8)
            )
        )

    def forward(self, x):
        out = self.cnn(x)
        return out