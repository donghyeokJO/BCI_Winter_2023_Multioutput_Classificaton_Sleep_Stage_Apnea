import os
import torch

import numpy as np
import torch.nn as nn

from torchmetrics.functional import f1_score
from model.util import TorchDataset, np_to_var
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1_score_func
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class NET(nn.Module):
    def __init__(self, sampling_rate, **kwargs):
        super(NET, self).__init__()

        # set kernel_size to be the half of sampling rate
        self.kernel_size = int(sampling_rate / 2)

        self.model = nn.Sequential()

        # filter
        self.f1 = 8
        self.f2 = 16
        # depth
        self.d = 2
        self.dropout = 0.5
        self.stride = 1
        self.bias = False
        self.channel_size = 6
        self.time_length = 6000

        self.classes = kwargs.get('classes')

        # build EEGNET model

        # 1st block
        self.model.add_module(
            name="temporal_convolutional_network",
            module=Conv2dWithConstraint(
                in_channels=1,
                out_channels=self.f1,
                kernel_size=(1, self.kernel_size),
                stride=self.stride,
                bias=self.bias,
                padding=(0, int(self.kernel_size/2))
            )
        )

        self.model.add_module(
            name="batch_normalization1",
            module=nn.BatchNorm2d(self.f1)
        )

        self.model.add_module(
            name="spatial_convolutional_network",
            module=Conv2dWithConstraint(
                in_channels=self.f1,
                out_channels=self.f1 * self.d,
                kernel_size=(self.channel_size, 1),
                stride=self.stride,
                bias=self.bias,
                padding=(0, 0),
                groups=self.f1
            )
        )

        self.model.add_module(
            name="batch_normalization2",
            module=nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True, eps=1e-3)
        )

        self.model.add_module(
            name="activation1",
            module=nn.ELU()
        )

        self.model.add_module(
            name="avg_pooling1",
            module=nn.AvgPool2d(kernel_size=(1, 4))
        )

        self.model.add_module(
            name="Dropout1",
            module=nn.Dropout(self.dropout)
        )

        # 2nd block
        self.model.add_module(
            name="separable_conv",
            module=Conv2dWithConstraint(
                in_channels=self.f1 * self.d,
                out_channels=self.f1 * self.d,
                kernel_size=(1, 16),
                stride=1,
                bias=self.bias,
                groups=self.f1 * self.d,
                padding=(0, 16//2)
            )
        )

        self.model.add_module(
            name="batch_normalization3",
            module=nn.BatchNorm2d(self.f2)
        )

        self.model.add_module(
            name="activation2",
            module=nn.ELU()
        )

        self.model.add_module(
            name='average_pool_2d_2',
            module=nn.AvgPool2d(
                kernel_size=(1, 8)
            )
        )

        self.model.add_module(
            name='dropout_rate_2',
            module=nn.Dropout(self.dropout)
        )

        out = self.model(
            np_to_var(
                np.ones(
                    (1, 1, self.channel_size, self.time_length), dtype=np.float32
                ),
            )
        )

        final_length = out.reshape(-1).shape[0]

        self.flatten = nn.Sequential()
        self.flatten.add_module(
            name="flatten",
            module=nn.Linear(
                in_features=final_length,
                # out_features=2,
                out_features=self.classes,
            )
        )

    def forward(self, x):
        b = x.size()[0]
        x = x.unsqueeze(dim=1)
        x = self.model(x)

        x = torch.reshape(x, [b, -1])
        x = self.flatten(x)
        x = torch.softmax(x, dim=-1)
        return x


class EEGNET:
    def __init__(self, sampling_rate, epochs, batch, lr, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = nn.MultiLabelSoftMarginLoss().to(self.device)
        self.epochs = epochs
        self.learning_rate = lr
        self.batch_size = batch
        self.sampling_rate = sampling_rate

        self.model = NET(
            sampling_rate=sampling_rate,
            **kwargs
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.classes = kwargs.get('classes')

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

        for m_ in m.modules():
            if isinstance(m_, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m_.weight.data, 0.0, 0.02)

    def train_(self, train_dataloader: DataLoader, valid_dataset, test_dataset, writer: SummaryWriter, ckpt_dir: str):
        # self.model.apply(self.init_weights)
        total_iter = 0

        for epoch in range(self.epochs):
            for i, data in enumerate(train_dataloader):
                self.model.train()
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                t_out = self.model(train_x)
                t_f1, t_accuracy, t_loss = self.get_accuracy_loss(preds=t_out, label=train_y)

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

                        v_out, t_out = self.model(val_x), self.model(test_x)
                        val_f1, val_accuracy, val_loss = self.get_accuracy_loss(preds=v_out, label=val_y)
                        test_f1, test_accuracy, test_loss = self.get_accuracy_loss(preds=t_out, label=test_y)

                        writer.add_scalar('train/loss', t_loss.item(), total_iter)
                        writer.add_scalar('train/f1', t_f1.item(), total_iter)
                        writer.add_scalar('train/accuracy', t_accuracy, total_iter)

                        writer.add_scalar('validation/loss', val_loss.item(), total_iter)
                        writer.add_scalar('validation/f1', val_f1.item(), total_iter)
                        writer.add_scalar('validation/accuracy', val_accuracy, total_iter)

                        writer.add_scalar('test/loss', test_loss.item(), total_iter)
                        writer.add_scalar('test/f1', test_f1.item(), total_iter)
                        writer.add_scalar('test/accuracy', test_accuracy, total_iter)

                        print(
                            '[Epoch] : {0:2d}  '
                            '[Iteration] : {1:4d}  '
                            '[Train Acc] : {2:.4f}  '
                            '[Train f1] : {3:.4f}  '
                            '[Train Loss] : {4:.4f}    '
                            '[Val Acc] : {5:.4f}    '
                            '[Val f1] : {6:.4f}    '
                            '[Val Loss] : {7:.4f}    '
                            '[Test Acc] : {8:.4f}    '
                            '[Test f1] : {9:.4f}    '
                            '[Test Loss]: {10:.4f}    '.format(
                                epoch, i, t_accuracy, t_f1.item(), t_loss.item(),
                                val_accuracy, val_f1.item(), val_loss.item(),
                                test_accuracy, test_f1.item(), test_loss.item()
                            )
                        )

                total_iter += 1
            save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch))
            self.save_model(save_dir=save_dir, epoch=epoch)

        save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(self.epochs))
        self.save_model(save_dir=save_dir, epoch=self.epochs)

    def get_accuracy_loss(self, preds, label):
        label = label.float()
        loss = self.criterion(preds, label)

        acc = []

        pred = preds > 0.5
        pred = pred.to(torch.int32)

        for i, j in zip(pred, label):
            acc.append(torch.equal(i, j))

        acc = torch.mean(torch.Tensor(acc))
        acc = float(acc)

        f1 = f1_score(preds, label.argmax(dim=-1), num_classes=self.classes, average='macro')

        return f1, acc, loss

    def save_model(self, save_dir, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            # 'parameter': self.model_args
        }, os.path.join(save_dir))

