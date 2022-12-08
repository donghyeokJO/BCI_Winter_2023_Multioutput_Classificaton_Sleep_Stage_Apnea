# -*- coding:utf-8 -*-
import os
import torch
import math
import copy

import torch.nn as nn
import torch.nn.functional as f

from copy import deepcopy
from sklearn.metrics import f1_score
from model.util import EarlyStopping, FocalLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        N = 2
        # d_model = 150
        d_model = 150
        d_ff = 120
        h = 5
        dropout = 0.1
        num_classes = kwargs.get('classes')
        stage_classes = kwargs.get('stage_classes')
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN_ISRUC(afr_reduced_cnn_size)  # ISRUC

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc1 = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(d_model * afr_reduced_cnn_size, stage_classes)

    def forward(self, x):
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        out1 = self.fc1(encoded_features)
        out1 = self.sigmoid(out1)
        out2 = self.fc2(encoded_features)
        out2 = torch.softmax(out2, dim=-1)

        return out1, out2


class AttnSleep:
    def __init__(self, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = kwargs.get('epochs')
        self.batch_size = kwargs.get('batch_size')
        self.model = Net(**kwargs)
        self.model.to(device=self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=kwargs.get('lr'))
        # self.criterion = torch.nn.BCELoss()
        self.criterion = FocalLoss(gamma=2.0)

        self.model_state_dict = self.model.state_dict()
        self.optimizer_state_dict = self.optimizer.state_dict()

        self.model_args = kwargs

        self.early_stopping = EarlyStopping(patience=kwargs.get('patience'))

    def train_(self, train_dataloader: DataLoader, valid_dataset, test_dataset, writer: SummaryWriter, ckpt_dir: str):
        total_iter = 0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        for epoch in range(self.epochs):
            for i, data in enumerate(train_dataloader):
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                train_out_apnea, train_out_stage = self.model(train_x)
                apnea, stage = train_y[:, 0], train_y[:, 1:]

                # apnea
                t_apnea_accuracy, t_apnea_loss, t_apnea_f1 = self.get_accuracy_loss(preds=train_out_apnea, label=apnea)
                # sleep stage
                t_stage_accuracy, t_stage_loss, t_stage_f1 = self.get_accuracy_loss(preds=train_out_stage, label=stage)

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

                        val_apnea_accuracy, val_apnea_loss, val_apnea_f1 = self.get_accuracy_loss(preds=v_out_apnea, label=val_apnea)
                        val_stage_accuracy, val_stage_loss, val_stage_f1 = self.get_accuracy_loss(preds=v_out_stage, label=val_stage)

                        val_loss = val_apnea_loss + val_stage_loss

                        test_apnea_accuracy, test_apnea_loss, test_apnea_f1 = self.get_accuracy_loss(preds=test_out_apnea, label=test_apnea)
                        test_stage_accuracy, test_stage_loss, test_stage_f1 = self.get_accuracy_loss(preds=test_out_stage, label=test_stage)

                        test_loss = test_apnea_loss + test_stage_loss

                        writer.add_scalar('train/loss', t_loss.item(), total_iter)
                        writer.add_scalar('train/accuracy/apnea', t_apnea_accuracy, total_iter)
                        writer.add_scalar('train/accuracy/stage', t_stage_accuracy, total_iter)
                        writer.add_scalar('train/f1/apnea', t_apnea_f1, total_iter)
                        writer.add_scalar('train/f1/stage', t_stage_f1, total_iter)

                        writer.add_scalar('validation/loss', val_loss.item(), total_iter)
                        writer.add_scalar('validation/accuracy/apnea', val_apnea_accuracy, total_iter)
                        writer.add_scalar('validation/accuracy/stage', val_stage_accuracy, total_iter)
                        writer.add_scalar('validation/f1/apnea', val_apnea_f1, total_iter)
                        writer.add_scalar('validation/f1/stage', val_stage_f1, total_iter)

                        writer.add_scalar('test/loss', test_loss.item(), total_iter)
                        writer.add_scalar('test/accuracy/apnea', test_apnea_accuracy, total_iter)
                        writer.add_scalar('test/accuracy/stage', test_stage_accuracy, total_iter)
                        writer.add_scalar('test/f1/apnea', test_apnea_f1, total_iter)
                        writer.add_scalar('test/f1/stage', test_stage_f1, total_iter)

                        print(
                            '[Epoch] : {0:2d}  '
                            '[Iteration] : {1:4d}  '
                            '[Train Apnea Acc] : {2:.4f}  '
                            '[Train Stage Acc] : {3:.4f}  '
                            '[Train Apnea f1] : {4:.4f}  '
                            '[Train Stage f1] : {5:.4f}  '
                            '[Train Loss] : {6:.4f}    '
                            '[Val Apnea Acc] : {7:.4f}    '
                            '[Val Stage Acc] : {8:.4f}    '
                            '[Val Apnea f1] : {9:.4f}    '
                            '[Val Stage f1] : {10:.4f}    '
                            '[Val Loss] : {11:.4f}    '
                            '[Test Apnea Acc] : {12:.4f}    '
                            '[Test Stage Acc] : {13:.4f}    '
                            '[Test Apnea f1] : {14:.4f}    '
                            '[Test Stage f1] : {15:.4f}    '
                            '[Test Loss]: {16:.4f}    '.format(
                                epoch, i,
                                t_apnea_accuracy, t_stage_accuracy, t_apnea_f1, t_stage_f1, t_loss.item(),
                                val_apnea_accuracy, val_stage_accuracy, val_apnea_f1, val_stage_f1, val_loss.item(),
                                test_apnea_accuracy, test_stage_accuracy, test_apnea_f1, test_stage_f1, test_loss.item()
                            )
                        )

                        self.early_stopping(val_loss=val_loss.item())

                if self.early_stopping.early_stop:
                    save_dir = os.path.join(ckpt_dir, 'best_model.pth')
                    self.save_model(save_dir=save_dir, epoch=epoch)

                    exit()

                total_iter += 1

            save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch))
            self.save_model(save_dir=save_dir, epoch=epoch)
            scheduler.step()

        save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(self.epochs))
        self.save_model(save_dir=save_dir, epoch=self.epochs)

        writer.close()

    def get_accuracy_loss(self, preds, label):
        label = label.float()

        # apnea
        if preds.shape[1] == 1:
            label = label.reshape(label.shape[0], 1)
            loss = self.criterion(preds, label)
            output = preds > 0.5
            acc = torch.mean(torch.eq(label, output.to(torch.int32)).to(dtype=torch.float32))
            f1 = f1_score(label.reshape(label.shape[0]).tolist(), output.reshape(output.shape[0]).to(torch.int32).cpu().tolist(), average='macro')

        else:
            loss = self.criterion(preds, label)
            label = torch.argmax(label, dim=-1)
            output = torch.argmax(preds, dim=-1)
            acc = torch.mean(torch.eq(label, output).to(dtype=torch.float32))
            f1 = f1_score(label.tolist(), output.cpu().tolist(), average='macro')

        return acc, loss, f1

    def save_model(self, save_dir, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            'parameter': self.model_args
        }, os.path.join(save_dir))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x


class MRCNN_ISRUC(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN_ISRUC, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            # nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.Conv1d(3, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.features2 = nn.Sequential(
            # nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.Conv1d(3, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.fc = nn.Linear(in_features=158, out_features=150)
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.fc(x_concat)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = f.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(f.relu(self.w_1(x))))

