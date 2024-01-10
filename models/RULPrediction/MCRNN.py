import torch
import torch.nn as nn
import numpy as np

from train.trainable import TrainableModule
from ContrastiveModules import ContrastiveModel, pn_rul_compute


class Rnn2DCell(nn.Module):
    def __init__(self, features, hidden_dim, kernel_size=3, has_temporal=True, dropout=0.5):
        super(Rnn2DCell, self).__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.has_temporal = has_temporal
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=hidden_dim,
                               kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if has_temporal:
            self.conv2 = nn.Conv1d(in_channels=features, out_channels=hidden_dim,
                                   kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.conv3 = nn.Conv1d(in_channels=features, out_channels=hidden_dim,
                                   kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim,
                                   kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.conv5 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim,
                                   kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.conv6 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim,
                                   kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else None

    def forward(self, x, h=None):
        b, f, l = x.shape
        if self.has_temporal:
            if h is None:
                h = torch.zeros((b, self.hidden_dim, l)).to(x.device)
            r = nn.functional.sigmoid(self.conv1(x) + self.conv4(h))  # shape = (b, h, l)
            z = nn.functional.sigmoid(self.conv2(x) + self.conv5(h))
            # print(z)
            n = nn.functional.tanh(self.conv3(x) + torch.mul(r, self.conv6(h)))
            ht = torch.mul((1 - z), n) + torch.mul(z, h)
        else:
            ht = nn.functional.tanh(self.conv1(x))
        ht = self.dropout(ht) if self.dropout is not None else ht
        return ht


class Rnn2D(nn.Module):
    def __init__(self, cell_size, features, hidden_dim, kernel_size=3, has_temporal=True, dropout=0.5):
        super(Rnn2D, self).__init__()
        self.cell_size = cell_size
        self.features = features
        self.hidden_dim = hidden_dim
        self.has_temporal = has_temporal
        self.RnnCell = Rnn2DCell(self.features, self.hidden_dim, kernel_size=kernel_size,
                                 has_temporal=self.has_temporal, dropout=dropout)

    def forward(self, x):
        b, l, f = x.shape
        recurrent_times = l // self.cell_size
        x = x.transpose(-1, -2)  # (b, f, l)
        h = None
        outs = []
        for s in range(recurrent_times):
            h = self.RnnCell(x[:, :, s * self.cell_size:s * self.cell_size + self.cell_size], h)
            outs.append(h)
        return torch.concat(outs, dim=-1).transpose(-1, -2), h.transpose(-1, -2)


class Rnn2DNet(ContrastiveModel):
    def __init__(self,
                 features,
                 hidden_dim,
                 window_size_split=None,
                 kernel_size=3,
                 dropout=0.5,
                 model_flag="Rnn2D", device="cuda:0", label_norm=True):
        super(Rnn2DNet, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        if window_size_split is None:
            window_size_split = [30, 15, 6, 3, 1]
        self.features = features
        self.hidden_dim = hidden_dim
        self.layers1 = nn.ModuleList()
        self.layers1.append(Rnn2D(cell_size=window_size_split[0],
                                  features=features,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  dropout=dropout))
        for w in range(1, len(window_size_split)):
            self.layers1.append(Rnn2D(cell_size=window_size_split[w],
                                      features=hidden_dim,
                                      hidden_dim=hidden_dim,
                                      kernel_size=kernel_size,
                                      dropout=dropout))
        # self.layers2 = nn.ModuleList()
        # for w in window_size_split:
        #     self.layers2.append(Rnn2D(cell_size=w,
        #                               features=hidden_dim,
        #                               hidden_dim=hidden_dim,
        #                               kernel_size=3))
        # self.layers3 = nn.ModuleList()
        # for w in window_size_split:
        #     self.layers3.append(Rnn2D(cell_size=w,
        #                               features=hidden_dim,
        #                               hidden_dim=hidden_dim,
        #                               kernel_size=3))
        # self.fuse = nn.Sequential(
        #     nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
        #            batch_first=True)
        # )

        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim // 2, out_features=1),
        )
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.out(x)
        else:
            f_pos, f_apos, f_neg, w = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.out, f_pos, f_neg), f_pos, f_apos, f_neg, w

    def feature_extractor(self, x):
        # x.shape = (b, l, f)
        f = x
        for l in range(0, len(self.layers1)):
            _, f = self.layers1[l](f)
            # print(f.shape)
            # print(_.shape)

        # fs = []
        # for l in self.layers2:
        #     fs.append(l(f)+f)
        # f = None
        # for features in fs:
        #     f = f + features if f is not None else features
        # f = f/(2*len(self.layers2))

        # fs = []
        # for l in self.layers3:
        #     fs.append(l(f)+f)
        # f = None
        # for features in fs:
        #     f = f + features if f is not None else features
        # f = f/(2*len(self.layers2))

        # _, h = self.fuse(f)
        #
        # f1 = self.fuse1_1(torch.concat([f1, f2], dim=-1))
        # f2 = self.fuse1_2(torch.concat([f2, f3], dim=-1))
        # f3 = self.fuse1_3(torch.concat([f3, f4], dim=-1))
        # f4 = self.fuse1_4(torch.concat([f4, f5], dim=-1))
        #
        # f1 = self.fuse2_1(torch.concat([f1, f2], dim=-1))
        # f2 = self.fuse2_2(torch.concat([f2, f3], dim=-1))
        # f3 = self.fuse2_3(torch.concat([f3, f4], dim=-1))
        #
        # f1 = self.fuse3_1(torch.concat([f1, f2], dim=-1))
        # f2 = self.fuse3_2(torch.concat([f2, f3], dim=-1))

        # f1 = self.fuse4_1(torch.concat([f1, f2], dim=-1))

        return f.squeeze()


#
# class MLP_Mixer(nn.Module):
#     def __init__(self, window_size, in_features, hidden_dim):
#         super(MLP_Mixer, self).__init__()
#         self.features_layer_1 = nn.Sequential(
#             nn.Linear(in_features, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(0.7),
#         )
#         self.temporal_layer_1 = nn.Sequential(
#             nn.Linear(window_size, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(0.7),
#         )
#
#     def forward(self, x):
#         # x.shape = (b, t, f)
#         tf = self.features_layer_1(x)
#         ff = self.features_layer_1(tf.transpose(-1, -2))
#         return ff
#
#
# class LSTM_Mixer_Cell(nn.Module):
#     def __init__(self, window, features, hidden_dim):
#         super(LSTM_Mixer_Cell, self).__init__()
#         self.window = window
#         self.features = features
#         self.hidden_dim = hidden_dim
#         self.wf = MLP_Mixer(window_size=window, in_features=features, hidden_dim=hidden_dim)
#         self.wi = MLP_Mixer(window_size=window, in_features=features, hidden_dim=hidden_dim)
#         self.wc = MLP_Mixer(window_size=window, in_features=features, hidden_dim=hidden_dim)
#         self.wo = MLP_Mixer(window_size=window, in_features=features, hidden_dim=hidden_dim)
#
#     def forward(self, x, h=None):
#         # x.shape = (b, t, f), t=window
#         b, t, f = x.shape
#         if h is None:
#             h = torch.zeros((b, self.hidden_dim, self.hidden_dim)).to(x.device)
#         ft = nn.functional.sigmoid(self.wf(x))
#         it = nn.functional.sigmoid(self.wi(x))
#         ct = nn.functional.tanh(self.wc(x))


if __name__ == '__main__':
    cnngru = Rnn2DNet(features=14, hidden_dim=256, window_size_split=[30, 15, 6, 3, 1])
    # cnngru = Rnn2DCell(features=14, hidden_dim=256, kernel_size=3)
    a = torch.randn((32, 30, 14)).to("cuda:0")
    out = cnngru(a)
