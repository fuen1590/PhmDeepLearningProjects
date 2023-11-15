import torch
import torch.nn as nn
import numpy as np

from train.trainable import TrainableModule
from ContrastiveModules import ContrastiveModel, pn_rul_compute


class Rnn2DCell(nn.Module):
    def __init__(self, window_size, features, hidden_dim, has_temporal=True):
        super(Rnn2DCell, self).__init__()
        self.window_size = window_size
        self.features = features
        self.hidden_dim = hidden_dim
        self.has_temporal = has_temporal
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=hidden_dim,
                               kernel_size=window_size, stride=1)
        self.out = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        if has_temporal:
            self.conv2 = nn.Conv1d(in_channels=features, out_channels=hidden_dim,
                                   kernel_size=window_size, stride=1)
            self.conv3 = nn.Conv1d(in_channels=features, out_channels=hidden_dim,
                                   kernel_size=window_size, stride=1)
            self.l4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            self.l5 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            self.l6 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x, h=None):
        b, f, l = x.shape
        if self.has_temporal:
            if h is None:
                h = torch.zeros((b, self.hidden_dim)).to(x.device)
            r = nn.functional.sigmoid(self.conv1(x).squeeze() + self.l4(h))
            z = nn.functional.sigmoid(self.conv2(x).squeeze() + self.l5(h))
            # print(z)
            n = nn.functional.tanh(self.conv3(x).squeeze() + torch.mul(r, self.l6(h)))
            ht = torch.mul((1 - z), n) + torch.mul(z, h)
        else:
            ht = nn.functional.tanh(self.conv1(x).squeeze())
        out = nn.functional.tanh(self.out(ht))
        return out, ht


class Rnn2D(nn.Module):
    def __init__(self, cell_size, features, hidden_dim, has_temporal=True):
        super(Rnn2D, self).__init__()
        self.cell_size = cell_size
        self.features = features
        self.hidden_dim = hidden_dim
        self.has_temporal = has_temporal
        self.RnnCell = Rnn2DCell(self.cell_size, self.features, self.hidden_dim, self.has_temporal)

    def forward(self, x):
        b, l, f = x.shape
        recurrent_times = l // self.cell_size
        x = x.transpose(-1, -2)  # (b, f, l)
        h = None
        last_out = None
        outs = []
        for s in range(recurrent_times):
            (last_out, h) = self.RnnCell(x[:, :, s * self.cell_size:s * self.cell_size + self.cell_size], h)
            outs.append(last_out)
        return torch.stack(outs, dim=1), last_out


class Rnn2DNet(ContrastiveModel):
    def __init__(self,
                 window_size,
                 features,
                 hidden_dim,
                 model_flag="Rnn2D", device="cuda:0", label_norm=True):
        super(Rnn2DNet, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        self.window_size = window_size
        self.features = features
        self.hidden_dim = hidden_dim
        self.l1 = Rnn2D(window_size, features, hidden_dim, has_temporal=False)
        self.l2 = Rnn2D(window_size // 2, features, hidden_dim, has_temporal=True)
        self.l3 = Rnn2D(window_size // 4, features, hidden_dim, has_temporal=True)
        self.l4 = Rnn2D(window_size // 8, features, hidden_dim, has_temporal=True)
        self.l5 = Rnn2D(window_size // 16, features, hidden_dim, has_temporal=True)

        self.fuse1_1 = nn.Sequential(
            nn.Linear(in_features=hidden_dim*2, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.fuse1_2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim*2, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.fuse1_3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim*2, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.fuse1_4 = nn.Sequential(
            nn.Linear(in_features=hidden_dim*2, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )

        self.fuse2_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.fuse2_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.fuse2_3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )

        self.fuse3_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.fuse3_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )

        self.fuse4_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1),
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
        _, f1 = self.l1(x)
        _, f2 = self.l2(x)
        _, f3 = self.l3(x)
        _, f4 = self.l4(x)
        _, f5 = self.l5(x)

        f1 = self.fuse1_1(torch.concat([f1, f2], dim=-1))
        f2 = self.fuse1_2(torch.concat([f2, f3], dim=-1))
        f3 = self.fuse1_3(torch.concat([f3, f4], dim=-1))
        f4 = self.fuse1_4(torch.concat([f4, f5], dim=-1))

        f1 = self.fuse2_1(torch.concat([f1, f2], dim=-1))
        f2 = self.fuse2_2(torch.concat([f2, f3], dim=-1))
        f3 = self.fuse2_3(torch.concat([f3, f4], dim=-1))

        f1 = self.fuse3_1(torch.concat([f1, f2], dim=-1))
        f2 = self.fuse3_2(torch.concat([f2, f3], dim=-1))

        # f1 = self.fuse4_1(torch.concat([f1, f2], dim=-1))

        return torch.concat([f1, f2], dim=-1)


if __name__ == '__main__':
    cnngru = Rnn2DNet(window_size=30, features=14, hidden_dim=256, device="cuda:0")
    a = torch.randn((32, 5, 30, 14)).to("cuda")
    b = torch.randn((32, 1)).to("cuda")
    out = cnngru(a, b)
