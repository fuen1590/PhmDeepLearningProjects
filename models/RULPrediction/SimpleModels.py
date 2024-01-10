import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ContrastiveModules import ContrastiveModel, pn_rul_compute
from train.trainable import TrainableModule

"""
Input shape: (batch, w, f0)
Feature shape: (batch, f1)
Output shape: (batch, 1)
"""


class LSTMNet(ContrastiveModel):
    def __init__(self, window_size,
                 in_features,
                 hidden_dim=256,
                 label_norm=False, model_flag="LSTM", device="cuda:0"):
        super(LSTMNet, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        if window_size > 1000:
            window_size = window_size // 32
            self.MaV = nn.AvgPool1d(kernel_size=32, stride=32)
        else:
            window_size = window_size
            self.MaV = None
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, num_layers=3,
                            batch_first=True, dropout=0.4)
        # self.lstm1 = LSTM(input_size=in_features, hidden_size=256, dropout=0.4, device=device)
        # self.lstm2 = LSTM(input_size=256, hidden_size=256, dropout=0.4, device=device)
        # self.lstm3 = LSTM(input_size=256, hidden_size=256, dropout=0.4, device=device)
        self.linear = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=1))
        self.to(device)

    def feature_extractor(self, x):
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        _, (ht, _) = self.lstm(x)
        # out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)
        # _, (ht, _) = self.lstm3(out)
        return ht[-1]
        # return ht

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.linear(x)
        else:
            f_pos, f_apos, f_neg, weights = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.linear, f_pos, f_neg), f_pos, f_apos, f_neg, weights


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.4, device="cuda:0"):
        super().__init__()
        self.wf = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.wi = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.wc = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.wo = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.hidden_size = hidden_size
        self.to(device)

    def forward(self, x):
        # x.shape=(batch, l, f)
        b, l, f = x.shape
        h = torch.zeros((b, self.hidden_size)).to(x.device)
        c = torch.zeros((b, self.hidden_size)).to(x.device)
        outputs = []
        for i in range(l):
            ft = F.sigmoid(self.wf(torch.concat([x[:, i, :], h], dim=-1)))
            it = F.sigmoid(self.wi(torch.concat([x[:, i, :], h], dim=-1)))
            c_ = F.tanh(self.wc(torch.concat([x[:, i, :], h], dim=-1)))
            c = ft * c + it * c_
            ot = F.sigmoid(self.wo(torch.concat([x[:, i, :], h], dim=-1)))
            h = ot * F.tanh(c)
            h = self.dropout(h) if self.dropout is not None else h
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class MLP(ContrastiveModel):
    def __init__(self,
                 window_size,
                 in_features,
                 filter_size=0,
                 hidden_dim=256,
                 label_norm=False,
                 model_flag="MLP", device="cuda:0"):
        super(MLP, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        if filter_size > 0:
            window_size = window_size // filter_size
            self.MaV = nn.AvgPool1d(kernel_size=filter_size, stride=filter_size)
        else:
            window_size = window_size
            self.MaV = None
        self.features_layer_1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
        )
        self.temporal_layer_1 = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(),
        )
        self.features_layer_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim//2, hidden_dim//8),
            nn.GELU(),
            nn.Dropout(),
        )
        self.temporal_layer_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim//2, hidden_dim//8),
            nn.GELU(),
            nn.Dropout(),
        )
        self.linear = nn.Sequential(nn.Dropout(),
                                    nn.Linear(in_features=(hidden_dim//8)**2, out_features=1))
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.linear(x)
        else:
            f_pos, f_apos, f_neg, weight = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.linear, f_pos, f_neg), f_pos, f_apos, f_neg, weight

    def feature_extractor(self, x):
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        # x.shape = (b, t, f)
        ff = self.features_layer_1(x)  # (b, t, f)
        tf = self.temporal_layer_1(ff.transpose(-1, -2))
        ff = self.features_layer_2(tf.transpose(-1, -2))
        tf = self.temporal_layer_2(ff.transpose(-1, -2))
        f = torch.flatten(tf, -2, -1)
        return f


class MLP_LSTM(ContrastiveModel):
    def __init__(self, in_features, window_size,
                 model_flag="MLPLstm", label_norm=True, device="cuda:0"):
        super(MLP_LSTM, self).__init__(model_flag=model_flag, label_norm=label_norm,
                                       device=device)
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=256, num_layers=3,
                            batch_first=True, dropout=0.7)
        self.features_layer_1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.temporal_layer_1 = nn.Sequential(
            nn.Linear(window_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.features_layer_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.temporal_layer_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.linear = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.7))
        self.out = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                 nn.LeakyReLU(),
                                 nn.Dropout(),
                                 nn.Linear(in_features=128, out_features=1))
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.out(x)
        else:
            f_pos, f_apos, f_neg, weight = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.out, f_pos, f_neg), f_pos, f_apos, f_neg, weight

    def feature_extractor(self, x):
        # x.shape = (b, t, f)
        ff = self.features_layer_1(x)  # (b, t, f)
        tf = self.temporal_layer_1(ff.transpose(-1, -2))
        ff = self.features_layer_2(tf.transpose(-1, -2))
        tf = self.temporal_layer_2(ff.transpose(-1, -2))
        f = torch.flatten(tf, -2, -1)
        f = self.linear(f)
        _, (ht, _) = self.lstm(x)
        return ht[-1] + f


class PositionEmbedding(nn.Module):
    def __init__(self, dim, window_size, dropout=0.5, device="cuda:0"):
        super(PositionEmbedding, self).__init__()
        # 计算pe编码
        self.pe = torch.zeros(window_size, dim)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, window_size).unsqueeze(1)  # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, dim, 2) *  # 计算公式中10000**（2i/d_model)
                             -(np.log(10000.0) / dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        self.pe = self.pe.unsqueeze(0).to(device)  # size=(1, L, d_model)，为了
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


if __name__ == '__main__':
    from dataset.utils import count_parameters
    mlp = MLP(window_size=30, in_features=14, hidden_dim=32, device="cuda:0")
    lstm = LSTMNet(30, 14, 13)
    print(count_parameters(mlp))
    print(count_parameters(lstm))
    x = torch.ones(2, 30, 14)
