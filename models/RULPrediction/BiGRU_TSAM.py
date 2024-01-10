import torch
import torch.nn as nn

from ContrastiveModules import ContrastiveModel, pn_rul_compute

"""
Implementation of https://doi.org/10.1016/j.ress.2021.108297
"""


class TSAM(nn.Module):
    def __init__(self, window_size, in_features):
        super(TSAM, self).__init__()
        # self.layers = nn.ModuleList()
        # for _ in range(window_size):
        #     self.layers.append(nn.Sequential(
        #         nn.Linear(in_features, 1),
        #         nn.Sigmoid()
        #     ))
        self.layers = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        self.window_size = window_size

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape = (b, t, f)
        _, t, f = x.shape
        assert t == self.window_size
        f = []
        for i in range(t):
            # f.append(self.layers[i](x[:, i, :]))  # (b, 1)
            f.append(self.layers(x[:, i, :]))  # (b, 1)
        f = torch.concat(f, dim=-1)  # (b, t)
        f = self.softmax(f)  # (b, t)
        f = f.unsqueeze(dim=-1) * x
        return f


class BiGRU_TSAM(ContrastiveModel):
    def __init__(self, window_size, in_features, filter_size,
                 model_flag="BiGRU_TSAM", label_norm=True, device="cuda:0"):
        super(BiGRU_TSAM, self).__init__(model_flag=model_flag, label_norm=label_norm, device=device)
        if filter_size > 0:
            window_size = window_size // filter_size
            self.MaV = nn.AvgPool1d(kernel_size=filter_size, stride=filter_size)
        else:
            window_size = window_size
            self.MaV = None
        self.tsam = TSAM(window_size=window_size, in_features=in_features)
        self.gru = nn.GRU(input_size=in_features, hidden_size=256, num_layers=3,
                          bidirectional=True, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=128, out_features=1)
        )
        self.to(device)

    def feature_extractor(self, x):
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.tsam(x)
        _, x = self.gru(x)
        x = torch.concat([x[-1], x[-2]], dim=-1)
        return self.linear(x)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.output(x)
        else:
            f_pos, f_apos, f_neg, weight = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.output, f_pos, f_neg), f_pos, f_apos, f_neg, weight


if __name__ == '__main__':
    a = torch.ones((2, 30, 14)).to("cuda:0")
    l = BiGRU_TSAM(window_size=30, in_features=14)
    b = l(a)
