import torch
import torch.nn as nn
from ContrastiveModules import ContrastiveModel, pn_rul_compute
from train.trainable import TrainableModule

"""
Input shape: (batch, w, f0)
Feature shape: (batch, f1)
Output shape: (batch, 1)
"""


class LSTMNet(ContrastiveModel):
    def __init__(self, in_features,
                 label_norm=False, model_flag="LSTM", device="cuda:0"):
        super(LSTMNet, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.4)
        self.linear = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                    nn.LeakyReLU(),
                                    nn.Dropout(),
                                    nn.Linear(in_features=128, out_features=1))
        self.to(device)

    def feature_extractor(self, x):
        _, (ht, _) = self.lstm(x)
        return ht[-1]

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.linear(x)
        else:
            f_pos, f_apos, f_neg, weights = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.linear, f_pos, f_neg), f_pos, f_apos, f_neg, weights


class MLP(ContrastiveModel):
    def __init__(self,
                 window_size,
                 in_features,
                 label_norm=False,
                 model_flag="MLP", device="cuda:0"):
        super(MLP, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        self.features_layer_1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.temporal_layer_1 = nn.Sequential(
            nn.Linear(window_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.features_layer_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
        )
        self.temporal_layer_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
        )
        self.linear = nn.Sequential(nn.Linear(in_features=1024, out_features=128),
                                    nn.LeakyReLU(),
                                    nn.Dropout(),
                                    nn.Linear(in_features=128, out_features=1))
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.linear(x)
        else:
            f_pos, f_apos, f_neg, weight = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.linear, f_pos, f_neg), f_pos, f_apos, f_neg, weight

    def feature_extractor(self, x):
        ff = self.features_layer_1(x)
        tf = self.temporal_layer_1(ff.transpose(-1, -2))
        ff = self.features_layer_2(tf.transpose(-1, -2))
        tf = self.temporal_layer_2(ff.transpose(-1, -2))
        f = torch.flatten(tf, -2, -1)
        return f



