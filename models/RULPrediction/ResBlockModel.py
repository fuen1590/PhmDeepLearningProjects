import torch
import torch.nn as nn
import numpy as np
from sklearn import manifold

import dataset.cmapss as cmapss
from models.RULPrediction.ContrastiveModules import ContrastiveModel, MSEContrastiveLoss, pn_rul_compute


class ResNet(ContrastiveModel):
    def __init__(self, in_features, window_size,
                 model_flag="ContrastiveResNet", device="cuda"):
        super(ResNet, self).__init__(model_flag=model_flag, device=device, label_norm=True)
        # if window_size > 1000:
        #     window_size = window_size // 32
        #     self.MaV = nn.AvgPool1d(kernel_size=32, stride=32)
        # else:
        #     window_size = window_size
        #     self.MaV = None
        self.tsne = None
        self.visual_samples = None
        self.embedding = []
        self.epoch_num = 0
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.res1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.res_con1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2)
        self.res2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        self.res_con2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.res3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.dense = nn.Sequential(
            nn.Linear(in_features=128 * ((window_size // 2) // 2), out_features=64),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=1)
        )
        self.to(device)

    def forward(self, x, labels=None):
        # x.shape=(batch, num, window, features)
        # labels.shape=(batch, num)
        if len(x.shape) == 4:  # if len(shape) == 4, use the contrastive computing process
            assert labels is not None
            batch, num, w, f = x.shape
            x = x.view(batch, num, f, w)
            feature_pos, feature_pos_aug, feature_neg, neg_weights = self.generate_contrastive_samples(x, labels)
            return pn_rul_compute(self.dense, feature_pos, feature_neg), feature_pos, feature_pos_aug, feature_neg, w
        else:  # if len(shape) == 3, use the regression computing process
            batch, w, f = x.shape
            x = x.view(batch, f, w)
            feature = self.feature_extractor(x)
            out = self.dense(feature)
            return out

    def feature_extractor(self, x):
        # if self.MaV:
        #     x = self.MaV(x)
        x1 = self.conv(x)
        x2 = self.res1(x1) + x1
        x3 = self.res_con1(x2) + self.res2(x2)
        x4 = self.res_con2(x3) + self.res3(x3)
        flat = self.flatten(x4)
        return flat

    def set_visual_samples(self, samples):
        """
        Sets the visualization samples used in epoch_start.

        :param samples: (batch, len, features)
        :return:
        """
        self.visual_samples = samples
        self.visual_samples = torch.transpose(self.visual_samples, -1, -2)
        self.tsne = manifold.TSNE(n_components=2, random_state=2023)
