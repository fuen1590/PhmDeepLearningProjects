import torch
import torch.nn as nn
import numpy as np
from sklearn import manifold

import dataset.cmapss as cmapss
from models.RULPrediction.ContrastiveModules import ContrastiveModel, MSEContrastiveLoss


class ResNet(ContrastiveModel):
    def __init__(self, in_features, window_size,
                 model_flag="ContrastiveResNet", device="cuda"):
        super(ResNet, self).__init__(model_flag=model_flag, device=device)
        self.tsne = None
        self.visual_samples = None
        self.embedding = []
        self.epoch_num = 0
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(64)
        self.res1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res_con1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2)
        self.res2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_con2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.res3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
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
            out = self.dense(feature_pos)
            return out, feature_pos, feature_pos_aug, feature_neg, neg_weights
        else:  # if len(shape) == 3, use the regression computing process
            batch, w, f = x.shape
            x = x.view(batch, f, w)
            feature = self.feature_extractor(x)
            out = self.dense(feature)
            return out

    def feature_extractor(self, x):
        x1 = self.norm1(self.conv(x))
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


if __name__ == '__main__':
    window_size = 16
    threshold = 125
    neg_samples = 4
    batch_size = 256 // (neg_samples-1)
    # batch_size = 256
    subset = cmapss.Subset.FD001
    Loss = "InfoNCE"
    model_flag = "RUL-ContrastiveResNet-w{}-batch{}-thresh{}-{}-neg{}-dim64-{}-1-Theta". \
        format(window_size,
               batch_size,
               threshold,
               subset.value,
               neg_samples-1,
               Loss)
    train, test, val, scalar = cmapss.get_data(cmapss.DEFAULT_ROOT,
                                               subset,
                                               window_size=window_size,
                                               slide_step=1,
                                               sensors=cmapss.DEFAULT_SENSORS,
                                               rul_threshold=threshold,
                                               label_norm=True,
                                               val_ratio=0.2)
    net = ResNet(in_features=len(cmapss.DEFAULT_SENSORS), window_size=window_size, model_flag=model_flag,
                 device="cuda:1")
    visual_samples = torch.tensor(train.data[np.where(train.ids == 1)], dtype=torch.float32).to("cuda:1")
    net.set_visual_samples(visual_samples)
    sampler = cmapss.CmapssNegativeSampler(train, 1, neg_samples)
    net.prepare_data(train, test, val, batch_size=batch_size, num_workers=1, eval_shuffle=False)
    net.train_model(epoch=100,
                    lr=1e-3,
                    # criterion=nn.MSELoss(),
                    criterion=MSEContrastiveLoss(contrastive=Loss),
                    optimizer="adam",
                    lr_lambda=lambda x: 10 ** -(x // 15),
                    early_stop=10,
                    show_batch_loss=False)
