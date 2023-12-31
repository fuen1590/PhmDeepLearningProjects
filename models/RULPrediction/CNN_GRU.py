import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import models.RULPrediction.ContrastiveModules
from dataset import cmapss
from train import TrainableModule
from ContrastiveModules import ContrastiveModel


class CnnGru(ContrastiveModel):
    """
    A model proposed by https://doi.org/10.1109/TIM.2022.3227956.
    """
    def __init__(self,
                 in_features,
                 window_size,
                 model_flag="CnnGru",
                 device="cuda"):
        super(CnnGru, self).__init__(model_flag, device=device)
        self.tsne = None
        self.visual_samples = None
        self.embedding = []
        self.epoch_num = 0
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=window_size, out_channels=16, kernel_size=10, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(in_features=64 * (((in_features // 2) // 2) // 2), out_features=256)
        )
        self.grus = nn.GRU(input_size=in_features, hidden_size=128, num_layers=3, batch_first=True,
                           bidirectional=True)
        self.linears = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape)<4:
            fea = self.feature_extractor(x)
            return self.linears(fea)
        else:
            assert label is not None
            feature_pos, feature_pos_aug, feature_neg, neg_weights = self.generate_contrastive_samples(x, label)
            out = self.linears(feature_pos)
            return out, feature_pos, feature_pos_aug, feature_neg, neg_weights

    def set_visual_samples(self, samples):
        """
        Sets the visualization samples used in epoch_start.

        :param samples: (batch, len, features)
        :return:
        """
        self.visual_samples = samples
        self.tsne = manifold.TSNE(n_components=2, random_state=2023)

    def feature_extractor(self, x):
        # x.shape = (batch, length, features)
        batch, l, f = x.shape
        x_conv = x
        x_conv = self.convs(x_conv)
        _, x_grus = self.grus(x)  # (batch, length, 256)
        x_grus = torch.concat([x_grus[-1], x_grus[-2]], dim=-1)
        fea = x_conv+x_grus
        return fea

    def epoch_start(self):
        if self.visual_samples is not None:
            print("Visualizing samples processing...")
            features = self.feature_extractor(self.visual_samples)
            features = features.cpu().detach().numpy()
            embedding = self.tsne.fit_transform(features)
            self.embedding.append(embedding)
            # plt.figure(dpi=600)
            # plt.scatter(embedding[:, 0], embedding[:, 1], c=plt.cm.Spectral(range(len(embedding))))
            # plt.title("Epoch:{}".format(self.epoch_num))
            # plt.savefig(self.get_model_result_path()+"visual_embedding_{}.png".format(self.epoch_num))
            self.epoch_num += 1
        else:
            print("Visualizing samples is None, ignored.")

    def train_end(self):
        plt.figure(dpi=600)
        plt.title("Total")
        index = 0
        emd_index = [0, len(self.embedding)//2, len(self.embedding)-1]
        for i in emd_index:
            plt.scatter(self.embedding[i][:, 0], self.embedding[i][:, 1],
                        c=plt.cm.tab20(index),
                        edgecolors=plt.cm.Wistia(range(len(self.embedding[i][:, 0]))),
                        label="epoch: {}".format(i))
            index+=1
        plt.legend()
        plt.savefig(self.get_model_result_path() + "total_embedding.png")


if __name__ == '__main__':
    from dataset.cmapss import Cmapss
    window_size = 32
    batch_size = 256
    threshold = 125
    neg_samples = 3
    subset = cmapss.Subset.FD004
    model_flag = "RUL-1DCNN_GRU-w{}-batch{}-thresh{}-{}-neg{}-Contra-3". \
        format(window_size,
               batch_size,
               threshold,
               subset.value,
               neg_samples)
    train, test, val, scalar = cmapss.get_data(cmapss.DEFAULT_ROOT,
                                               subset,
                                               window_size=window_size,
                                               slide_step=1,
                                               sensors=cmapss.DEFAULT_SENSORS,
                                               rul_threshold=threshold,
                                               label_norm=True,
                                               val_ratio=0.3)
    net = CnnGru(len(cmapss.DEFAULT_SENSORS), window_size, model_flag, device="cuda:1")
    visual_samples = torch.tensor(train.data[np.where(train.ids == 1)], dtype=torch.float32, device="cuda:1")
    net.set_visual_samples(visual_samples)
    sampler = cmapss.CmapssNegativeSampler(train, neg_nums=neg_samples)
    net.prepare_data(train, test, val, batch_size=batch_size, num_workers=1)
    net.train_model(epoch=100,
                    lr=0.001,
                    criterion=models.RULPrediction.ContrastiveModules.MSEContrastiveLoss(),
                    early_stop=5,
                    lr_lambda=lambda epoch: 10 ** -(epoch // 15))
