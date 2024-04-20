import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn.manifold as manifold
from train.trainable import TrainableModule
from functools import wraps


def pn_rul_compute(predictor, f_pos, f_neg):
    """
    Used to compute Rul of the positive and negative samples. Because the Weighted Info
    NCE LOSS needs all the positive and negative rul to compute the final loss.

    :param predictor: The predictor layer
    :param f_pos: The positive samples with shape (batch, features)
    :param f_neg: The negative samples with shape (batch, nums, features), where nums indicates
                  the number of negative samples.
    :return: All the rul with shape (batch, nums+1)
    """
    out_all = predictor(f_pos)
    neg_nums = f_neg.shape[1]
    neg_out = []
    for neg_i in range(neg_nums):
        neg_out.append(predictor(f_neg[:, neg_i]))
    neg_out = torch.concat(neg_out, dim=-1)
    return torch.concat([out_all, neg_out], dim=-1)


class WeightedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(WeightedInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, pos, neg, neg_weight=None):
        """
        :param x: The input of the network with shape (batch, length, feature) or (batch, feature)
        :param pos: The positive samples of x with shape (batch, num_p, length, feature) or (batch, num_p, feature),
                    where num is the number of the positive samples.
        :param neg: The negative samples of x with shape (batch, num_n, length, feature)
        :param neg_weight: The weight used for different negative samples with shape (batch, num_n).

        :return: A scalar of the contrastive loss.
        """
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)  # (batch, feature)
        if len(pos.shape) > 2:
            pos = torch.flatten(pos, 2)  # (batch, num_p, feature)
        if len(neg.shape) > 2:
            neg = torch.flatten(neg, 2)  # (batch, num_n, feature)
        x = x.unsqueeze(dim=1)  # (batch, 1, feature)
        pos_sim = torch.cosine_similarity(x, pos, dim=2)  # positive samples similarity (batch, num_p)
        neg_sim = torch.cosine_similarity(x, neg, dim=2)  # negative samples similarity (batch, num_n)
        if neg_weight is not None:
            neg_sim = torch.mul(neg_sim, neg_weight)
        nominator = torch.exp((torch.div(pos_sim, self.temperature)))  # (batch, num_p)
        denominator = torch.exp(
            torch.div(torch.concat([pos_sim, neg_sim], dim=1), self.temperature)  # (batch, num_p + num_n)
        )
        nominator = nominator.sum(dim=-1)  # (batch, )
        denominator = denominator.sum(dim=-1)  # (batch, )
        loss = -torch.log(torch.mean(nominator / denominator))
        return loss


class MSEContrastiveLoss(nn.Module):
    def __init__(self, contrastive="InfoNCE"):
        super(MSEContrastiveLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        assert contrastive in ["InfoNCE", "Triplet"]
        if contrastive == "InfoNCE":
            self.contrastive = WeightedInfoNCELoss(0.2)
        elif contrastive == "Triplet":
            self.contrastive = TripletLoss()

    def forward(self, predict, label, x=None, pos=None, neg=None, neg_weight=None):
        if x is not None and pos is not None and neg is not None:
            # print(f"MSE:{self.mse(predict, label)}")
            # print(f"Contra:{self.contrastive(x, pos, neg, neg_weight)}")
            loss = self.mse(predict, label) + self.contrastive(x, pos, neg, neg_weight)
        else:
            loss = self.mse(predict, label)
        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, x, pos, neg, neg_weight):
        """

        :param x: Anchor samples with shape (b, f)
        :param pos: Positive sample with shape (b, f)
        :param neg: Negative sample with shape (b, n_n, f)
        :param neg_weight: Alpha for every negative samples with shape (b, n_n)
        :return: A scalar value of Triplet Loss.
        """
        if neg_weight is None:
            raise RuntimeError("The neg_weight could not be None when using Triplet Loss.")
        x = torch.unsqueeze(x, dim=1)  # (b, 1, f)
        pos = torch.unsqueeze(pos, dim=1)  # (b, 1, f)
        pos_dis = torch.sum(torch.square(torch.subtract(x, pos)), 2)  # (b, 1)
        neg_dis = torch.sum(torch.square(torch.subtract(x, neg)), 2)  # (b, n)
        basic_loss = torch.add(torch.subtract(pos_dis, neg_dis), neg_weight)
        loss = torch.mean(torch.max(basic_loss, torch.zeros_like(basic_loss)))
        return loss


class ContrastiveModel(TrainableModule):
    def __init__(self, label_norm, model_flag, device):
        super(ContrastiveModel, self).__init__(model_flag=model_flag, device=device)
        self.tsne = None
        self.visual_samples = None
        self.embedding = []
        self.epoch_num = 0
        self.label_norm = label_norm

    def compute_loss(self,
                     x: torch.Tensor,
                     label: torch.Tensor,
                     criterion) -> [torch.Tensor, torch.Tensor]:
        if len(x.shape) == 4:
            rul, f_pos, f_posa, f_neg, weights = self(x, label)
            loss = criterion(rul.to(self.device), label.to(self.device), f_pos, f_posa, f_neg, weights)
        else:
            rul = self(x)
            loss = criterion(rul.to(self.device), label.to(self.device))
        return [loss, rul]

    def generate_contrastive_samples(self, x, labels):
        """
        This method is used to provide a Contrastive Loss computing arguments.

        Note
        ----
        This method is just used for the ContrastiveModule.MSEContrastiveLoss(). And you must override the
        feature_extractor() method to achieve the feature extracting process.

        :param x: x.shape = (batch, num, length, feature)
        :return: feature_pos, feature_pos_aug, feature_neg, neg_weights
        """
        assert len(x.shape) == 4
        assert labels is not None
        batch, num, w, f = x.shape

        x_ = x.view(batch * num, w, f)
        pos = x[:, 0, :, :]
        # mask = torch.zeros_like(pos).uniform_(0, 1)  # random drop features from x to get augment samples. (30%)
        # mask = torch.where(mask < 0.7, 1, 0).to(self.device)
        # pos_aug = mask * pos
        mask = torch.normal(0, 0.15, (batch, w, f), device=pos.device)  # random noise
        pos_aug = mask + pos
        all_features = self.feature_extractor(x_)
        feature_pos_aug = self.feature_extractor(pos_aug)
        features = all_features.view(batch, num, -1)
        feature_pos = features[:, 0]
        feature_neg = features[:, 1:]
        neg_weights = torch.abs(labels[:, 1:] - labels[:, 0:1]) * 2

        # assert len(x.shape) == 4
        # batch, num, w, f = x.shape
        # pos = x[:, 0, :, :]  # (batch, w, f)
        # neg = x[:, 1:, :, :]  # (batch, num_n, w, f)
        # # mask = torch.zeros_like(pos).uniform_(0, 1)  # random drop features from x to get augment samples. (30%)
        # # mask = torch.where(mask < 0.7, 1, 0).to(self.device)
        # mask = torch.randn_like(pos)  # random noise
        # pos_aug = mask + pos
        # feature_pos = self.feature_extractor(pos)
        # feature_neg = [0] * (num - 1)
        # for i in range(num - 1):
        #     feature_neg[i] = self.feature_extractor(neg[:, i, :, :])
        # feature_neg = torch.stack(feature_neg, dim=1)
        # feature_pos_aug = self.feature_extractor(pos_aug)
        # neg_weights = torch.abs(labels[:, 1:] - labels[:, 0:1]) * 2

        return feature_pos, feature_pos_aug, feature_neg, neg_weights

    def feature_extractor(self, x):
        """
        Note
        ----
        This method must be overridden to custom your own feature extracting process when you compute the contrastive
        loss by

        >>> self.generate_contrastive_samples(x, label)

        :param x: Input
        :return: tensors of feature
        """
        raise NotImplementedError("The feature_extractor method must be implemented.")

    def forward(self, x, label=None):
        """
        The forward method in contrastive models must have two parts: the one is normal
        forward process, and the other one is forward process with negative samples.

        Base Implamentation
        ----

        >>> if len(x.shape) < 4:  # the normal forward, default shape with (b, l, f)
        >>>     x = self.feature_extractor(x)
        >>>     return self.predictor(x)
        >>> else:  # the forward with negative samples, default shape with (b, num, l, f)
        >>>     f_pos, f_apos, f_neg, w = self.generate_contrastive_samples(x, label)
        >>>     return pn_rul_compute(self.predictor, f_pos, f_neg), f_pos, f_apos, f_neg, w
        :return: rul, f_pos, f_apos, f_neg, w
        """
        raise NotImplementedError("The forward method must be implemented.")

    def set_visual_samples(self, samples):
        """
        Sets the visualization samples used in epoch_start.

        :param samples: (batch, len, features)
        :return:
        """
        self.visual_samples = samples
        self.tsne = manifold.TSNE(n_components=2, random_state=2023)

    def epoch_start(self):
        if self.visual_samples is not None:
            print("Visualizing samples processing...")
            features = self.feature_extractor(self.visual_samples)
            features = features.cpu().detach().numpy().squeeze()
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
        emd_index = [0, len(self.embedding) // 2, len(self.embedding) - 1]
        for i in emd_index:
            plt.scatter(self.embedding[i][:, 0], self.embedding[i][:, 1],
                        c=plt.cm.tab20(index),
                        edgecolors=plt.cm.Wistia(range(len(self.embedding[i][:, 0]))),
                        label="epoch: {}".format(i))
            index += 1
        plt.legend()
        plt.savefig(self.get_model_result_path() + "total_embedding.png")

