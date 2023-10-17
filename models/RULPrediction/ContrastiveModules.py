import torch
import torch.nn as nn

from train.trainable import TrainableModule


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
        :param neg_weight: The weight used for different negative samples with shape (batch, num_n). Weights are
                           from [0, 1], the bigger weight represents the sample will be pushed more powerful from
                           x.

        :return: A scalar of the contrastive loss.
        """
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)  # (batch, feature)
        if len(pos.shape) > 2:
            pos = torch.flatten(pos, 2)  # (batch, num_p, feature)
        if len(neg.shape) > 2:
            neg = torch.flatten(neg, 2)  # (batch, num_n, feature)
        x = x.unsqueeze(dim=1)  # (batch, 1, feature)
        # x_norm = torch.nn.functional.normalize(x, dim=-1)
        # pos_norm = torch.nn.functional.normalize(pos, dim=-1)
        # neg_norm = torch.nn.functional.normalize(neg, dim=-1)
        pos_sim = torch.cosine_similarity(x, pos, dim=2)  # positive samples similarity (batch, num_p)
        neg_sim = torch.cosine_similarity(x, neg, dim=2)  # positive samples similarity (batch, num_n)
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
            loss = self.mse(predict, label[:, 0:1]) + self.contrastive(x, pos, neg, neg_weight)
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
        batch, num, w, f = x.shape

        x_ = x.view(batch*num, w, f)
        pos = x[:, 0, :, :]
        mask = torch.zeros_like(pos).uniform_(0, 1)  # random drop features from x to get augment samples. (30%)
        mask = torch.where(mask < 0.7, 1, 0).to(self.device)
        pos_aug = mask * pos
        features = self.feature_extractor(x_)
        feature_pos_aug = self.feature_extractor(pos_aug)
        features = features.view(batch, num, -1)
        feature_pos = features[:, 0]
        feature_neg = features[:, 1:]
        neg_weights = torch.abs(labels[:, 1:] - labels[:, 0:1])
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
