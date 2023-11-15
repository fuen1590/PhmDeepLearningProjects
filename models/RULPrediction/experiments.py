import dataset.cmapss as cmapss
import models.RULPrediction as rul
from dataset.utils import compute_metrics

import torch
import numpy as np
from SimpleModels import *
from OrMLP import OrMLPNet
from MCRNN import Rnn2DNet


def train_cmapss(model: rul.ContrastiveModel,
                 window_size,
                 neg_samples,
                 batch_size,
                 subset: cmapss.Subset,
                 exp_time,
                 contra=True,
                 label_norm=True):
    threshold = 125
    batch_size = batch_size // neg_samples if contra else batch_size
    # batch_size = batch_size
    # batch_size = 256
    Loss = "InfoNCE" if contra else ""
    model_flag = "RUL-{}-norm{}-w{}-batch{}-thresh{}-{}-neg{}-{}{}". \
        format(model.flag,
               1 if label_norm else 0,
               window_size,
               batch_size,
               threshold,
               subset.value,
               neg_samples - 1 if contra else 0,
               Loss,
               exp_time)
    train, test, val, scalar = cmapss.get_data(cmapss.DEFAULT_ROOT,
                                               subset,
                                               window_size=window_size,
                                               slide_step=1,
                                               sensors=cmapss.DEFAULT_SENSORS,
                                               rul_threshold=threshold,
                                               label_norm=label_norm,
                                               val_ratio=0.2)
    model.flag = model_flag
    net = model
    visual_samples = torch.tensor(train.data[np.where(train.ids == 1)], dtype=torch.float32).to(net.device)
    net.set_visual_samples(visual_samples)
    if contra:
        cmapss.CmapssNegativeSampler(train, 1, neg_samples)
    net.prepare_data(train, test, val, batch_size=batch_size, num_workers=0, eval_shuffle=False)
    net.train_model(epoch=100,
                    lr=1e-3,
                    criterion=rul.MSEContrastiveLoss(contrastive=Loss) if contra else torch.nn.MSELoss(),
                    optimizer="adam",
                    lr_lambda=lambda x: 10 ** -(x // 15),
                    early_stop=10,
                    show_batch_loss=False)
    return net


if __name__ == '__main__':
    length = 30
    negs = 5
    bs = 1024
    dataset = cmapss.Subset.FD004
    device = "cuda:1"
    exp_ti = 2
    contra_training = False
    label_norm = True

    # net = rul.CnnGru(len(cmapss.DEFAULT_SENSORS), window_size=length, model_flag="CnnGru", device=device)
    # net = rul.ResNet(in_features=len(cmapss.DEFAULT_SENSORS), window_size=length, model_flag="ResNet",
    #                  device=device)
    # net = rul.DAMCNN(length, len(cmapss.DEFAULT_SENSORS), model_flag="DAMCNN", device=device, label_norm=label_norm)
    # net = LSTMNet(len(cmapss.DEFAULT_SENSORS), device=device, model_flag="LSTM", label_norm=label_norm)
    # net = MLP(length,
    #           in_features=len(cmapss.DEFAULT_SENSORS),
    #           device=device, model_flag="MLP", label_norm=label_norm)
    # net = OrMLPNet(length, len(cmapss.DEFAULT_SENSORS), 10, model_flag="OrMLPNet4",
    #                regularize=True,
    #                device=device)
    net = Rnn2DNet(window_size=length, features=len(cmapss.DEFAULT_SENSORS), hidden_dim=256, model_flag="Rnn2D-V1",
                   device=device, label_norm=label_norm)
    net = train_cmapss(net, length, negs, bs, dataset, exp_time=exp_ti, contra=contra_training, label_norm=label_norm)
    _ = compute_metrics(net.get_model_result_path())
