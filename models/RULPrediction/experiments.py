import dataset.cmapss as cmapss
import dataset.xjtu as xjtu
import models.RULPrediction as rul
from dataset.utils import compute_metrics

import torch
import numpy as np
from SimpleModels import *
from OrMLP import OrMLPNet
from MCRNN import Rnn2DNet
from MLPMixer import MLPMixer, DualMLPMixer
from BiGRU_TSAM import BiGRU_TSAM
from IMDSSN import IMDSSN


def train_model(model, train_set, test_set, val_set, model_flag,
                batch_size, visual_sample=None, contra=True):
    Loss = "InfoNCE" if contra else ""
    model.flag = model_flag
    net = model
    net.set_visual_samples(visual_sample)
    net.prepare_data(train_set, test_set, val_set, batch_size=batch_size, num_workers=0,
                     eval_shuffle=False)
    net.train_model(epoch=100,
                    lr=1e-3,
                    criterion=rul.MSEContrastiveLoss(contrastive=Loss) if contra else torch.nn.MSELoss(),
                    optimizer="adam",
                    # lr_lambda=lambda x: 10 ** -(x // 15),
                    early_stop=5,
                    show_batch_loss=False)
    return net


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
    model_flag = "RUL-{}-norm{}-w{}-batch{}-thresh{}-{}-neg{}-{}". \
        format(model.flag,
               1 if label_norm else 0,
               window_size,
               batch_size,
               threshold,
               subset.value,
               neg_samples - 1 if contra else 0,
               # "GSampler",
               exp_time)
    train, test, val, scalar = cmapss.get_data(cmapss.DEFAULT_ROOT,
                                               subset,
                                               window_size=window_size,
                                               slide_step=1,
                                               sensors=cmapss.DEFAULT_SENSORS,
                                               rul_threshold=threshold,
                                               label_norm=label_norm,
                                               val_ratio=0.2)
    if contra:
        # cmapss.CmapssPiecewiseNegativeSampler(train, 1, neg_samples)
        cmapss.CmapssGaussianNegativeSampler(train, neg_samples, thresh=0.5, std=0.3)
    visual_samples = torch.tensor(train.data[np.where(train.ids == 1)], dtype=torch.float32).to(model.device)
    model = train_model(model=model, train_set=train, test_set=test, val_set=val,
                        model_flag=model_flag, batch_size=batch_size, visual_sample=visual_samples, contra=contra)
    return model


def train_xjtu(model: rul.ContrastiveModel,
               window_size,
               step_size,
               condition: xjtu.Condition,
               neg_samples,
               batch_size,
               exp_time,
               contra=True):
    train_set = xjtu.XJTU(xjtu.DEFAULT_ROOT,
                          [condition],
                          bearing_indexes=[[1, 2, 3]],
                          start_tokens=[[1, 1, 1]],
                          end_tokens=[[-1, -1, -1]],
                          labels_type=xjtu.LabelsType.TYPE_P,
                          window_size=window_size,
                          step_size=step_size)
    test_set = xjtu.XJTU(xjtu.DEFAULT_ROOT,
                         [condition],
                         bearing_indexes=[[5]],
                         start_tokens=[[1]],
                         end_tokens=[[-1]],
                         labels_type=xjtu.LabelsType.TYPE_P,
                         window_size=window_size,
                         step_size=32768)
    val_set = xjtu.XJTU(xjtu.DEFAULT_ROOT,
                        [condition],
                        bearing_indexes=[[4]],
                        start_tokens=[[1]],
                        end_tokens=[[-1]],
                        labels_type=xjtu.LabelsType.TYPE_P,
                        window_size=window_size,
                        step_size=step_size)
    # scaler = xjtu.XJTUScaler()
    # scaler.fit_transform(train_set)
    # scaler.transform(test_set)
    # scaler.transform(val_set)
    batch_size = batch_size // neg_samples if contra else batch_size
    if condition.value == xjtu.Condition.OP_A.value:
        condition_value = "Con1"
    elif condition.value == xjtu.Condition.OP_B.value:
        condition_value = "Con2"
    else:
        condition_value = "Con3"
    model_flag = "RUL-{}-w{}-s{}-batch{}-{}-neg{}-{}". \
        format(model.flag,
               window_size,
               step_size,
               batch_size,
               condition_value,
               neg_samples - 1 if contra else 0,
               exp_time)
    if contra:
        xjtu.XJTURegressionPiecewiseNegativeSampler(train_set, neg_num=neg_samples)
    bearing1_end_token = train_set.bearing_split_index[0]
    vh = bearing1_end_token // 100
    visual_samples = []
    for i in range(0, bearing1_end_token - window_size, vh):
        visual_samples.append(train_set.raw_data[i:i+window_size])
    visual_samples = torch.tensor(np.stack(visual_samples, axis=0)).to(model.device)
    model = train_model(model, train_set, test_set, val_set, model_flag, batch_size, visual_samples,
                        contra)
    return model


if __name__ == '__main__':
    length = 30
    step_size = 1
    negs = 5
    bs = 1024
    # dataset = xjtu.Condition.OP_B
    dataset = cmapss.Subset.FD004
    device = "cuda:1"
    exp_ti = 1
    contra_training = False
    label_norm = True
    # filter_size = 0
    filter_size = 0

    # DA-Mixer only
    mixer_layer_num = 6
    hidden_dim = 32
    dropout = 0

    # net = rul.CnnGru(14, window_size=length, filter_size=0, model_flag="CnnGru", device=device)
    # net = rul.ResNet(in_features=14, window_size=length, model_flag="ResNet",
    #                  device=device)
    # net = rul.DAMCNN(length, 2, model_flag="DAMCNN", device=device, label_norm=label_norm)
    net = LSTMNet(length, 14, hidden_dim=256, device=device, model_flag="1temporal-test-LSTM-256", label_norm=label_norm)
    # net = MLP(length,
    #           hidden_dim=128,
    #           in_features=2,
    #           filter_size=filter_size,
    #           device=device, model_flag="MLP", label_norm=label_norm)
    # net = OrMLPNet(length, len(cmapss.DEFAULT_SENSORS), 10, model_flag="OrMLPNet4",
    #                regularize=True,
    #                device=device,
    #                label_norm=label_norm)
    # net = Rnn2DNet(features=14, hidden_dim=256, window_size_split=[30, 15, 6, 3, 1],
    #                kernel_size=3, dropout=0,
    #                model_flag="Rnn2D-V3-256-k3-l4", device=device, label_norm=label_norm)
    # net = MLP_LSTM(in_features=len(cmapss.DEFAULT_SENSORS), window_size=length,
    #                label_norm=label_norm, device=device)

    # net = MLPMixer(window_size=length,
    #                in_features=14,
    #                hidden_dim=hidden_dim,
    #                num_layers=mixer_layer_num,
    #                filter_size=filter_size,
    #                device=device, model_flag=f"TSMixer-{mixer_layer_num}", label_norm=label_norm)
    # net = BiGRU_TSAM(window_size=length,
    #                  in_features=14,
    #                  filter_size=filter_size,
    #                  model_flag="BiGRU-TSAM",
    #                  device=device)
    # net = IMDSSN(window_size=length,
    #              in_features=2,
    #              hidden_dim=32,
    #              encoder_nums=1,
    #              n_heads=2,
    #              model_flag="IMDSSN",
    #              pe=True,
    #              device=device,
    #              label_norm=label_norm,
    #              filter_size=filter_size)
    # net = DualMLPMixer(window_size=length,
    #                    in_features=len(cmapss.DEFAULT_SENSORS),
    #                    hidden_dim=hidden_dim,
    #                    num_layers=mixer_layer_num,
    #                    dropout=dropout,
    #                    or_loss=False,
    #                    device=device, model_flag=f"MLPDualMixer-h{hidden_dim}-{mixer_layer_num}", label_norm=label_norm,
    #                    filter_size=filter_size)

    net = train_cmapss(net, length, negs, bs, dataset, exp_time=exp_ti, contra=contra_training, label_norm=label_norm, )
    # _ = compute_metrics(net.get_model_result_path())
    # net = train_xjtu(net, window_size=length, step_size=step_size, condition=dataset,
    #                  neg_samples=negs, batch_size=bs, exp_time=exp_ti, contra=contra_training)
    _ = compute_metrics(net.get_model_result_path())
