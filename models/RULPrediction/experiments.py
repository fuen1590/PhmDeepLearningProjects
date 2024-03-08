import dataset.cmapss as cmapss
import dataset.xjtu as xjtu
import models.RULPrediction as rul
from dataset.utils import compute_metrics

import torch
import numpy as np
from SimpleModels import *
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


if __name__ == '__main__':
    length = 30
    step_size = 1  # a step size to construct training samples
    negs = 5  # the number of negative samples if using FSGRI
    bs = 1024
    dataset = cmapss.Subset.FD004  # a enum object, see detail in cmapss.Subset.
    device = "cuda:0"  # which device, 'cpu', 'cuda', 'cuda:*'
    exp_ti = 1  # experiment count, using to construct a model_flag
    contra_training = False  # if using FSGRI
    label_norm = True  # if True, the RUL label will be in [0, 1], else [0, number of cycles]
    # filter_size = 0

    # Dual-Mixer only
    mixer_layer_num = 6
    hidden_dim = 32
    dropout = 0
    net = DualMLPMixer(window_size=length,
                       in_features=len(cmapss.DEFAULT_SENSORS),
                       hidden_dim=hidden_dim,
                       num_layers=mixer_layer_num,
                       dropout=dropout,
                       or_loss=False,
                       device=device, model_flag=f"MLPDualMixer-h{hidden_dim}-{mixer_layer_num}", label_norm=label_norm,
                       filter_size=0)

    net = train_cmapss(net, length, negs, bs, dataset, exp_time=exp_ti, contra=contra_training, label_norm=label_norm, )
