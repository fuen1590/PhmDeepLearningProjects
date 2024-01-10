import torch
import torch.nn as nn
import numpy as np
from ContrastiveModules import ContrastiveModel, pn_rul_compute


class MLPBlock(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_dim,
                 out_features,
                 dropout=0.5,
                 device="cuda:0"):
        super(MLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_features),
            nn.Dropout(dropout)
        )
        self.to(device)

    def forward(self, x):
        return self.block(x)


class GatedAttention(nn.Module):
    def __init__(self, hidden_dim, dim=-1, device="cuda:0"):
        super(GatedAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            # nn.Softmax(dim=dim),
            nn.Sigmoid()
        )
        self.weights = None
        self.to(device)

    def forward(self, x):
        weights = self.encoder(x)
        self.weights = weights
        return torch.mul(weights, x)


class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, device="cuda:0"):
        super(MLPLayer, self).__init__()
        self.mlp = MLPBlock(in_features=in_features, hidden_dim=hidden_dim, out_features=out_features, device=device)
        self.gat = GatedAttention(hidden_dim=out_features, device=device)
        self.to(device)

    def forward(self, x):
        f = self.mlp(x)
        f = self.gat(f)
        return f


class MixerLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, device="cuda:0"):
        super(MixerLayer, self).__init__()
        self.time_mixer = MLPLayer(in_features=hidden_dim, hidden_dim=hidden_dim*2, out_features=hidden_dim,
                                   device=device)
        self.feature_mixer = MLPLayer(in_features=in_features, hidden_dim=in_features*2, out_features=in_features,
                                      device=device)
        self.to(device)

    def forward(self, x):
        # x.shape = (b, h, f)
        x = x.transpose(-1, -2)
        f = self.time_mixer(x) + x  # (b, f, h)
        f = f.transpose(-1, -2)
        f = self.feature_mixer(f) + f  # (b, h, f)
        return f


class MLPMixer(ContrastiveModel):
    def __init__(self, window_size, in_features, hidden_dim, num_layers, filter_size=0,
                 device="cuda:0", model_flag="TSMixer", label_norm=True):
        super(MLPMixer, self).__init__(device=device, label_norm=label_norm, model_flag=model_flag)
        if filter_size > 0:
            window_size = window_size // filter_size
            self.MaV = nn.AvgPool1d(kernel_size=filter_size, stride=filter_size)
        else:
            window_size = window_size
            self.MaV = None
        self.window_size = window_size
        self.in_features = in_features
        self.input_embedding = nn.Linear(in_features=window_size, out_features=hidden_dim)
        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(MixerLayer(in_features=in_features, hidden_dim=hidden_dim, device=device))
        self.output = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=in_features*hidden_dim, out_features=1)
        )
        self.to(device)

    def feature_extractor(self, x):
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        # x.shape = (b, w, f)
        emb = self.input_embedding(x.transpose(-1, -2))  # (b, f, h)
        f = self.layers(emb.transpose(-1, -2))
        return torch.flatten(f, start_dim=-2, end_dim=-1)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.output(x)
        else:
            f_pos, f_apos, f_neg, weight = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.output, f_pos, f_neg), f_pos, f_apos, f_neg, weight


class DualMLPLayer(nn.Module):
    def __init__(self, window_size, hidden_dim, dropout=0.5):
        super(DualMLPLayer, self).__init__()
        self.block1 = nn.Sequential(
            # nn.LayerNorm(normalized_shape=window_size, elementwise_affine=False),
            nn.Linear(in_features=window_size, out_features=window_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=window_size * 2, out_features=window_size),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            # nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=False),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(normalized_shape=window_size, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=True)
        self.ln3 = nn.LayerNorm(normalized_shape=window_size, elementwise_affine=True)
        self.ln4 = nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=True)
        self.gat_weights_1 = None
        self.gat_weights_2 = None
        self.gat1 = GatedAttention(hidden_dim=window_size, dim=-1)
        self.gat2 = GatedAttention(hidden_dim=hidden_dim, dim=-2)

    def forward(self, x1, x2):
        # x1.shape = (b, w, f), x2.shape = (b, w, f)
        x1 = x1.transpose(-1, -2)  # x1.shape = (b, f, w)
        x1 = self.ln1(self.block1(x1) + x1)  # x1.shape = (b, f, w)
        x2 = self.ln2(self.block2(x2) + x2)  # x2.shape = (b, w, f)
        x1 = self.ln3(x1 + self.gat2(x2).transpose(-1, -2))
        x2 = self.ln4(x2 + self.gat1(x1).transpose(-1, -2))  # x2.shape = (b, f, w)
        self.gat_weights_1 = self.gat1.weights
        self.gat_weights_2 = self.gat2.weights
        return x1.transpose(-1, -2), x2


class DualMLPMixer(ContrastiveModel):
    def __init__(self,
                 window_size,
                 in_features,
                 hidden_dim,
                 num_layers,
                 dropout=0.5,
                 or_loss=False,
                 model_flag="MLPDualMixer", device="cuda:0", label_norm=True,
                 filter_size=0):
        super(DualMLPMixer, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        if filter_size > 0:
            window_size = window_size // filter_size
            self.MaV = nn.Conv1d(in_channels=in_features, out_channels=hidden_dim, kernel_size=filter_size,
                                 stride=filter_size)
            self.input_embedding = None
        else:
            window_size = window_size
            self.MaV = None
            self.input_embedding = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DualMLPLayer(window_size=window_size, hidden_dim=hidden_dim, dropout=dropout))
        self.out_gat1 = GatedAttention(hidden_dim=window_size)
        self.out_gat2 = GatedAttention(hidden_dim=hidden_dim, dim=-2)
        # self.fuse = nn.Linear(in_features=in_features*hidden_dim, out_features=768)
        self.output = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim*window_size, out_features=1)
        )
        self.hidden_out_1 = []
        self.hidden_out_2 = []
        self.gat_weights_1 = []
        self.gat_weights_2 = []
        self.or_loss = or_loss
        self.to(device)

    def feature_extractor(self, x):
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        self.hidden_out_1 = []
        self.hidden_out_2 = []
        # x.shape = (b, w, f)
        x = self.input_embedding(x) if self.input_embedding is not None else x  # x.shape = (b, w, h)
        f1 = x
        f2 = x
        for l in self.layers:
            f1, f2 = l(f1, f2)
            # self.hidden_out_1.append(f1)
            # self.hidden_out_2.append(f2)
            # self.gat_weights_1.append(l.gat_weights_1)
            # self.gat_weights_2.append(l.gat_weights_2)
        f1 = self.out_gat1(f1.transpose(-1, -2))
        f2 = self.out_gat2(f2)
        f = torch.flatten(f1.transpose(-1, -2) + f2, start_dim=-2, end_dim=-1)
        # f = torch.flatten(f1 + f2, start_dim=-2, end_dim=-1)
        return f

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return self.output(x)
        else:
            f_pos, f_apos, f_neg, weight = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.output, f_pos, f_neg), f_pos, f_apos, f_neg, weight

    def orthogonal_loss(self, reg=1e-6):
        loss = 0
        weights = self.input_embedding.weight
        org = weights @ weights.T
        org = org - torch.eye(org.shape[0]).to(self.device)
        loss = loss + reg * org.abs().sum()
        for layer in self.layers:
            weights = layer.block1[0].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
            weights = layer.block1[3].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
            weights = layer.block2[0].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
            weights = layer.block2[3].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
        return loss

    def compute_loss(self,
                     x: torch.Tensor,
                     label: torch.Tensor,
                     criterion) -> [torch.Tensor, torch.Tensor]:
        [loss, rul] = super(DualMLPMixer, self).compute_loss(x, label, criterion)
        if self.or_loss:
            return loss + self.orthogonal_loss(), rul
        else:
            return loss, rul


if __name__ == '__main__':
    from dataset.utils import count_parameters
    net = DualMLPMixer(window_size=15, in_features=14, hidden_dim=256, num_layers=12, dropout=0.4)
    count_parameters(net)
    # import dataset.cmapss as cmapss
    # import models.RULPrediction as rul
    # from dataset.utils import compute_metrics
    # net.load_state_dict(torch.load(r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-MLPDualMixer-h32-8-norm1-w30-batch51-thresh125-FD001-neg4-InfoNCE1/model.pt"))
    # train, test, val, scalar = cmapss.get_data(cmapss.DEFAULT_ROOT,
    #                                            cmapss.Subset.FD001,
    #                                            window_size=30,
    #                                            slide_step=1,
    #                                            sensors=cmapss.DEFAULT_SENSORS,
    #                                            rul_threshold=125,
    #                                            label_norm=True,
    #                                            val_ratio=0.2)
    # inp = torch.FloatTensor(np.stack([test[i][0] for i in range(3000)])).cuda()
    # out = net(inp[0:1])
