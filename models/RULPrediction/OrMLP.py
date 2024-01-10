import torch
import torch.nn as nn
from models.RULPrediction.ContrastiveModules import ContrastiveModel, pn_rul_compute


class OrMLPMapper(nn.Module):
    def __init__(self, window_size, feature_num, hidden_modules=10, device="cuda:0"):
        super(OrMLPMapper, self).__init__()
        self.feature_num = feature_num
        self.hidden_modules = hidden_modules
        self.feature_linear = torch.nn.ModuleList()
        self.temporal_linear = torch.nn.ModuleList()
        for i in range(hidden_modules):
            self.feature_linear.append(
                nn.Sequential(
                    nn.Linear(in_features=feature_num, out_features=feature_num),
                    nn.LeakyReLU()
                )
            )

        for i in range(hidden_modules):
            self.temporal_linear.append(
                nn.Sequential(
                    nn.Linear(in_features=window_size, out_features=window_size),
                    nn.LeakyReLU()
                )
            )
        self.feature_fuse = nn.Linear(in_features=hidden_modules * feature_num,
                                      out_features=hidden_modules * feature_num)
        self.temporal_fuse = nn.Linear(in_features=hidden_modules * window_size,
                                       out_features=hidden_modules * window_size)
        self.device = device
        self.to(device)

    def forward(self, x):  # x.shape = (batch, window, feature)
        ff = []
        for layer in self.feature_linear:
            ff.append(layer(x))  # (batch, window, feature)
        ff = torch.concat(ff, dim=-1)  # (batch, window, n*feature)
        ff = self.feature_fuse(ff)  # (batch, window, n*feature)
        tf = []
        x = torch.transpose(ff, -1, -2)  # (batch, n*feature, window)
        for layer in self.temporal_linear:
            tf.append(layer(x))
        tf = torch.concat(tf, dim=-1)  # (batch, n*feature, n*window)
        tf = self.temporal_fuse(tf)
        tf = self.feature_fuse(tf.transpose(-1, -2))
        return tf

    def orthogonal_loss(self, reg=1e-6):
        loss = 0
        for layer in self.feature_linear:
            weights = layer[0].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
        for layer in self.temporal_linear:
            weights = layer[0].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
        weights = self.fuse.weight
        org = weights @ weights.T
        org = org - torch.eye(org.shape[0]).to(self.device)
        loss = loss + reg * org.abs().sum()
        return loss


class OrMLP(nn.Module):
    def __init__(self, feature_num, hidden_modules, device="cuda:0"):
        super(OrMLP, self).__init__()
        assert feature_num % hidden_modules == 0
        self.feature_num = feature_num
        self.hidden_modules = hidden_modules
        self.sub_feature_num = self.feature_num//self.hidden_modules
        self.feature_linear = torch.nn.ModuleList()
        for i in range(hidden_modules):
            self.feature_linear.append(
                nn.Sequential(
                    nn.Linear(in_features=feature_num//hidden_modules,
                              out_features=feature_num//hidden_modules),
                    nn.LeakyReLU()
                )
            )
        self.fuse = nn.Linear(in_features=feature_num, out_features=feature_num)
        self.device = device
        self.to(device)

    def forward(self, x):  # x.shape = (batch, feature_num)
        f = []
        for i in range(self.hidden_modules):
            block = x[:, i*self.sub_feature_num:i*self.sub_feature_num+self.sub_feature_num]
            f.append(self.feature_linear[i](block)+block)
        return self.fuse(torch.concat(f, dim=-1))

    def orthogonal_loss(self, reg=1e-6):
        loss = 0
        for layer in self.feature_linear:
            weights = layer[0].weight
            org = weights @ weights.T
            org = org - torch.eye(org.shape[0]).to(self.device)
            loss = loss + reg * org.abs().sum()
        weights = self.fuse.weight
        org = weights @ weights.T
        org = org - torch.eye(org.shape[0]).to(self.device)
        loss = loss + reg * org.abs().sum()
        return loss


class OrMLPNet(ContrastiveModel):
    def __init__(self,
                 window_size,
                 feature_num,
                 hidden_modules,
                 regularize=True,
                 model_flag="OrMLPNet",
                 device="cuda:0",
                 label_norm=True):
        super(OrMLPNet, self).__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        self.hidden_dim = hidden_modules*(window_size+feature_num)
        self.input_layer = OrMLPMapper(window_size=window_size,
                                       feature_num=feature_num,
                                       hidden_modules=hidden_modules,
                                       device=device)
        self.mlp1 = OrMLP(feature_num=self.hidden_dim,
                          hidden_modules=hidden_modules,
                          device=device)
        self.mlp2 = OrMLP(feature_num=self.hidden_dim,
                          hidden_modules=hidden_modules,
                          device=device)
        self.mlp3 = OrMLP(feature_num=self.hidden_dim,
                          hidden_modules=hidden_modules,
                          device=device)
        self.mlp4 = OrMLP(feature_num=self.hidden_dim,
                          hidden_modules=hidden_modules,
                          device=device)
        self.out_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.regularize = regularize
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            feature = self.feature_extractor(x)
            print(feature.shape)
            return self.out_layer(feature)
            # return feature
        else:
            f_pos, f_apos, f_neg, w = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.out_layer, f_pos, f_neg), f_pos, f_apos, f_neg, w

    def feature_extractor(self, x):
        x = self.input_layer(x)
        print(x.shape)
        x = self.mlp1(x)
        print(x.shape)
        x = self.mlp2(x)
        print(x.shape)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

    def orthogonal_loss(self, reg=1e-6):
        loss = self.input_layer.orthogonal_loss(reg)
        return loss

    def compute_loss(self,
                     x: torch.Tensor,
                     label: torch.Tensor,
                     criterion) -> [torch.Tensor, torch.Tensor]:
        if self.regularize:
            [loss, rul] = super(OrMLPNet, self).compute_loss(x, label, criterion)
            return loss+self.orthogonal_loss(), rul
        else:
            return super(OrMLPNet, self).compute_loss(x, label, criterion)


if __name__ == '__main__':
    import dataset.utils as u
    w = 30
    f = 14
    a = torch.ones((32, w, f))
    # b = torch.ones((32, 5, w, f))
    # l = torch.ones((32, 5))
    model = OrMLPNet(w, f, 10, device="cpu")
    a = model(a)
    # b = model(b, l)
    print(u.count_parameters(model))
