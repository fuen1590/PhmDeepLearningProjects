import torch
from torch import nn
from models.RULPrediction.ContrastiveModules import ContrastiveModel

"""
10.1109/TIM.2022.3210933
"""


class channel_attn(nn.Module):
    # input_size:(N, C, H, W)
    # output_size:(N, C, 1, 1)
    def __init__(self, window_size=8192, features=2):
        super().__init__()
        self.max_pool = nn.MaxPool2d([1, features])  # (2, 8192, 1, 2) -> (2, 8192, 1, 1)
        self.avg_pool = nn.AvgPool2d([1, features])  # (2, 8192, 1, 2) -> (2, 8192, 1, 1)
        self.channel_attn_fc1 = nn.Linear(window_size, window_size)  # (2, 8192, 1, 1) -> (2, 8192, 1, 1)
        self.channel_attn_fc2 = nn.Linear(window_size, window_size)  # (2, 8192, 1, 1) -> (2, 8192, 1, 1)
        self.window_size = window_size

    def forward(self, x):
        max_pool_x = self.max_pool(x).squeeze()
        avg_pool_x = self.avg_pool(x).squeeze()
        max_pool_x = self.channel_attn_fc1(max_pool_x)
        max_pool_x = self.channel_attn_fc2(max_pool_x)
        avg_pool_x = self.channel_attn_fc1(avg_pool_x)
        avg_pool_x = self.channel_attn_fc2(avg_pool_x)
        x = torch.sigmoid(max_pool_x + avg_pool_x)
        x = x.reshape(-1, self.window_size, 1, 1)
        return x


class temp_attn(nn.Module):
    # input_size:(N, C, H, W)
    # output_size:(N, 1, H, W)
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        max_pool_x, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_x = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat((avg_pool_x, max_pool_x), dim=1)
        x = torch.sigmoid(self.conv(x))
        return x


class CBAM(nn.Module):
    # input_size:(N, C, H, W)
    # output_size:(N, C, H, W)
    def __init__(self, window_size, features):
        super().__init__()
        self.channel_attn = channel_attn(window_size, features)
        self.temp_attn = temp_attn()

    def forward(self, x):
        channel_x = self.channel_attn(x)
        x = channel_x * x
        temp_x = self.temp_attn(x)
        x = temp_x * x
        return x


class MSCNN(nn.Module):
    # input_size: (N, window_size, 1, features)
    # output_size: (N, 512, 1, 32)
    def __init__(self, window_size, features):
        super().__init__()
        self.window_size = window_size
        self.features = features
        self.conv1 = nn.Sequential(nn.Conv1d(self.features, 8, 1, 10 if self.window_size % 10 == 0 else 16),
                                   nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(8))
        self.conv2 = nn.Sequential(nn.Conv1d(self.features, 8, 3, 10 if self.window_size % 10 == 0 else 16),
                                   nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(8))
        self.conv3 = nn.Sequential(nn.Conv1d(self.features, 8, 5, 10 if self.window_size % 10 == 0 else 16),
                                   nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(8))
        self.conv4 = nn.Sequential(nn.Conv1d(self.features, 8, 7, 10 if self.window_size % 10 == 0 else 16),
                                   nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(8))

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(-1), -1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class PRED(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 1, 1)
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.dense = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)  # (N, 1, 1, 128)
        x = x.contiguous().view(x.size(0), -1, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :].contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return x


class DAMCNN(ContrastiveModel):
    def __init__(self, window_size, features,
                 label_norm=False, model_flag="Model", device="cuda:0"):
        super().__init__(model_flag=model_flag, device=device, label_norm=label_norm)
        self.cbam = CBAM(window_size=window_size, features=features)
        self.mscnn = MSCNN(window_size=window_size, features=features)
        self.conv1 = nn.Sequential(nn.Conv1d(32, 32, 3, 1, 1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.Conv1d(32, 32, 3, 1, 1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.AvgPool1d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.Conv1d(64, 64, 3, 1, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.AvgPool1d(2, 2))
        self.conv = nn.Conv2d(64, 1, 1)
        # prediction layers
        self.conv_2 = nn.Conv2d(64, 1, 1)
        self.lstm_2 = nn.LSTM(1, 32, batch_first=True)
        self.dense = nn.Linear(32, 1)
        self.to(device)

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            feature = self.feature_extractor(x)
            out = self.dense(feature)  # (N, 1)
            return out
        else:
            assert label is not None
            pos, pos_aug, neg, weights = self.generate_contrastive_samples(x, label)
            out_all = self.dense(pos)
            neg_nums = neg.shape[1]
            neg_out = []
            for neg_i in range(neg_nums):
                neg_out.append(self.dense(neg[:, neg_i]))
            neg_out = torch.concat(neg_out, dim=-1)
            return torch.concat([out_all, neg_out], dim=-1), pos, pos_aug, neg, weights

    def feature_extractor(self, x):
            x = torch.unsqueeze(x, -2)
            x = self.cbam(x)  # (N, 8192, 1, 2)
            x = self.mscnn(x)  # (N, 32, 512)
            x = self.conv1(x)  # (N, 32, 256)
            x = self.conv2(x)  # (N, 64, 128)
            x = x.unsqueeze(2)

            x = self.conv_2(x)  # (N, 1, 1, 128)
            x = x.contiguous().view(x.size(0), -1, 1)
            x, _ = self.lstm_2(x)
            x = x[:, -1, :].contiguous().view(x.size(0), -1)
            return x

    def epoch_start(self):
        super(DAMCNN, self).epoch_start()
