import torch
import torch.nn as nn

import dataset
import train


class CNN(train.TrainableModule):
    def __init__(self, in_len, num_class):
        super(CNN, self).__init__(model_flag="CNN")
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.output = nn.Sequential(nn.Linear(in_features=in_len // 32 * 512, out_features=5))

    def forward(self, x):
        # x.shape = (batch, len, 2)
        b, l, d = x.shape
        x = x.view(b, d, l)  # x.shape = (batch, 2, len)
        x = self.layers(x)
        x = self.flatten(x)
        out = nn.functional.softmax(self.output(x), dim=-1)
        return out


if __name__ == '__main__':

    train.precision()
    # from dataset.xjtu import XJTU_Dataset, Condition
    #
    # model = CNN(10000, 5).to("cuda:0")
    # train_set = XJTU_Dataset(dataset.XJTU.DEFAULT_ROOT, Condition.OP_B, bearing_indexes=[[1, 1, 2, 3]],
    #                          start_tokens=[[1, 455, 47, 127]],
    #                          end_tokens=[[80, 491, 127, 207]])
    # test_set = XJTU_Dataset(dataset.XJTU.DEFAULT_ROOT, [Condition.OP_A, Condition.OP_C],
    #                         bearing_indexes=[[4, 4], [1, 3]],
    #                         start_tokens=[[20, 102], [2437, 362]],
    #                         end_tokens=[[40, -1], [2457, 371]])
    # val_set = XJTU_Dataset(dataset.XJTU.DEFAULT_ROOT, Condition.OP_B, bearing_indexes=[[1, 2, 3]],
    #                          start_tokens=[[40, 88, 167]],
    #                          end_tokens=[[80, 128, 207]])
    # model.prepare_data(train_set=train_set, test_set=test_set, eval_set=val_set, batch_size=256, num_workers=1)
    # model.train_model(100,
    #                   1e-4,
    #                   torch.nn.CrossEntropyLoss(),
    #                   "adam",
    #                   lambda x: 10 ** -(x // 20),
    #                   early_stop=0)
