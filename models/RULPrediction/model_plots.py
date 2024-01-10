import dataset.cmapss as cmapss
import dataset.xjtu as xjtu
import models.RULPrediction as rul
from dataset.utils import compute_metrics

import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as manifold

from SimpleModels import *
from OrMLP import OrMLPNet
from MCRNN import Rnn2DNet
from MLPMixer import MLPMixer, DualMLPMixer
from BiGRU_TSAM import BiGRU_TSAM
from IMDSSN import IMDSSN


def load_model(net, path, length=30):
    weights = torch.load(path)
    net.load_state_dict(weights)
    train, test, val, scalar = cmapss.get_data(cmapss.DEFAULT_ROOT,
                                               cmapss.Subset.FD004,
                                               window_size=length,
                                               slide_step=1,
                                               sensors=cmapss.DEFAULT_SENSORS,
                                               rul_threshold=125,
                                               label_norm=True,
                                               val_ratio=0.2)
    data = torch.tensor(train.data[np.where(train.ids == 1)], dtype=torch.float32).to("cuda")
    y = train.labels[np.where(train.ids == 1)]
    out = net(data)
    features = net.feature_extractor(data).cpu().detach().numpy().squeeze()
    tsne = manifold.TSNE(n_components=3, random_state=2023)
    embedding = tsne.fit_transform(features)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.plot(y, label="GT")
    # plt.plot(out[:, 0].detach().cpu().numpy(), label="Predict")
    # plt.legend(prop={'size': 15})
    # plt.grid()
    # plt.show()
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.scatter(embedding[:, 0], embedding[:, 1],
    #             c=plt.cm.tab20(0),
    #             edgecolors=plt.cm.Wistia(range(len(embedding[:, 0]))))
    # plt.show()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.tick_params(axis='z', labelsize=15)
    ax.scatter(embedding[:, 0], embedding[:, 1], np.ones_like(embedding[:, 0]) * np.min(embedding[:, 2])-2, c="silver")
    ax.scatter(embedding[:, 0], np.ones_like(embedding[:, 0]) * np.max(embedding[:, 1])+2, embedding[:, 2], c="silver")
    ax.scatter(np.ones_like(embedding[:, 0]) * np.min(embedding[:, 0])-2, embedding[:, 1], embedding[:, 2], c="silver")
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c='deeppink', edgecolors=plt.cm.gist_yarg(range(len(embedding[:, 0]))))
    plt.show()
    # np.save("/home/fuen/DeepLearningProjects/FaultDiagnosis/models/RULPrediction/features.npy",
    #         embedding)
    return net


# net = rul.CnnGru(14, window_size=30, filter_size=0, model_flag="CnnGru", device="cuda")
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-CnnGru-norm1-w30-batch1024-thresh125-FD004-neg0-1/model.pt")
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-CnnGru-norm1-w30-batch204-thresh125-FD004-neg4-InfoNCE1/model.pt")
#
length = 40
net = rul.DAMCNN(length, 14, model_flag="DAMCNN", device="cuda", label_norm=True)
net = load_model(net,
                 r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-DAMCNN-w40-batch1024-thresh125-FD004-neg0-1/model.pt",
                 length)
net = load_model(net,
                 r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-DAMCNN-w40-batch204-thresh125-FD004-neg4-InfoNCE1/model.pt",
                 length)
# length = 30
# net = IMDSSN(window_size=30,
#              in_features=14,
#              hidden_dim=32,
#              encoder_nums=1,
#              n_heads=2,
#              model_flag="IMDSSN",
#              pe=True,
#              device="cuda",
#              label_norm=True,
#              filter_size=0)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-IMDSSN-norm1-w30-batch128-thresh125-FD004-neg0-1/model.pt",
#                  length)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-IMDSSN-norm1-w30-batch25-thresh125-FD004-neg4-2/model.pt",
#                  length)
#
# length = 30
# net = DualMLPMixer(window_size=length,
#                    in_features=len(cmapss.DEFAULT_SENSORS),
#                    hidden_dim=32,
#                    num_layers=6,
#                    dropout=0,
#                    or_loss=False,
#                    device="cuda", model_flag=f"MLPDualMixer-h{32}-{6}", label_norm=True,
#                    filter_size=0)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-MLPDualMixer-h32-6-norm1-w30-batch128-thresh125-FD004-neg0-2/model.pt",
#                  length)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-MLPDualMixer-h32-6-norm1-w30-batch25-thresh125-FD004-neg4-GSampler1/model.pt",
#                  length)

# length = 15
# net = BiGRU_TSAM(window_size=length,
#                  in_features=14,
#                  filter_size=0,
#                  model_flag="BiGRU-TSAM",
#                  device="cuda")
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-BiGRU-TSAM-norm1-w15-batch256-thresh125-FD004-neg0-1/model.pt",
#                  length)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-BiGRU-TSAM-norm1-w15-batch51-thresh125-FD004-neg4-1/model.pt",
#                  length)

length = 30
net = MLP(length,
          hidden_dim=256,
          in_features=14,
          filter_size=0,
          device="cuda", model_flag="MLP", label_norm=True)
net = load_model(net,
                 r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-MLP-norm1-w30-batch1024-thresh125-FD004-neg0-1/model.pt",
                 length)
net = load_model(net,
                 r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-MLP-norm1-w30-batch204-thresh125-FD004-neg4-InfoNCE1/model.pt",
                 length)

# length = 15
# net = MLPMixer(window_size=length,
#                in_features=14,
#                hidden_dim=32,
#                num_layers=6,
#                filter_size=0,
#                device="cuda", model_flag=f"TSMixer-{6}", label_norm=True)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-TSMixer-6-norm1-w15-batch256-thresh125-FD004-neg0-1/model.pt",
#                  length)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-TSMixer-6-norm1-w15-batch51-thresh125-FD004-neg4-1/model.pt",
#                  length)

# length = 30
# net = LSTMNet(length, 14, hidden_dim=256, device="cuda", model_flag="1temporal-test-LSTM-256", label_norm=True)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-1temporal-test-LSTM-256-norm1-w30-batch256-thresh125-FD004-neg0-1/model.pt",
#                  length)
# net = load_model(net,
#                  r"/home/fuen/DeepLearningProjects/FaultDiagnosis/train/model_result/RUL-1temporal-test-LSTM-256-norm1-w30-batch51-thresh125-FD004-neg4-1/model.pt",
#                  length)


