import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.contrib import itertools

from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from pythonProject3.dann.test import test
from visdom import Visdom
from confusionmatrix import ConfusionMatrix
import matplotlib.pyplot as plt
source_dataset_name = 'train03'
target_dataset_name = 'train04'
test_dataset_name = 'train02'
source_image_root = os.path.join('/home/dulongkun/dataset/xjtu01', source_dataset_name)
target_image_root = os.path.join('/home/dulongkun/dataset/xjtu01', target_dataset_name)
model_root = '/home/dulongkun/dataset/xjtu/models'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 64
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data

img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,),
                             std=(0.3081,))
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
])
train_list = os.path.join('/home/dulongkun/dataset/chicun', 'val的副本.txt')

dataset_source = GetLoader(
    data_root=os.path.join(source_image_root),
    data_list=train_list,
    transform=img_transform_target
)

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

test_list = os.path.join('/home/dulongkun/dataset/chicun', 'val的副本.txt')

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root),
    data_list=test_list,
    transform=img_transform_target
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

# load model

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
print(len(dataset_source))
print(len(dataset_target))

train_loss = []
s_acc = []
t_acc = []
v_acc = []
for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)
    # print(len(dataloader_source))
    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()



        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)

        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label



        err.backward()
        optimizer.step()



        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        # 保存权重文件
        torch.save(my_net, '{0}/xjtu03.pth'.format(model_root))

    print('\n')
    accu_s = test(source_dataset_name)
    s_acc.append(accu_s)
    print('Accuracy of the %s dataset: %f' % ('source', accu_s))
    accu_t = test(target_dataset_name)
    t_acc.append(accu_t)
    print('Accuracy of the %s dataset: %f' % ('target', accu_t))
    accu_v = test(test_dataset_name)
    v_acc.append(accu_v)
    print('Accuracy of the %s dataset: %f\n' % ('val', accu_v))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        best_accu_v = accu_v
        torch.save(my_net, '{0}/xjtu03_best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('source', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))

print('Accuracy of the %s dataset: %f' % ('test', best_accu_v))
print('Corresponding model was save in ' + model_root + '/12.pth')

plt.plot(np.arange(len(s_acc)), s_acc, label='s_acc')
plt.plot(np.arange(len(t_acc)), t_acc, label='t_acc')
plt.plot(np.arange(len(v_acc)), v_acc, label='v_acc')

plt.legend()
plt.xlabel('epoch')
plt.title('Model acc&loss')
plt.show()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = [0, 1, 2]
confusion = ConfusionMatrix(num_classes=3, labels = labels)
my_net.eval()
with torch.no_grad():
    for val_data in dataloader_target:
        val_images, val_labels = val_data
        outputs, _ = my_net(val_images.to(device), alpha)
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
confusion.plot(4000)
confusion.summary()
