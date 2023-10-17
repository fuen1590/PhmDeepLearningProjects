import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import time

root = os.path.dirname(__file__)


def _check_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


class TrainableModule(nn.Module):
    """
    The base module of Trainable Models. So call 'trainable' means the models can be trained by
    following method easily:

        >>> model.prepare_data(...)
        >>> model.train_model(...)
    """

    def __init__(self, model_flag="model", device="cuda"):
        super(TrainableModule, self).__init__()
        self.eval_losses = None
        self.train_losses = None

        self.eval_loader = None
        self.test_loader = None
        self.train_loader = None

        self.optimizer = None
        self.criterion = None
        self.lr_schedular = None

        self.flag = model_flag
        self.device = device

    def prepare_data(self,
                     train_set: Dataset,
                     test_set: Dataset,
                     eval_set: Dataset = None,
                     batch_size: int = 256,
                     num_workers: int = 8,
                     eval_shuffle=True):
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
        if eval_set is not None:
            self.eval_loader = DataLoader(eval_set, batch_size, shuffle=eval_shuffle, num_workers=num_workers)
        print("train size:{}".format(len(train_set)))
        print("test size:{}".format(len(test_set)))
        print("validate size:{}".format(len(eval_set)))
        _check_path(self.get_model_result_path())

    def train_model(self,
                    epoch: int,
                    lr: float,
                    criterion,
                    optimizer: str = "adam",
                    lr_lambda=None,
                    early_stop=2,
                    show_batch_loss=False):
        if self.train_loader is None:
            raise RuntimeError("The data_loader is None! Set the param data_loader not None or use "
                               "model.prepare_data(Dataset, batch_size, num_workers) to provide the"
                               "training data.")
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(lr=lr, params=self.parameters())
        elif optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(lr=lr, params=self.parameters())
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(lr=lr, params=self.parameters())
        else:
            raise RuntimeError("Unknown optimizer {}.".format(optimizer))
        if lr_lambda is not None:
            self.lr_schedular = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)
        if early_stop is not None and early_stop > 0:
            mini_eval_loss = None
            patience = early_stop
            now_patience = 0
        self.criterion = criterion
        self.train_losses = []
        self.eval_losses = []
        print("Model flag: {}".format(self.flag))
        print("Start training epoch {}".format(epoch))

        # training
        start_time = time.time()
        self.train_start()  # callback function
        for e in range(epoch):
            self.epoch_start()  # callback function
            print("epoch: {}/{}".format(e + 1, epoch))
            epoch_start_time = time.time()
            self.train()
            batch_losses = []
            for step, (x, y) in enumerate(self.train_loader):
                x = x.to(torch.float32).to(self.device)
                y = y.to(torch.float32).to(self.device)
                loss, out = self.compute_loss(x, y, self.criterion)
                if step == 0 and e == epoch-1:
                    np.save(self.get_model_result_path() + "train_x_batch{}".format(step), y.cpu().detach().numpy())
                    np.save(self.get_model_result_path() + "train_y_batch{}".format(step), x.cpu().detach().numpy())
                if show_batch_loss:
                    print("\tbatch: {}/{}, loss:{:.4f}".format(step + 1, len(self.train_loader), loss.item()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

            if self.lr_schedular is not None:
                self.lr_schedular.step()
            batch_loss = np.average(batch_losses)
            self.train_losses.append(batch_loss)
            batch_losses.clear()

            # evaluation
            if self.eval_loader is not None:
                self.eval()
                eval_losses = []

                with torch.no_grad():
                    for step, (e_x, e_y) in enumerate(self.eval_loader):
                        e_x = e_x.to(torch.float32).to(self.device)
                        e_y = e_y.to(torch.float32)
                        loss, _ = self.compute_loss(e_x, e_y, self.criterion)
                        eval_losses.append(loss.item())
                    eval_loss = np.average(eval_losses)
                    self.eval_losses.append(eval_loss)

                print("\ttraining loss: {:.4}\n \teval loss: {:.4} \tCurrent learning rate: {}".
                      format(batch_loss, eval_loss, self.optimizer.state_dict()['param_groups'][0]['lr']))
            else:
                print("\ttraining loss: {:.4}\n \tCurrent learning rate: {}".
                      format(batch_loss, self.optimizer.state_dict()['param_groups'][0]['lr']))

            print("\tEpoch time spent: %s s" % (time.time() - epoch_start_time))
            # early stop
            if early_stop > 0:
                if mini_eval_loss is None:
                    mini_eval_loss = eval_loss
                    torch.save(self.state_dict(), self.get_model_result_path() + 'check_point.pt')
                    continue
                if eval_loss >= mini_eval_loss:
                    now_patience = now_patience + 1
                    print("\tEarly Stopping Monitor: bigger eval loss, now patience score {}/{}"
                          .format(now_patience, patience))
                else:
                    now_patience = 0
                    mini_eval_loss = eval_loss
                    print("\tEarly Stopping Monitor: smaller eval loss achieved, saving model...")
                    torch.save(self.state_dict(), self.get_model_result_path() + 'check_point.pt')
                if now_patience >= patience:
                    print("\tEarly Stopping in epoch {}".format(e))
                    self.load_state_dict(torch.load(self.get_model_result_path() + 'check_point.pt'))
                    break
            self.epoch_end()  # callback function
        end_time = time.time()
        self.train_end()  # callback function
        print("Total time spent: %s s" % round(end_time - start_time, 2))
        self.plot_losses()
        torch.save(self.state_dict(), self.get_model_result_path() + 'model.pt')
        self.test_model()

    def test_model(self):
        """
        TODO: 修改结果切割保存的逻辑，目前的逻辑过于简陋，最好通过设备物理内存进行判断，决定切割大小
        """
        self.test_start()  # callback function
        output = None
        labels = None
        losses = []
        self.eval()
        with torch.no_grad():
            index = 1
            for step, (x, y) in enumerate(self.test_loader):
                x = x.to(torch.float32).to(self.device)
                y = y.to(torch.float32).to(self.device)
                loss, model_out = self.compute_loss(x, y, self.criterion)
                model_out = model_out.detach().cpu()
                y = y.detach().cpu()
                losses.append(loss.item())
                output = torch.cat([output, model_out], dim=0) if output is not None else model_out
                labels = torch.cat([labels, y], dim=0) if labels is not None else y
                """
                此处进行预测结果和label的切割保存。为了节约加载时的内存占用，必须将结果切割成多个部分分别保存
                """
                if output.numel() >= 40000000:  # result cut
                    print(output.numel())
                    print(labels.numel())
                    np.save(self.get_model_result_path() + "model_test_output_part{}".format(index), output.cpu().detach().numpy())
                    np.save(self.get_model_result_path() + "model_test_labels_part{}".format(index), labels.cpu().detach().numpy())
                    output, labels = None, None
                    index += 1
            # output = torch.cat(output, dim=0)
            # labels = torch.cat(labels, dim=0)
            if output is not None:
                np.save(self.get_model_result_path() + "model_test_output_part{}".format(index), output.cpu().detach().numpy())
                np.save(self.get_model_result_path() + "model_test_labels_part{}".format(index), labels.cpu().detach().numpy())
                np.save(self.get_model_result_path() + "model_test_loss_part{}".format(index), np.average(losses))
        self.test_end()  # callback function

    def set_criterion(self,
                      criterion):
        self.criterion = criterion

    def plot_losses(self, show=False):
        if self.train_losses is None or self.eval_losses is None:
            raise RuntimeWarning("The model is not trained by internal training method. "
                                 "You could call plot_losses(show=False) after training the model by:"
                                 ">>> model.prepare_data(...)"
                                 ">>> model.train_model(...)."
                                 "Tips: plot_losses(show=False) will not work if you train your model manually"
                                 "but not the above process.")
        if show:
            matplotlib.use("QtAgg")
        else:
            matplotlib.use("Agg")
        plt.suptitle("Model Loss")
        plt.plot(self.train_losses, label="training loss")
        plt.plot(self.eval_losses, label="evalidate loss")
        plt.xlabel("epoch")
        plt.ylabel("criterion loss")
        plt.legend()
        _check_path(self.get_model_result_path())
        plt.savefig(self.get_model_result_path() + "train_eval_losses.png")
        plt.cla()
        if show:
            plt.show(block=True)

    def get_model_result_path(self):
        return root + "/model_result/" + self.flag + "/"

    def _criterion(self, y, label):
        if label.device.type == 'cpu':
            y = y.detach().cpu()
            label = label.detach().cpu()
        elif label.device.type != y.device.type:
            y = y.to(label.device)
        return self.criterion(y, label)

    def compute_loss(self,
                     x: torch.Tensor,
                     label: torch.Tensor,
                     criterion) -> [torch.Tensor, torch.Tensor]:
        """
        An overridable method for different process of loss computation. The default process is simple
        single output computation. This method should only be overrideen if custom loss computation is
        required when training the model by:
            >>> self.prepare_data(...)
            >>> self.train_model(...)

        return: must be a list containing [ loss, model_out ].
        """
        model_out = self(x)
        loss = criterion(model_out.to(self.device), label.to(self.device))
        return [loss, model_out]

    def epoch_start(self):
        """
        A callback function called before every training epoch starting.
        """
        return

    def epoch_end(self):
        """
        A callback function called after every training epoch finished.
        """
        return

    def train_start(self):
        """
        A callback function called before training process starting.
        """
        return

    def train_end(self):
        """
        A callback function called after training process finished.
        """
        return

    def test_start(self):
        """
        A callback function called before testing process starting.
        """
        return

    def test_end(self):
        """
        A callback function called after testing process finished.
        """
        return
