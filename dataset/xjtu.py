from enum import Enum
from threading import Thread
import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

DEFAULT_ROOT = r"/home/fuen/DeepLearningProjects/FaultdiagnosisDataset/XJTU/XJTU-SY_Bearing_Datasets"


class Condition(Enum):
    """
    This enum class contains all the operating conditions in XJTU bearing dataset.
    This enum indicated the operation conditions of XJTUDataset.
    """
    OP_A = "35Hz12kN"
    OP_B = "37.5Hz11kN"
    OP_C = "40Hz10kN"


def get_label(XJTU_path: str,
              OP_condition: Condition,
              bearing_index: int,
              start: int,
              end: int):
    """
    Return fault labels of the bearing with 'index' in the XJTU_path from start to end.

    :param XJTU_path: the absolute ROOT path of XJTU dataset in the disk.
    :param OP_condition: The XJTU_Condition enum.
    :param bearing_index: The target bearing index in [1, 2, 3, 4, 5].
    :param start: the start file index to process. Start from 1.
    :param end: the end file index. If 0, the number of the last Start files will be read. If -1, equals to the last index.
    :return: A int array contains the faults type (multiple faults are possible).
             0 - normal
             1 - inner race fault
             2 - outer race fault
             3 - cage fault
             4 - ball fault
    Sample:
        [[1, 3], [1, 2], [1], [1], [0]] means the target bearing data points' labels are
        [[inner race fault, cage fault], [inner race fault, outer race fault], [inner race fault]...]

    """
    OP_A = [[2], [2], [2], [3], [1, 2]]
    OP_A_fault_points = [73, 35, 108, 82, 36]
    OP_B = [[1], [2], [3], [2], [2]]
    OP_B_fault_points = [455, 47, 127, 31, 121]
    OP_C = [[2], [1, 2, 3, 4], [1], [1], [2]]
    OP_C_fault_points = [2417, 2163, 342, 1420, 7]

    path = XJTU_path + "/" + OP_condition.value + "/" + "Bearing" + str(bearing_index) + "/"
    bearing_index -= 1
    length = len(os.listdir(path))
    start, end = get_index_range(length, start, end)
    labels = [[]] * (end - start)
    label_index = 0
    if OP_condition == Condition.OP_A:
        for i in range(start, end):
            labels[label_index] = OP_A[bearing_index] if i >= OP_A_fault_points[bearing_index] else [0]
            label_index += 1
    elif OP_condition == Condition.OP_B:
        for i in range(start, end):
            labels[label_index] = OP_B[bearing_index] if i >= OP_B_fault_points[bearing_index] else [0]
            label_index += 1
    elif OP_condition == Condition.OP_C:
        for i in range(start, end):
            labels[label_index] = OP_C[bearing_index] if i >= OP_C_fault_points[bearing_index] else [0]
            label_index += 1
    else:
        raise Exception("Unexpected value of OP_Condition:" + OP_condition.value)
    return labels


def read_bearing_data(XJTU_path: str,
                      OP_condition: Condition,
                      bearing_index: int,
                      start: int,
                      end=0):
    """
    Reading all the csv files restored in the XJTU_path/condition/bearingX_n from start to end.

    :param XJTU_path: the absolute ROOT path of XJTU dataset in the disk.
    :param OP_condition: The XJTU_Condition enum.
    :param bearing_index: The target bearing index in [1, 2, 3, 4, 5].
    :param start: the start file index to read. Start from 1.
    :param end: the end file index. If 0, the number of the last Start files will be read. If -1, equals to the last index
    :return: a np.ndarray containing all the data.
    """
    path = XJTU_path + "/" + OP_condition.value + "/" + "Bearing" + str(bearing_index) + "/"
    data_files = os.listdir(path)
    if not (0 <= start <= len(data_files) and -1 <= end <= len(data_files)):
        raise Exception("The start or end token is not expected, start should be from [{} - {}],"
                        "end should be from [{} - {}], but got start = {}, end = {}".format(0, len(data_files),
                                                                                            -1, len(data_files),
                                                                                            start, end))
    start, end = get_index_range(len(data_files), start, end)
    data_frame = [pd.DataFrame] * (end - start)
    data_index = 0
    for i in range(start, end):
        # print("Reading " + str(i) + ".csv ...")
        data_frame[data_index] = pd.read_csv(path + str(i) + ".csv")
        data_index += 1
    return pd.concat(data_frame).to_numpy(np.float32)


def get_index_range(length: int, start: int, end: int):
    if end == 0:
        return length - start + 1, length + 1
    else:
        return start, end + 1 if end != -1 else length + 1


def check_degradation_point(condition: Condition,
                            bearing_index: int):
    import matplotlib.pyplot as plt
    start = 80
    a = read_bearing_data(r"D:\Learning\FaultdiagnosisDataset\XJTU\XJTU-SY_Bearing_Datasets",
                          Condition.OP_A, bearing_index=1, start=start, end=0)
    plt.xticks(ticks=np.linspace(0, a.shape[0], start + 1), labels=np.linspace(1, start + 1, start + 1), rotation=60)
    plt.grid()
    plt.plot(a[:, 0])
    plt.show(block=True)


class XJTU_Dataset(Dataset):
    def __init__(self,
                 XJTU_path: str,
                 op_conditions,
                 bearing_indexes,
                 start_tokens,
                 end_tokens,
                 class_num=5,
                 window_size=10000,
                 step_size=10000):
        r"""
        This class is a XJTU bearing Dataset used for constructing a PyTorch DataLoader.
        .. note::
            The fault points for different bearings are as follows:

            11 - 73     21 - 455    31 - 2417

            12 - 35     22 - 47     32 - 2163

            13 - 108    23 - 127    33 - 342

            14 - 82     24 - 31     34 - 1420

            15 - 36     25 - 121    35 - 7

        :param XJTU_path: The absolute ROOT path (/../XJTU/XJTU-SY_Bearing_Datasets) of XJTU dataset in the disk.
        :param op_conditions: A single value or a list of values of Condition Enum. If a single value, all the bearing_indexes
                              will be processed as this Condition. If a list of values of Condition Enum, the len(op_condition)
                              == len(bearing_indexes) to consist with target bearings on different Conditions.
        :param bearing_indexes: To indicate the target bearings with a two-dim list of integer.
                                The one-dim is used when the 'op_conditions' is a single value. The two-dim is used when
                                it is a list.
        :param start_tokens: The list of start file indexes to read. Start from 1.
        :param end_tokens: The list of end file indexes. If 0, the number of the last Start files will be read.
                           If -1, equals to the last index
        """
        import time
        a = time.time()
        if not isinstance(op_conditions, Condition) and not isinstance(op_conditions, list):
            raise Exception("Unexpected value of op_conditions. It should be a Condition Enum value or the list of it.")
        elif isinstance(op_conditions, list):
            assert len(op_conditions) == len(bearing_indexes)
            # ([a, b], [[1, 2, 3], [2, 3, 5]], [[50, 60, 70], [10, 20, 30]], [[50, 60, 70], [10, 20, 30]])
        self.threads = []
        self.raw_data = []
        self.labels = []
        for con in range(len(op_conditions) if isinstance(op_conditions, list) else 1):
            for bearing_index in range(len(bearing_indexes[con])):
                start = start_tokens[con][bearing_index]
                end = end_tokens[con][bearing_index]
                reader = self.ReaderThread(XJTU_path,
                                           op_conditions[con]
                                           if isinstance(op_conditions, list) else op_conditions,
                                           bearing_indexes[con][bearing_index],
                                           start,
                                           end)
                reader.start()
                self.threads.append(reader)
        for thread in self.threads:
            thread.join()
        for thread in self.threads:
            self.raw_data.append(thread.get_result()[0])
            self.labels += thread.get_result()[1]
        self.raw_data = np.concatenate(self.raw_data)
        self.window_size = window_size
        self.step_size = step_size
        self.class_num = class_num
        b = time.time() - a
        print("读取完毕，处理用时：{:.4f}".format(b))

    def __getitem__(self, index) -> T_co:
        bearing_index = self.step_size * index
        label_skip = self.raw_data.shape[0] // len(self.labels)  # How long a label represents the data stamp.
        label_index = bearing_index // label_skip
        label = np.zeros(self.class_num)
        for i in self.labels[label_index]:
            label[i] = 1
        return self.raw_data[label_index: label_index + self.window_size], label

    def __len__(self):
        return (self.raw_data.shape[0] - self.window_size) // self.step_size

    class ReaderThread(Thread):
        def __init__(self, path, condition, bearing_index, start, end):
            Thread.__init__(self)
            self.raw_data = []
            self.label = []
            self.path = path
            self.condition = condition
            self.bearing_index = bearing_index
            self.startToken = start
            self.endToken = end

        def run(self) -> None:
            print(self.name + " is running for : {} [{}], from {} to {}...".format(self.condition, self.bearing_index,
                                                                                   self.startToken, self.endToken))
            self.raw_data = read_bearing_data(self.path,
                                              self.condition,
                                              self.bearing_index,
                                              self.startToken,
                                              self.endToken)
            self.label = get_label(self.path, self.condition, self.bearing_index, self.startToken, self.endToken)
            print(self.raw_data.shape)
            print(len(self.label))

        def get_result(self):
            return self.raw_data, self.label


if __name__ == '__main__':
    a = get_label(DEFAULT_ROOT, Condition.OP_A, 1, 5, -1)
