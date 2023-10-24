from enum import Enum
from threading import Thread
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Condition(Enum):
    OP_A = "35Hz12kN"
    OP_B = "37.5Hz11kN"
    OP_C = "40Hz10kN"


class LabelsType(Enum):
    TYPE_C = "classification"
    TYPE_P = "piecewise"
    TYPE_R = "regression"


def get_labels(XJTU_path: str,
               condition: Condition,
               bearing_index: int,
               start: int,
               end: int,
               labels_type: LabelsType = LabelsType.TYPE_P):
    """
    Get RUL computed by FPT or the whole life or fault_points, and one .csv file corresponding to one label.
    This function can extract the whole given bearings labels and put them into a 3-dimension list.

    Noted:
        FPTs is gotten from conference paper, and the fourth bearing in first condition
        and the second bearing in third condition are not provided in the paper,
        thus both of them can not use when using "piecewise" type labels.
    :param XJTU_path: The root path of XJTU dataset in the disk. i.e. "../XJTU/XJTU-SY_Bearing_Datasets"
    :param condition: A list of Condition enum. i.e. [Condition.OP_A, Condition.OP_B, Condition.OP_C]
    :param bearing_index: A 2-dimension list corresponding to variable condition. i.e. [[1,2,3], [1,2,3,4], [1,3,4]]
    :param start: It illustrates the start indexes of the .csv file in the given bearing to read,
            so it has the same shape with parameter bearing_indexes, and the elements in it start from 1.
    :param end: Like start_tokens.
            If 0, the number of the last Start files will be read, if -1, equals to the last index.
    :param labels_type: A LabelsType illustrating the type of getting data.
    :return: A 3-dimension list storing RUL value of every .csv file.
            i.e. labels[a][b][c], a in [0, 1, 2] is number of conditions from 0,
            b in [0, 1, 2, 3, 4] is the number of 5 bearings from 0, c is the name of .csv files minus 1.
    """
    FPTs = [[77, 31, 58, None, 34], [454, 46, 314, 30, 120], [2376, None, 340, 1416, 6]]

    OP_A = [[2], [2], [2], [3], [1, 2]]
    OP_A_fault_points = [73, 35, 108, 82, 36]
    OP_B = [[1], [2], [3], [2], [2]]
    OP_B_fault_points = [455, 47, 127, 31, 121]
    OP_C = [[2], [1, 2, 3, 4], [1], [1], [2]]
    OP_C_fault_points = [2417, 2163, 342, 1420, 7]

    path = XJTU_path + "/" + condition.value + "/" + "Bearing" + str(bearing_index) + "/"
    bearing_index -= 1
    length = len(os.listdir(path))
    start, end = get_index_range(length, start, end)
    labels = [[]] * (end - start)
    label_index = 0
    if condition == Condition.OP_A:
        FPT = FPTs[0][bearing_index]
        for i in range(start, end):
            if labels_type == LabelsType.TYPE_C:
                labels[label_index] = OP_A[bearing_index] if i >= OP_A_fault_points[bearing_index] else [0]
            elif labels_type == LabelsType.TYPE_P:
                labels[label_index] = [(length - i) / (length - FPT) if i > FPT else 1]
            else:
                labels[label_index] = [(length - i) / length]
            label_index += 1
    elif condition == Condition.OP_B:
        FPT = FPTs[1][bearing_index]
        for i in range(start, end):
            if labels_type == LabelsType.TYPE_C:
                labels[label_index] = OP_B[bearing_index] if i >= OP_B_fault_points[bearing_index] else [0]
            elif labels_type == LabelsType.TYPE_P:
                labels[label_index] = [(length - i) / (length - FPT) if i > FPT else 1]
            else:
                labels[label_index] = [(length - i) / length]
            label_index += 1
    elif condition == Condition.OP_C:
        FPT = FPTs[2][bearing_index]
        for i in range(start, end):
            if labels_type == LabelsType.TYPE_C:
                labels[label_index] = OP_C[bearing_index] if i >= OP_C_fault_points[bearing_index] else [0]
            elif labels_type == LabelsType.TYPE_P:
                labels[label_index] = [(length - i) / (length - FPT) if i > FPT else 1]
            else:
                labels[label_index] = [(length - i) / length]
            label_index += 1
    else:
        raise Exception("Unexpected value of OP_Condition:" + condition.value)
    return labels


def get_index_range(length: int, start: int, end: int):
    if end == 0:
        return length - start + 1, length + 1
    else:
        return start, end + 1 if end != -1 else length + 1


def read_bearing_data(XJTU_path: str,
                      OP_condition: Condition,
                      bearing_index: int,
                      start=1,
                      end=-1):
    """
    Reading all the csv files restored in the XJTU_path/condition/bearingX_n from start to end.
    This function get the data from one bearing in one condition,
    so a loop is needed if you want to get data from different bearings.

    :param XJTU_path: The absolute ROOT path of XJTU dataset in the disk.
    :param OP_condition: The XJTU_Condition enum.
    :param bearing_index: The target bearing index in [1, 2, 3, 4, 5].
    :param start: The start file index to read. Start from 1.
    :param end: The end file index. If 0, the number of the last Start files will be read. If -1, equals to the last index
    :return  A np.ndarray containing all the data of the given bearings. i.e. (1376256, 2)
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
                 condition: list,
                 bearing_indexes: list,
                 start_tokens: list,
                 end_tokens: list,
                 labels_type=LabelsType.TYPE_P,
                 class_num=5,
                 window_size=10000,
                 step_size=10000):
        """
        Get the samples and labels from XJTU dataset.
        :param XJTU_path: The absolute root path of XJTU dataset file.
                i.e. "../XJTU/XJTU-SY_Bearing_Datasets"
        :param condition: A list of bearing operation conditions which need to be processed.
                i.e. [Condition.OP_A, Condition.OP_B, Condition.OP_C], [Condition.OP_A]
        :param bearing_indexes: A 2-dimension list of bearing corresponding to variable condition.
                i.e. [[1, 2, 3], [1, 2, 3, 4], [1, 3, 4]], [[5]]
        :param start_tokens: The list of start file indexes to read. Start from 1.
        :param end_tokens: The list of end file indexes. If 0, the number of the last Start files will be read.
                If -1, equals to the last index.
        :param labels_type: A LabelsType illustrating the type of getting data.
                LabelsType.TYPE_R, TYPE_P and TYPE_C represent "regression", "piecewise" and "classification", respectively.
                The first illustrates the labels decreasing graduated,
                the second illustrates the labels begin with 1 and gradually decrease when bearing lying on fault stage,
                and the last illustrates the class labels.
        :param class_num: The number of classes in classification task.
        :param window_size: The size of sliding window when getting data.
                The size is set to 8192 in the conference paper using regression data.
        :param step_size: The size of sliding step when getting data.
                The size is set to 1024 in the conference paper using regression data.
        :return: A XJTU_Dataset type data which could be loaded by DataLoader.
        """
        import time
        time0 = time.time()
        if isinstance(condition, list):
            assert len(condition) == len(bearing_indexes)
        else:
            raise Exception("Unexpected value of op_conditions. It should be a Condition Enum value or the list of it.")
        if not isinstance(labels_type, LabelsType):
            raise Exception("Parameter labels_type is not expected!")
        else:
            print(f"The {labels_type.value} labels will be extracted.")
        self.labels_type = labels_type
        self.threads = []
        self.raw_data = []
        self.labels = []
        for con in range(len(condition)):
            for bearing_index in range(len(bearing_indexes[con])):
                start = start_tokens[con][bearing_index]
                end = end_tokens[con][bearing_index]
                reader = self.ReaderThread(XJTU_path,
                                           condition[con],
                                           bearing_indexes[con][bearing_index],
                                           start,
                                           end,
                                           self.labels_type)
                reader.start()
                self.threads.append(reader)
        for thread in self.threads:
            thread.join()
        for thread in self.threads:
            self.raw_data.append(thread.get_result()[0])
            self.labels.extend(thread.get_result()[1])
        self.raw_data = np.concatenate(self.raw_data)
        self.window_size = window_size
        self.step_size = step_size
        self.class_num = class_num
        print("Finishing reading data, processing time：{:.4f}s".format(time.time()-time0))

    def __getitem__(self, index):
        bearing_index = self.step_size * index
        label_skip = self.raw_data.shape[0] // len(self.labels)  # How long a label represents the data stamp.
        label_index = bearing_index // label_skip
        if self.labels_type == LabelsType.TYPE_C:
            label = np.zeros(self.class_num)
            for i in self.labels[label_index]:
                label[i] = 1
            return self.raw_data[bearing_index: bearing_index + self.window_size], label
        else:
            return self.raw_data[bearing_index: bearing_index + self.window_size], \
                self.labels[(bearing_index + self.window_size) // label_skip]

    def __len__(self):
        return (self.raw_data.shape[0] - self.window_size) // self.step_size + 1

    class ReaderThread(Thread):
        def __init__(self, path, condition, bearing_index, start, end, labels_type):
            Thread.__init__(self)
            self.raw_data = []
            self.label = []
            self.path = path
            self.condition = condition
            self.bearing_index = bearing_index
            self.startToken = start
            self.endToken = end
            self.labels_type = labels_type

        def run(self) -> None:
            print(self.name + " is running for : {} [{}], from {} to {}...".format(self.condition, self.bearing_index,
                                                                                   self.startToken, self.endToken))
            self.raw_data = read_bearing_data(self.path,
                                              self.condition,
                                              self.bearing_index,
                                              self.startToken,
                                              self.endToken)
            self.label = get_labels(self.path, self.condition, self.bearing_index, self.startToken,
                                    self.endToken, self.labels_type)
            print(self.raw_data.shape)
            print(len(self.label))

        def get_result(self):
            return self.raw_data, self.label


if __name__ == '__main__':
    test_path = r"/home/dell/chuyuxin/Recurrent/re1/test_files"
    DEFAULT_ROOT = r"/home/dell/chuyuxin/Recurrent/re1/XJTU/XJTU-SY_Bearing_Datasets"
    conditions = [Condition.OP_A, Condition.OP_B]
    train_bearings = [[1, 2, 3, 5], [1, 2, 3, 4]]
    train_start = [[1, 1, 1, 1], [1, 1, 1, 1]]
    train_end = [[-1, -1, -1, -1], [-1, -1, -1, -1]]
    test_conditions = [Condition.OP_A, Condition.OP_C]
    test_bearings = [[5], [1]]
    test_start = [[1], [1]]
    test_end = [[-1], [-1]]
    window_size, step_size = 8192, 1024
    # train_set = XJTU_Dataset(DEFAULT_ROOT, conditions, train_bearings, train_start, train_end, LabelsType.TYPE_C)
    train_set = XJTU_Dataset(DEFAULT_ROOT, conditions, train_bearings, train_start, train_end, LabelsType.TYPE_R,
                             window_size=window_size, step_size=step_size)
    # test_set = XJTU_Dataset(DEFAULT_ROOT, conditions, test_bearings, test_start, test_end, LabelsType.TYPE_C,
    #                         window_size=window_size, step_size=step_size)
    # test_set = XJTU_Dataset(DEFAULT_ROOT, test_conditions, test_bearings, test_start, test_end, LabelsType.TYPE_P,
    #                         window_size=window_size, step_size=step_size)
    # labels = get_labels(DEFAULT_ROOT, Condition.OP_A, 1, 1, -1, LabelsType.TYPE_C)
