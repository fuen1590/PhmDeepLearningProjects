from enum import Enum
from threading import Thread
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class Condition(Enum):
    OP_A = "35Hz12kN"
    OP_B = "37.5Hz11kN"
    OP_C = "40Hz10kN"


def compute_label(data_type,
                  label,
                  length,
                  FPT):
    """
    Compute the different regression type labels of processed dataset.
    :param data_type: data_type: A string demonstrating the type of getting labels.
            "regression" or "piecewise", which the former illustrates the labels decreasing graduated,
            while the latter illustrates the labels begin with 1 and gradually decrease when bearing lying on fault stage.
    :param label: The current time stamp. Start from 1.
    :param length: The whole life of bearing, it also equals to the number of .csv files.
    :param FPT: A parameter used to compute the labels when data_type is "piecewise".
    :return: The result of computed labels, a float num.
    """
    if data_type == "piecewise":
        res = (length - label) / (length - FPT) if label > FPT else 1
    elif data_type == "regression":
        res = (length - label) / length
    else:
        raise Exception("The variable data_type is not expected!")
    return res


def get_labels(XJTU_path: str,
               condition: list,
               bearing_indexes: list,
               start_tokens: list,
               end_tokens: list,
               data_type="piecewise"):
    """
    Get RUL computed by FPT or the whole life or fault_points, and one .csv file corresponding to one label.
    This function can extract the whole given bearings labels and put them into a 3-dimension list.

    Noted:
        FPTs is gotten from conference paper, and the fourth bearing in first condition
        and the second bearing in third condition are not provided in the paper,
        thus both of them can not use when using "piecewise" type labels.
    :param XJTU_path: The root path of XJTU dataset in the disk. i.e. "../XJTU/XJTU-SY_Bearing_Datasets"
    :param condition: A list of Condition enum. i.e. [Condition.OP_A, Condition.OP_B, Condition.OP_C]
    :param bearing_indexes: A 2-dimension list corresponding to variable condition. i.e. [[1,2,3], [1,2,3,4], [1,3,4]]
    :param start_tokens: It illustrates the start indexes of the .csv file in the given bearing to read,
            so it has the same shape with parameter bearing_indexes, and the elements in it start from 1.
    :param end_tokens: Like start_tokens.
            If 0, the number of the last Start files will be read, if -1, equals to the last index.
    :param data_type: A string illustrating the type of getting data. "regression" or "piecewise" or "classification".
            The first illustrates the labels decreasing graduated,
            the second illustrates the labels begin with 1 and gradually decrease when bearing lying on fault stage,
            and the last illustrates the class labels.
    :return: A 3-dimension list storing RUL value of every .csv file.
            i.e. labels[a][b][c], a in [0,1,2] is number of conditions from 0,
            b in [0, 1, 2, 3, 4] is the number of 5 bearings from 0, c is the name of .csv files minus 1.
    """
    FPTs = [[77, 31, 58, None, 34], [454, 46, 314, 30, 120], [2376, None, 340, 1416, 6]]

    OP_A = [[2], [2], [2], [3], [1, 2]]
    OP_A_fault_points = [73, 35, 108, 82, 36]
    OP_B = [[1], [2], [3], [2], [2]]
    OP_B_fault_points = [455, 47, 127, 31, 121]
    OP_C = [[2], [1, 2, 3, 4], [1], [1], [2]]
    OP_C_fault_points = [2417, 2163, 342, 1420, 7]

    labels = [[[], [], [], [], []],
              [[], [], [], [], []],
              [[], [], [], [], []]]

    for con in range(len(condition)):  # con 指参数提供的每个工况的顺序号
        for bearing_index in range(len(bearing_indexes[con])):  # bearing_index 指参数提供的每个工况下的每个轴承顺序号
            op_condition = condition[con]
            bearing_num = bearing_indexes[con][bearing_index]
            bearing_path = os.path.join(XJTU_path, op_condition.value, "Bearing" + str(bearing_num))
            length = len(os.listdir(bearing_path))
            start, end = get_index_range(length, start_tokens[con][bearing_index], end_tokens[con][bearing_index])
            bearing_num -= 1
            if op_condition == Condition.OP_A:
                FPT = FPTs[0][bearing_num]
                for i in range(start, end):
                    if data_type == "classification":
                        label = OP_A[bearing_index] if i >= OP_A_fault_points[bearing_index] else [0]
                    else:
                        label = compute_label(data_type, i, length, FPT)
                    labels[0][bearing_num].append(label)
            elif op_condition == Condition.OP_B:
                FPT = FPTs[1][bearing_num]
                for i in range(start, end):
                    if data_type == "classification":
                        label = OP_B[bearing_index] if i >= OP_B_fault_points[bearing_index] else [0]
                    else:
                        label = compute_label(data_type, i, length, FPT)
                    labels[1][bearing_num].append(label)
            elif op_condition == Condition.OP_C:
                FPT = FPTs[2][bearing_num]
                for i in range(start, end):
                    if data_type == "classification":
                        label = OP_C[bearing_index] if i >= OP_C_fault_points[bearing_index] else [0]
                    else:
                        label = compute_label(data_type, i, length, FPT)
                    labels[2][bearing_num].append(label)
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


def sliding(raw_data,
            window_size=8192,
            sliding_steps=1024):
    """
    Get the result of sliding from raw data, and the result is used to regression.
    :param raw_data: In this task, it should be a np.ndarray containing shape: [m * 32768, 2]
    :param window_size: The sliding window size.
    :param sliding_steps: The stride of sliding task.
    :return A np.ndarray containing shape: [m * 25, 8192, 2]
    """
    length = 32768
    file_nums = raw_data.shape[0] // length
    sliding_data = []
    for num in range(file_nums):
        raw_data_single = raw_data[num * length:(num + 1) * length, :]
        for start in range(0, raw_data_single.shape[0] - window_size + 1, sliding_steps):
            sliding_data.append(raw_data_single[start:start + window_size, :])
    sliding_data = np.array(sliding_data, dtype=np.float32)
    return sliding_data


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
                 data_type="piecewise",
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
        :param data_type: A string demonstrating the type of getting labels.
                i.e. "regression" or "piecewise" or "classification".
                The former illustrates the labels decreasing graduated,
                the second illustrates the labels begin with 1 and gradually decrease when bearing lying on fault stage,
                and the last is the classification labels.
        :param class_num: The number of classes in classification task.
        :param window_size: The size of sliding window when getting data.
                The size is set to 8192 in the conference paper using regression data.
        :param step_size: The size of sliding step when getting data.
                The size is set to 1024 in the conference paper using regression data.
        :return: A XJTU_Dataset type data which could be loaded by DataLoader.
        """
        if isinstance(condition, list):
            assert len(condition) == len(bearing_indexes)
        else:
            raise Exception("Unexpected value of op_conditions. It should be a Condition Enum value or the list of it.")
        if data_type not in ['regression', 'piecewise', 'classification']:
            raise Exception("Parameter data_type is not expected!")
        self.data_type = data_type
        if self.data_type != "classification":
            self.Samples, self.Labels = [], []
            self.reg_labels = get_labels(XJTU_path,
                                         [Condition.OP_A, Condition.OP_B, Condition.OP_C],
                                         [[1, 2, 3, 5], [1, 2, 3, 4, 5], [1, 3, 4, 5]],
                                         [[1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1]],
                                         [[-1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1]],
                                         self.data_type)
            for con in range(len(condition)):
                for bearing_index in range(len(bearing_indexes[con])):
                    print(f"Process: condition:{condition[con].value}, bearing:{bearing_indexes[con][bearing_index]}")
                    start = start_tokens[con][bearing_index]
                    end = end_tokens[con][bearing_index]
                    raw_data_bearing = read_bearing_data(XJTU_path, condition[con], bearing_indexes[con][bearing_index],
                                                         start, end)
                    samples_bearing = sliding(raw_data_bearing, window_size, sliding_steps=step_size)
                    if condition[con] == Condition.OP_A:
                        labels_bearing = np.array(self.reg_labels[0][bearing_indexes[con][bearing_index] - 1],
                                                  dtype=np.float32).reshape(-1, 1)
                    elif condition[con] == Condition.OP_B:
                        labels_bearing = np.array(self.reg_labels[1][bearing_indexes[con][bearing_index] - 1],
                                                  dtype=np.float32).reshape(-1, 1)
                    elif condition[con] == Condition.OP_C:
                        labels_bearing = np.array(self.reg_labels[2][bearing_indexes[con][bearing_index] - 1],
                                                  dtype=np.float32).reshape(-1, 1)
                    else:
                        raise Exception("Condition is not the expected when getting labels!")
                    labels_bearing_copy = labels_bearing.copy()
                    # The number of samples generated by one .csv file.
                    samples_num = (32768 - window_size) // step_size + 1
                    for copy_num in range(samples_num - 1):
                        labels_bearing = np.concatenate((labels_bearing, labels_bearing_copy), axis=1)
                    labels_bearing = labels_bearing.reshape(-1, 1)
                    self.Samples = samples_bearing if isinstance(self.Samples, list) else np.concatenate(
                        (self.Samples, samples_bearing), axis=0)
                    self.Labels = labels_bearing if isinstance(self.Labels, list) else np.concatenate(
                        (self.Labels, labels_bearing), axis=0)
        else:
            self.threads = []
            self.raw_data = []
            self.cls_labels = []
            for con in range(len(condition)):
                for bearing_index in range(len(bearing_indexes[con])):
                    start = start_tokens[con][bearing_index]
                    end = end_tokens[con][bearing_index]
                    reader = self.ClsReaderThread(XJTU_path,
                                                  condition[con],
                                                  bearing_indexes[con][bearing_index],
                                                  start,
                                                  end)
                    reader.start()
                    self.threads.append(reader)
            for thread in self.threads:
                thread.join()
            for thread in self.threads:
                self.raw_data.append(thread.get_result()[0])
                self.cls_labels += thread.get_result()[1]
            self.raw_data = np.concatenate(self.raw_data)
            self.window_size = window_size
            self.step_size = step_size
            self.class_num = class_num

    def __getitem__(self, index):
        if self.data_type != "classification":
            return self.Samples[index], self.Labels[index]
        else:
            bearing_index = self.step_size * index
            label_skip = self.raw_data.shape[0] // len(self.cls_labels)  # How long a label represents the data stamp.
            label_index = bearing_index // label_skip
            label = np.zeros(self.class_num)
            for i in self.cls_labels[label_index]:
                label[i] = 1
            return self.raw_data[label_index: label_index + self.window_size], label

    def __len__(self):
        if self.data_type != "classification":
            return self.Samples.shape[0]
        else:
            return (self.raw_data.shape[0] - self.window_size) // self.step_size + 1

    class ClsReaderThread(Thread):
        def __init__(self, path, condition, bearing_index, start, end):
            Thread.__init__(self)
            self.cls_raw_data = []
            self.cls_label = []
            self.path = path
            self.condition = condition
            self.bearing_index = bearing_index
            self.startToken = start
            self.endToken = end

        def run(self) -> None:
            print(self.name + " is running for : {} [{}], from {} to {}...".format(self.condition, self.bearing_index,
                                                                                   self.startToken, self.endToken))
            self.cls_raw_data = read_bearing_data(self.path,
                                                  self.condition,
                                                  self.bearing_index,
                                                  self.startToken,
                                                  self.endToken)
            self.cls_label = get_labels(self.path, [self.condition], [[self.bearing_index]], [[self.startToken]],
                                        [[self.endToken]], "classification")
            if self.condition == Condition.OP_A:
                self.cls_label = self.cls_label[0][self.bearing_index - 1]
            elif self.condition == Condition.OP_B:
                self.cls_label = self.cls_label[1][self.bearing_index - 1]
            elif self.condition == Condition.OP_C:
                self.cls_label = self.cls_label[2][self.bearing_index - 1]
            else:
                raise Exception("Thread self.condition is not expected!")
            print(self.cls_raw_data.shape)
            print(len(self.cls_label))

        def get_result(self):
            return self.cls_raw_data, self.cls_label


if __name__ == '__main__':
    test_path = r"/home/dell/chuyuxin/Recurrent/re1/test_files"
    DEFAULT_ROOT = r"/home/dell/chuyuxin/Recurrent/re1/XJTU/XJTU-SY_Bearing_Datasets"
    conditions = [Condition.OP_A, Condition.OP_B]
    train_bearings = [[1, 2, 3], [1, 2, 3, 4]]
    train_start = [[1, 1, 1], [1, 1, 1, 1]]
    train_end = [[-1, -1, -1], [-1, -1, -1, -1]]
    test_bearings = [[5], [5]]
    test_start = [[1], [1]]
    test_end = [[-1], [-1]]
    window_size, step_size = 8192, 1024
    # train_set = XJTU_Dataset(DEFAULT_ROOT, conditions, train_bearings, train_start, train_end, 'classification')
    test_set = XJTU_Dataset(DEFAULT_ROOT, conditions, test_bearings, test_start, test_end, "piecewise",
                            window_size=window_size, step_size=step_size)
    test_set_reg = XJTU_Dataset(DEFAULT_ROOT, conditions, test_bearings, test_start, test_end, "regression",
                                window_size=window_size, step_size=step_size)
