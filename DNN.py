import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pandas as pd
import os
# -------------------------------------------------
# Data Sampler for Imbalance dataset
# Source: ufoym@CVTE(https://github.com/ufoym/imbalanced-dataset-sampler/blob/
# master/torchsampler/imbalanced.py)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices
       for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which
        takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        # define custom callback

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
# -------------------------------------------------
# Cauculate Information Entropy Loss
# Source:bravotty@SCUT(https://github.com/bravotty/Information-entropy-loss-pytorch/blob/master/entropy_loss_pytorch.py)


def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)


class Entropy(nn.Module):
    # https://github.com/bravotty/Information-entropy-loss-pytorch/blob/master/entropy_loss_pytorch.py
    def __init__(self, reduce=True, eps=1e-16):
        super().__init__()
        """
        the definition of Entropy is - '\'sum p(xi) log (p(xi))
        """
        self.eps = eps
        self.reduce = reduce

    def forward(self, input: torch.Tensor):
        assert input.shape.__len__() >= 2
        b, _, *s = input.shape
        input = F.softmax(input, dim=1)
        assert simplex(input)
        e = input * (input + self.eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self.reduce:
            return e.mean()
        return e
# -------------------------------------------------------


def preprocess(address, seperate_year):
    '''
    用來把row data 整理成沒有na值並經過normalize(L2-norm)的data
    input: row data 地址，想要切割成testing data的年份
    output: 整理好的data，train_data, train_data_y, test_data, test_data_y
    '''
    origin_data = pd.read_csv(address, encoding="utf-8")
    origin_data = origin_data.dropna()  # 去除na值

    # 把資料依照年份分成train和test data
    train_data = origin_data.loc[~(origin_data['年'] == seperate_year)].copy()
    test_data = origin_data.loc[origin_data['年'] == seperate_year].copy()

    # 拿出y
    train_data = train_data.drop(['公司', '年', '月'], axis=1)
    # 0是審計失敗，也就是y
    train_data_1 = train_data.loc[train_data['審計失敗'] == 1].values.astype(
            'float64')
    train_data_0 = train_data.loc[train_data['審計失敗'] == 0].values.astype(
            'float64')
    test_data = test_data.drop(['公司', '年', '月'], axis=1).values.astype(
            'float64')
    return train_data_1, train_data_0, test_data
# ----------------------------------------------------
# train dataset


class train_set(Dataset):

    def __init__(self, train_data):
        self.train_data_torch = torch.FloatTensor(train_data)

    def __getitem__(self, index):

        return self.train_data_torch[index][1:], int(
                self.train_data_torch[index][0].item())
        # item可以拿出只有單一值得tensor的值

    def __len__(self):
        return self.train_data_torch.shape[0]
# ----------------------------------------------------
# test dataset


class test_set(Dataset):

    def __init__(self, test_data):
        self.test_data_torch = torch.FloatTensor(test_data)
        self.test_data_torch_x = self.test_data_torch[:, 1:].detach().clone()
        self.test_data_torch_y = self.test_data_torch[:, 0].detach().clone()

    def __getitem__(self, index):
        return self.test_data_torch_x[index],\
                self.test_data_torch_y[index].item(), index
        # item可以拿出只有單一值得tensor的值

    def __len__(self):
        return self.test_data_torch.shape[0]

# ----------------------------------------------------
# DNN Model


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5679, 2500),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=2500),

            nn.Linear(2500, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1000),

            nn.Linear(1000, 2),
        )

    def forward(self, x):

        encoded = self.network(x)
        return encoded
# --------------------------------------------------------
# training


def training(train_loader, test_loader, turn, first_layer, test_data_batch,
             EPOCH, lr, weight_decay, log_save_loc, model_save_loc,
             entropy_loss_rate):  # epoch決定要幾個epooch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")

    model = DNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    log_save_location = log_save_loc + '/train_{}_log.txt'.format(turn)
    text_file = open(log_save_location, "w")
    entropy_loss = Entropy()
    loss_function = nn.CrossEntropyLoss()

    def validation(test_loader, model, device):
        model.eval()
        psudo_label_index = []
        for batch_idx, (data, label, index) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            max_value, pred_outcome = torch.max(output, 1)
            for i in range(label.shape[0]):
                if pred_outcome[i] == 1:
                    psudo_label_index.append(index[i])
        return psudo_label_index

    for epoch in range(EPOCH):
        data_length = 0
        model.train()  # 有些model功能只有train要用，像是dropout，加了這個就是讓他了解要使用
        train_loss = 0
        correct = 0
        confusing_matrix = [[0, 0, 0.0], [0, 0, 0.0], [0.0, 0.0, 0.0]]
        print("round" + str(turn) + " epoch: " + str(epoch) + " Start!")
        text_file.write("round" + str(turn) +
                        " epoch: " + str(epoch) + " Start!\n")
        for batch_idx, (data, label) in enumerate(train_loader):
            model.train()
            data, label = data.to(device), label.to(device)  # 把資料丟進GPU

            optimizer.zero_grad()
            output = model(data)
            max_value, pred_outcome = torch.max(output, 1)

            # Performance on Training data
            for i in range(label.shape[0]):
                data_length += 1
                if pred_outcome[i] == label[i]:
                    correct += 1
                    if label[i] == 1:
                        confusing_matrix[0][0] += 1
                    elif label[i] == 0:
                        confusing_matrix[1][1] += 1
                else:
                    if label[i] == 1:
                        confusing_matrix[0][1] += 1
                    elif label[i] == 0:
                        confusing_matrix[1][0] += 1

            # test_data_batch = test_data.shape[0]//batch_size
            # Information Entropy took too much time
            # So I only randon check one batch in test data for Entropy
            entro_loss = 0
            entropy_loss_rate = entropy_loss_rate
            idx = np.random.randint(low=0,
                                    high=test_data_batch, size=1).item(0)
            for batch_idx1, (data1, label1, index1) in enumerate(test_loader):
                if batch_idx1 == idx:
                    data1, label1 = data1.to(device), label1.to(device)
                    out = model(data1)
                    entro_loss += entropy_loss(out)

            loss = loss_function(output, label) + entropy_loss_rate*entro_loss

            train_loss += loss.item()  # 用來記錄數字
            loss.backward()
            optimizer.step()

        # print performance of training data
        confusing_matrix[0][2] = (confusing_matrix[0][0] /
                                  (confusing_matrix[0][0] +
                                  confusing_matrix[0][1]))*100
        confusing_matrix[1][2] = (confusing_matrix[1][1] /
                                  (confusing_matrix[1][0] +
                                  confusing_matrix[1][1]))*100
        confusing_matrix[2][0] = (confusing_matrix[0][0] /
                                  (confusing_matrix[0][0] +
                                  confusing_matrix[1][0]))*100
        confusing_matrix[2][1] = (confusing_matrix[1][1] /
                                  (confusing_matrix[0][1] +
                                  confusing_matrix[1][1]))*100
        confusing_matrix = pd.DataFrame(confusing_matrix, columns=["1 predict",
                                        "0 predict", "accuracy %"],
                                        index=["1 actual", "0 actual",
                                        "accuracy%"])
        print(confusing_matrix)

        # print accuracy
        print("data lenth = ", data_length)
        print("accuracy", correct/data_length)
        print("train_loss:", train_loss)
        if ((epoch + 1) % 1 == 0):
            text_file.write("data lenth = {}\n".format(data_length))
            text_file.write("accuracy:　{}\n".format(correct/data_length))
            text_file.write("train_loss: {}\n".format(train_loss))
            text_file.write(confusing_matrix.to_string())

        validation(test_loader, model, device)

        print("epoch: " + str(epoch) + " End\n")
        text_file.write("\nepoch: " + str(epoch) + " End\n")

        # save model of last epoch
        if epoch == EPOCH - 1:
            model_save = model_save_loc + '/model_{}_{}.pkl'.format(turn, epoch)
            torch.save(model.state_dict(), model_save)
    psudo = validation(test_loader, model, device)
    text_file.close()
    return psudo

# --------------------------------------------------
# testing part


def testing(test_loader, model_num, vote_cut, model_loc):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")

    answer_combine = []
    test_data_y = []

    for batch_idx, (data, label, index) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        for j in label.tolist():
            test_data_y.append(j)
    for i in range(25, 50):
        answer = []
        model = DNN()
        model = model.to(device)
        print(i, "model start predicting:")
        model_location = model_loc + '/model_{}_{}.pkl'.format(i, model_num)
        model.load_state_dict(torch.load(model_location))
        model.eval()
        for batch_idx, (data, label, index) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            max_value, pred_outcome = torch.max(output, 1)
            for j in pred_outcome.tolist():
                answer.append(j)
        answer_combine.append(answer)
    answer_combine = np.array(answer_combine).T
    print("answer_combine shape: ", answer_combine.shape)

    answer_vote = np.sum(answer_combine, axis=1)
    vote = []
    print("vote time")
    for i in answer_vote:
        if i > vote_cut:
            vote.append(1)
        else:
            vote.append(0)
    return vote
# -----------------------------------------------


class Dnn_run():
    def __init__(self, cofing=None):
        self.path = os.path.dirname(os.path.realpath(__file__)).replace('\\',
                                                                        '/')
        if cofing:
            self.cofing = cofing
        else:
            self.cofing = {
                'Seed': 777,
                'EPOCH': 20,
                'Semi_training_round': 50,
                'batch_size': 200,
                'lr': 0.0001,
                'weight_decay': 0,
                'Information_Entropy_loss_rate': 0.05,
                'model_num_for_testing': 19,
                'raw_data_loc': os.path.join(self.path, 'data/raw_data.xlsx')
                .replace('\\', '/'),
                'log_file_save_in': os.path.join(self.path, 'log')
                .replace('\\', '/'),
                'model_save_in': os.path.join(self.path, 'model')
                .replace('\\', '/'),
                'past_vote_thread': 24,
                'testing_year': 2019,
                'output_loc': os.path.join(self.path, "結果.xlsm")
                .replace('\\', '/')
            }
        self.psudo = None
        self.train_data = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.dataset_batch = 0
        self.train_data_shape = None

    def load_data(self):
        train_data_1, train_data_0, self.test_data = preprocess(self.cofing
                                                           ['raw_data_loc'],
                                                           self.cofing
                                                           ['testing_year'])

        self.test_dataset = test_set(self.test_data)
        self.test_data_batch = self.test_data.shape[0]//(self.cofing['batch_size']
                                                    + 1)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.cofing
                                                       ['batch_size'],
                                                       shuffle=True)
        self.train_data = np.concatenate((train_data_1, train_data_0), axis=0)
        self.train_data_shape = self.train_data.shape
        train_dataset = train_set(self.train_data)
        imba = ImbalancedDatasetSampler(train_dataset, num_samples=30000)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        sampler=imba,
                                                        batch_size=100,
                                                        shuffle=False)

    def run(self):
        for i in range(0, self.cofing['Semi_training_round']):
            pusdo = training(self.train_loader,
                             self.test_loader,
                             turn=i,
                             first_layer=self.train_data_shape[1],
                             test_data_batch=self.test_data_batch,
                             EPOCH=self.cofing['EPOCH'],
                             lr=self.cofing['lr'],
                             weight_decay=self.cofing['weight_decay'],
                             log_save_loc=self.cofing['log_file_save_in'],
                             model_save_loc=self.cofing['model_save_in'],
                             entropy_loss_rate=self.cofing[
                                 'Information_Entropy_loss_rate'])
            if i == self.cofing['Semi_training_round']-1:
                break
            train_ = np.concatenate((self.train_data, self.test_data[pusdo]),
                                    axis=0)
            train_s = train_set(train_)
            imba = ImbalancedDatasetSampler(train_s, num_samples=30000)
            self.train_loader = torch.utils.data.DataLoader(dataset=train_s,
                                                            sampler=imba,
                                                            batch_size=100,
                                                            shuffle=False)

    def test(self):
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.cofing[
                                                           'batch_size'],
                                                       shuffle=False)

        vote = testing(self.test_loader,
                       model_num=self.cofing['model_num_for_testing'],
                       vote_cut=self.cofing['past_vote_thread'],
                       model_loc=self.cofing['model_save_in'])

        origin_data = pd.read_csv(self.cofing['raw_data_loc'],
                                  encoding="utf-8")
        test_data = origin_data.loc[origin_data['年'] ==
                                    self.cofing['testing_year'],
                                    ['公司', '年', '月']].copy()
        test_data['預測結果'] = vote
        print(self.cofing['output_loc'])
        test_data.to_excel(self.cofing['output_loc'],
                           float_format='%g',
                           encoding="utf-8",
                           index=False)
