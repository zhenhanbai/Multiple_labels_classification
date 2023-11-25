import os
import torch
from tqdm import tqdm
import pickle as pkl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setSeed(seed=42)


vocab_file = "vocab.pkl"
word_to_id = pkl.load(open(vocab_file, 'rb'))  #加载词典


def tokenize_textCNN(s):
    max_size = 200
    ts = [w for i, w in enumerate(s) if i < max_size]
    ids = [word_to_id[w] if w in word_to_id.keys() else word_to_id['[UNK]'] for w in ts]  # 根据词典，将字符列表转换为id列表
    ids += [0 for _ in range(max_size-len(ts))]  # 若id列表达不到最大长度，则补0
    return ids

class MyDataV2(Dataset):  # 继承Dataset
    def __init__(self, tokenize_fun, filename):
        self.filename = filename  # 要加载的数据文件名
        self.tokenize_function = tokenize_fun  # 实例化时需传入分词器函数
        print("Loading dataset "+ self.filename +" ...")
        self.data, self.labels = self.load_data()  # 得到分词后的id序列和标签
    #读取文件，得到分词后的id序列和标签，返回的都是tensor类型的数据
    def load_data(self):
        labels = []
        data = []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading data', colour="green"):
                fields  = line.strip().split('\t')
                if len(fields) != 3 :
                    continue
                labels.append([float(1) if i == "1" else float(0) for i in fields[2].split(",")])  #标签转换为序号
                data.append(self.tokenize_function(fields[0]+fields[1]))  # 样本为词id序列
        f.close()
        return torch.tensor(data), torch.tensor(labels)
    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.data[index], self.labels[index]

class MyData(Dataset):  # 继承Dataset
    def __init__(self, tokenize_fun, filename):
        self.filename = filename  # 要加载的数据文件名
        self.tokenize_function = tokenize_fun  # 实例化时需传入分词器函数
        print("Loading dataset "+ self.filename +" ...")
        self.input_ids, self.mask, self.labels, self.data_init = self.load_data()  # 得到分词后的id序列和标签
    #读取文件，得到分词后的id序列和标签，返回的都是tensor类型的数据
    def load_data(self):
        input_ids, atention_mask = [], []
        labels, data_init = [], []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading data', colour="green"):
                fields  = line.strip().split('\t')
                if len(fields) != 3:
                    continue
                labels.append([float(1) if i == "1" else float(0) for i in fields[2].split(",")])  # one-hot
                data = self.tokenize_function(fields[0]+fields[1])
                input_ids.append(data['input_ids'])
                atention_mask.append(data['attention_mask'])
                data_init.append(fields[0]+fields[1])
        f.close()
        return torch.tensor(input_ids), torch.tensor(atention_mask), torch.tensor(labels), data_init
    def __len__(self):  # 返回整个数据集的大小
        return len(self.input_ids)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.input_ids[index], self.mask[index], self.labels[index], self.data_init[index]

def getDataLoader(train_dataset, dev_dataset):
    batch_size = 32
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=True,  # 加载数据时打乱样本顺序
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=False,  # 按原始数据集样本顺序加载
    )
    return train_dataloader, dev_dataloader