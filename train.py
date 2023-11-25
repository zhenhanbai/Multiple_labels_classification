import os
import torch
import torch.nn.functional as F
import numpy as np
from test import evaluate, evaluateDNN
from models import (
    ModelTextCNNForSequenceClassification,
    ModelRNNForSequenceClassification,
    ModelRCNNForSequenceClassification,
    ModelForSequenceClassification, 
    TextCNNModel, 
    FastTextModel,
    RNNModel)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from data_processing import MyData, MyDataV2, getDataLoader, tokenize_textCNN 

import logging
import sys

logging.basicConfig(level=logging.DEBUG,  # 设置最低记录级别为DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建一个记录器
logger = logging.getLogger('train')

# 创建一个文件处理程序，将日志信息输出到文件
file_handler = logging.FileHandler('./train.log')  # 指定日志文件名
file_handler.setLevel(logging.INFO)  # 设置输出处理程序的记录级别
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理程序添加到记录器
logger.addHandler(file_handler)

def setSeed(seed): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
setSeed(seed=42)

# tensorboard --logdir=training/

from tensorboardX import SummaryWriter
import time
from datetime import timedelta

def get_time_dif(start_time):  # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def trainDNNS(model, device, model_name, lr, train_dataloader, dev_dataloader):
    start_time = time.time()  # 记录起始时间
    model.train()  # 设置model为训练模式
    total_batch = 0  # 记录进行到多少batch
    num_epochs = 20  # 设置训练次数
    total_steps = len(train_dataloader) * num_epochs
    # Create the learning rate scheduler.
    optimizer = torch.optim.Adam(model.parameters(), 
                lr = lr
            )
    dev_best_f1_score = float('-inf')  # 记录验证集上的最好损失
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    if not os.path.exists('log'):
        os.makedirs('log', exist_ok=True)
    if not os.path.exists(f'./checkpoint/{model_name}'):
        os.makedirs(f'checkpoint/{model_name}', exist_ok=True)
    writer = SummaryWriter(log_dir='log/%s.'%(model_name) + time.strftime('%m-%d_%H.%M', time.localtime()))  # 实例化SummaryWriter
    for epoch in range(num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (data, labels) in enumerate(train_dataloader):
            outputs = model(data.to(device))  # 
            model.zero_grad()  # 模型梯度清零
            loss = F.binary_cross_entropy_with_logits(outputs, labels.to(device))  # 计算交叉熵损失
            loss.backward()  # 梯度回传
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # 更新参数
            writer.add_scalar("loss/train", loss.item(), total_batch)
            if total_batch % 100  == 0:
                accuracy, report, dev_loss, _ = evaluateDNN(model, device, dev_dataloader) 
                micro_f1_score, macro_f1_score = report["micro avg"]["f1-score"], report["macro avg"]["f1-score"]
                if macro_f1_score > dev_best_f1_score:
                    dev_best_f1_score = macro_f1_score
                    checkpoint_path = f"checkpoint/{model_name}/{model_name}_best_model" 
                    torch.save(model, checkpoint_path)
                    improve = '*'  # 设置标记
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)  # 得到当前运行时间

                msg = 'Iter:{0:>4}, Train-Loss:{1:>6.4}, Val-Loss:{2:>6.4}, Val-Acc:{3:>6.2%}, Macro-Score:{4:>6.2%}, Micro-Score:{5:>6.2%}, Time:{6}, Improve:{7}' 
                logger.info(msg.format(total_batch, loss.item(), dev_loss, accuracy, macro_f1_score, micro_f1_score, time_dif, improve))
                # 写入tensorboardX可视化用的日志信息
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("macro/dev", macro_f1_score, total_batch)
                writer.add_scalar("micro/dev", micro_f1_score, total_batch) 
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()  # 关闭writer对象

def train(model, device, model_name, lr, train_dataloader, dev_dataloader):
    start_time = time.time()  # 记录起始时间
    model.train()  # 设置model为训练模式
    total_batch = 0  # 记录进行到多少batch
    num_epochs = 5  # 设置训练次数
    total_steps = len(train_dataloader) * num_epochs

    if "TextCNN" in model_name:
        bert_parameters = list(model.model.parameters()) 
        cnn_parameters = list(model.convs.parameters()) 
        # 设置不同部分的学习率
        optimizer = torch.optim.AdamW([
            {'params': bert_parameters, 'lr': lr},
            {'params': cnn_parameters, 'lr': 1e-3}
        ], weight_decay=0.01, eps=1e-8)
    elif "RNN" in model_name or "RCNN" in model_name:
        bert_parameters = list(model.model.parameters()) 
        rnn_parameters = list(model.rnn.parameters())
        # 设置不同部分的学习率
        optimizer = torch.optim.AdamW([
            {'params': bert_parameters, 'lr': lr},
            {'params': rnn_parameters, 'lr': 1e-3} 
        ], weight_decay=0.01, eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-8
        )
    # Create the learning rate scheduler.
    ratio = 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = ratio * total_steps, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    dev_best_f1_score = float('-inf')  # 记录验证集上的最好损失
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    deberta_list = ["DeBERTa-v3-base-mnli-fever-anli", "deberta-v2-xlarge-mnli"]
    if not os.path.exists('log'):
        os.makedirs('log', exist_ok=True)
    if not os.path.exists(f'./checkpoint/{model_name}'):
        os.makedirs(f'checkpoint/{model_name}', exist_ok=True)
    writer = SummaryWriter(log_dir='log/%s.'%(model_name) + time.strftime('%m-%d_%H.%M', time.localtime()))  # 实例化SummaryWriter
    for epoch in range(num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (input_ids, mask, labels, _) in enumerate(train_dataloader):
            outputs = model(input_ids.to(device), mask.to(device))  # 
            model.zero_grad()  # 模型梯度清零
            if model_name in deberta_list:
                loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels.to(device))  # 计算交叉熵损失
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, labels.to(device))  # 计算交叉熵损失
            loss.backward()  # 梯度回传
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # 更新参数
            scheduler.step()
            writer.add_scalar("loss/train", loss.item(), total_batch)
            if total_batch % 80 == 0:
                accuracy, report, dev_loss, _ = evaluate(model, device, dev_dataloader, model_name)
                micro_f1_score, macro_f1_score = report["micro avg"]["f1-score"], report["macro avg"]["f1-score"]
                if macro_f1_score > dev_best_f1_score:
                    dev_best_f1_score = macro_f1_score
                    checkpoint_path = f"checkpoint/{model_name}/{model_name}_best_model"
                    torch.save(model, checkpoint_path)
                    improve = '*'  # 设置标记
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)  # 得到当前运行时间

                msg = 'Iter:{0:>4}, Train-Loss:{1:>6.4}, Val-Loss:{2:>6.4}, Val-Acc:{3:>6.2%}, Macro-Score:{4:>6.2%}, Micro-Score:{5:>6.2%}, Time:{6}, Improve:{7}'
                logger.info(msg.format(total_batch, loss.item(), dev_loss, accuracy, macro_f1_score, micro_f1_score, time_dif, improve))
                # 写入tensorboardX可视化用的日志信息
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("macro/dev", macro_f1_score, total_batch) 
                writer.add_scalar("micro/dev", micro_f1_score, total_batch)
            total_batch += 1
            if total_batch - last_improve >= 640:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()  # 关闭writer对象

def trainDNN():
    MODEL_LIST = [FastTextModel, TextCNNModel, RNNModel]
    for MODEL_NAME in MODEL_LIST:
        logger.info("----------------------\n %s", MODEL_NAME.__name__)
        train_dataset = MyDataV2(tokenize_fun=tokenize_textCNN, filename='data/train.txt')
        dev_dataset = MyDataV2(tokenize_fun=tokenize_textCNN, filename='data/dev.txt')
        train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        model = MODEL_NAME().to(device)
        lr = 1e-3  # Adam优化器学习率
        trainDNNS(model, device, MODEL_NAME.__name__, lr, train_dataset, dev_dataset)  # 开始


def trainBert():
    MODEL_LIST = ["xlm-roberta-base", "xlnet-base-cased", "bert-base-uncased", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"]
    deberta_list = ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "microsoft/deberta-v2-xlarge-mnli"]
    for MODEL_NAME in MODEL_LIST:
        logger.info("----------------------\n %s", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # 实例化分词器
        # 定义分词函数
        def tokenize_BERT(s):
            return tokenizer(s, truncation=True, max_length=200, padding="max_length")
        # 得到数据集
        train_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/train.txt')
        dev_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/dev.txt')
        # 得到数据加载器
        train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
        # 定义模型
        if MODEL_NAME in deberta_list:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6, ignore_mismatched_sizes=True)
        else:
            model = ModelForSequenceClassification(MODEL_NAME, num_classes=6) 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        model = model.to(device)
        lr = 3e-5  # 设置Adam优化器学习率
        model_name = MODEL_NAME.split('/')[1] if len(MODEL_NAME.split('/')) > 1 else MODEL_NAME.split('/')[0] 
        train(model, device, f"{model_name}", lr, train_dataset, dev_dataset)


def trainBertDNN():
    MODEL_LIST = [ModelTextCNNForSequenceClassification, ModelRNNForSequenceClassification, ModelRCNNForSequenceClassification,]
    PLM = ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "bert-base-uncased", "xlm-roberta-base", "xlnet-base-cased"]
    for model_base in PLM:
        for MODEL_NAME in MODEL_LIST:
            logger.info("----------------------\n %s", str(model_base + "-" + MODEL_NAME.__name__))
            tokenizer = AutoTokenizer.from_pretrained(model_base)  # 实例化分词器 
            # 得到数据集
            def tokenize_BERT(s):
                return tokenizer(s, max_length=200, truncation=True, padding="max_length")
            train_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/train.txt')
            dev_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/dev.txt')
            # 得到数据加载器
            train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
            # 定义模型
            model = MODEL_NAME(model_base, num_classes=6) 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
            model = model.to(device)
            lr = 3e-5  # 设置Adam优化器学习率
            model_base_filter = model_base.split('/')[1] if len(model_base.split('/')) > 1 else model_base.split('/')[0] 
            train(model, device, model_base_filter + "-" + MODEL_NAME.__name__, lr, train_dataset, dev_dataset)


if __name__ == '__main__': 
    trainDNN() 
    trainBert()
    trainBertDNN()
