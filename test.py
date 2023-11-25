import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, accuracy_score
from torch.utils.data import DataLoader
from data_processing import MyData, tokenize_textCNN
from transformers import AutoTokenizer
import logging
import sys

logging.basicConfig(level=logging.DEBUG,  # 设置最低记录级别为DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建一个记录器
logger = logging.getLogger('dev')

# 创建一个文件处理程序，将日志信息输出到文件
file_handler = logging.FileHandler('./test.log')  # 指定日志文件名
file_handler.setLevel(logging.INFO)  # 设置输出处理程序的记录级别
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理程序添加到记录器
logger.addHandler(file_handler)

label_list = {}
label_map = {}
with open("data/label.txt", mode='r') as f:
    for i, line in enumerate(f):
        l = line.strip()
        label_list[l] = i
        label_map[i] = l 

def evaluate(model, device, dataload, model_name):
    model.eval()
    loss_total = 0.
    threshold = 0.5
    probs = []
    labels = []
    deberta_list = ["DeBERTa-v3-base-mnli-fever-anli", "deberta-v2-xlarge-mnli"]
    with torch.no_grad():
        for input_ids, mask, labels_batch, _ in dataload:
            outputs = model(input_ids.to(device), mask.to(device))
            if model_name in deberta_list:
                loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels_batch.to(device))  # 计算交叉熵损失
                loss_total += loss
                labels.extend(labels_batch.data.cpu().numpy()) # extend
                probs.extend(F.sigmoid(outputs['logits']).cpu().numpy()) # extend
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, labels_batch.to(device))  # 计算交叉熵损失
                loss_total += loss
                labels.extend(labels_batch.data.cpu().numpy()) # extend
                probs.extend(F.sigmoid(outputs).cpu().numpy()) # extend
        probs = np.array(probs)
        labels = np.array(labels)
        preds = probs > threshold
        report = classification_report(labels, preds, digits=4, output_dict=True, zero_division=1)
        accuracy = accuracy_score(labels, preds)
    model.train()
    return accuracy, report, loss_total / len(dataload), (preds, labels)


def evaluateDNN(model, device, dataload):
    model.eval()
    loss_total = 0.
    threshold = 0.5
    probs = []
    labels = []
    with torch.no_grad():
        for data, labels_batch in dataload:
            outputs = model(data.to(device))
            loss = F.binary_cross_entropy_with_logits(outputs, labels_batch.to(device))  # 计算交叉熵损失
            loss_total += loss
            labels.extend(labels_batch.data.cpu().numpy()) # extend
            probs.extend(F.sigmoid(outputs).cpu().numpy()) # extend
        probs = np.array(probs)
        labels = np.array(labels)
        preds = probs > threshold
        report = classification_report(labels, preds, digits=4, output_dict=True, zero_division=1)
        accuracy = accuracy_score(labels, preds)
    model.train()
    return accuracy, report, loss_total / len(dataload), (preds, labels)

def testModel(model_file, test_file):
    MODEL_NAME = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # 实例化分词器

    def tokenize_BERT(s):
        return tokenizer.encode(s, max_length=200, truncation=True, padding="max_length")

    # 得到数据集
    test_dataset = MyData(tokenize_fun=tokenize_BERT, filename=test_file)
    batch_size = 64
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=False,  # 加载数据时不打乱样本顺序
    )
    model = torch.load(model_file, map_location=lambda s,l:s) # 不改变原始位置, 即还是与原来一样的位置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    model.to(device).eval()
    accuracy, report, loss, (preds, labels) = evaluate(model, device, test_dataloader)

    with open("failed_predictions.txt", mode='w') as f:
        f.write("data\ttrue_label\tpre_label\n")
        for item in range(len(preds)):
            data, label, predic = test_dataset.data_init[item], [label_map[i] for i in range(len(label_list)) if labels[item][i] == 1] ,[label_map[i] for i in range(len(label_list)) if preds[item][i] == True]
            line = f"{data}\t{label}\t{predic}\n" 
            f.write(line)

    logger.info("-----Evaluate model-------")
    logger.info("Test dataset size: {}".format(len(test_dataset)))
    logger.info("Test loss {:.4f}".format(loss))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(accuracy * 100))

    logger.info(
    "Micro avg in test dataset: precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
        report["micro avg"]["precision"] * 100,
        report["micro avg"]["recall"] * 100,
        report["micro avg"]["f1-score"] * 100,
    )
)
    logger.info(
        "Macro avg in test dataset: precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            report["macro avg"]["precision"] * 100,
            report["macro avg"]["recall"] * 100,
            report["macro avg"]["f1-score"] * 100,
        )
    )

    for i in range(len(label_map)):
        logger.info("Class name {}".format(label_map[i]))
        logger.info(
            "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                report[str(i)]["support"],
                100 * report[str(i)]["support"] / len(test_dataset),
                report[str(i)]["precision"] * 100,
                report[str(i)]["recall"] * 100,
                report[str(i)]["f1-score"] * 100,
            )
        )
    logger.info("----------------------------")

if __name__ == '__main__':
    testModel(model_file="checkpoint/bert-base-uncased/bert-base-uncased_best_model", test_file="data/test.txt")