import os
import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import evaluate_multilabel
import matplotlib.pyplot as plt

# 1. 数据加载
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(subset=['title', 'genres'], inplace=True)
    data['genres'] = data['genres'].str.split('|')
    return data

# 2. 标签编码
def encode_labels(genres):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(genres)
    return y, mlb.classes_

# 3. 自定义数据集类
class MovieDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# 4. 模型训练
def train_model(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 5. 模型评估
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).cpu().numpy() > 0.5
            y_true.extend(labels)
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

# 6. 可视化评估结果
def plot_metrics(results, model_name):
    metrics = list(results.keys())
    values = list(results.values())
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color='skyblue')
    plt.title(f"{model_name} Performance Metrics")
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.show()

# 主程序入口
if __name__ == "__main__":
    # 数据路径
    filepath = "./data/movies_split.csv"
    data = load_data(filepath)

    # 标签编码
    y, genre_classes = encode_labels(data['genres'])

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(data['title'], y, test_size=0.2, random_state=42)

    # 初始化预训练模型
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(genre_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 准备数据集
    train_dataset = MovieDataset(X_train.tolist(), y_train, tokenizer, max_len=50)
    test_dataset = MovieDataset(X_test.tolist(), y_test, tokenizer, max_len=50)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练
    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    # 评估
    y_true, y_pred = evaluate_model(model, test_loader, device)
    results = evaluate_multilabel(y_true, y_pred)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # 可视化
    plot_metrics(results, "BERT")
