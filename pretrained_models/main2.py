import os
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import evaluate_multilabel

# 数据加载和预处理
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    if 'title' not in data.columns or 'genres' not in data.columns:
        raise ValueError("CSV 文件需要包含 'title' 和 'genres' 两列！")
    data = data.dropna(subset=['title', 'genres'])
    data['genres'] = data['genres'].str.split('|')
    return data

# 标签编码
def encode_labels(data):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['genres'])
    return y, mlb.classes_

# 自定义数据集类
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

# 数据增强
def oversample_data(X, y):
    from imblearn.over_sampling import SMOTE

    # 检查每个标签的样本数量
    min_samples = np.min(np.sum(y, axis=0))
    print(f"Minimum samples per class: {min_samples}")

    # 如果最小样本数小于 6，调整策略
    if min_samples < 6:
        smote = SMOTE(random_state=42, sampling_strategy='not minority')
    else:
        smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# 模型训练
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

# 模型评估
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

if __name__ == "__main__":
    # 加载数据
    filepath = "data/movies_split.csv"
    data = load_and_preprocess_data(filepath)

    # 标签编码
    y, genre_classes = encode_labels(data)

    # 文本数值化
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['title'])
    sequences = tokenizer.texts_to_sequences(data['title'])
    padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

    # 数据增强
    print("Oversampling data...")
    X_resampled, y_resampled = oversample_data(padded_sequences, y)

    # 获取增强后标题对应的索引
    original_titles = data['title'].values
    resampled_indices = np.random.choice(len(original_titles), len(X_resampled))
    resampled_titles = original_titles[resampled_indices]

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(resampled_titles, y_resampled, test_size=0.2, random_state=42)

    # 初始化预训练模型
    model_name = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(genre_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 准备数据集
    train_dataset = MovieDataset(X_train.tolist(), y_train, bert_tokenizer, max_len=50)
    test_dataset = MovieDataset(X_test.tolist(), y_test, bert_tokenizer, max_len=50)

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



