import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import evaluate_multilabel_relaxed
import matplotlib.pyplot as plt
from xml_methods.dismec import DiSMEC
from xml_methods.xml_cnn import XMLCNN

# 数据加载
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(subset=['title', 'genres'], inplace=True)
    data['genres'] = data['genres'].str.split('|')
    return data

# 标签编码
def encode_labels(genres):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(genres)
    return y, mlb.classes_

# 可视化评估结果
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

# 主程序
if __name__ == "__main__":
    filepath = "./data/movies_split.csv"
    data = load_data(filepath)

    # 标签编码
    y, genre_classes = encode_labels(data['genres'])

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(data['title'], y, test_size=0.2, random_state=42)

    # 特征提取
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 方法 1：DiSMEC
    print("Training DiSMEC...")
    dismec_model = DiSMEC()
    dismec_model.fit(X_train_vec, y_train)
    y_pred_dismec = dismec_model.predict(X_test_vec)

    results_dismec = evaluate_multilabel_relaxed(y_test, y_pred_dismec)
    print("DiSMEC Results:")
    for metric, value in results_dismec.items():
        print(f"{metric}: {value:.4f}")
    plot_metrics(results_dismec, "DiSMEC")

    # 方法 2：XML-CNN
    print("Training XML-CNN...")
    vocab_size = 5000
    max_len = 50
    xml_cnn_model = XMLCNN(vocab_size, max_len, len(genre_classes))
    
    # 文本数据预处理
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    xml_cnn_model.fit(X_train_seq, y_train)
    y_pred_xmlcnn = xml_cnn_model.predict(X_test_seq)

    results_xmlcnn = evaluate_multilabel_relaxed(y_test, y_pred_xmlcnn)
    print("XML-CNN Results:")
    for metric, value in results_xmlcnn.items():
        print(f"{metric}: {value:.4f}")
    plot_metrics(results_xmlcnn, "XML-CNN")
