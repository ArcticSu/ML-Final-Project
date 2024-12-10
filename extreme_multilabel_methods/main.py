import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import evaluate_multilabel
import matplotlib.pyplot as plt
from extreme_multilabel_methods.fastxml import FastXML
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier  # 修正：添加此导入

from extreme_multilabel_methods.label_powerset import LabelPowerset


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
    # 数据路径
    filepath = "./data/movies_split.csv"
    data = load_data(filepath)

    # 标签编码
    y, genre_classes = encode_labels(data['genres'])

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(data['title'], y, test_size=0.2, random_state=42)

    # 文本特征提取
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 方法 1：使用 FastXML
    print("Training FastXML...")
    fastxml_model = FastXML(n_trees=5, max_depth=10)
    fastxml_model.fit(X_train_vec, y_train)
    y_pred_fastxml = fastxml_model.predict(X_test_vec)
    results_fastxml = evaluate_multilabel(y_test, y_pred_fastxml)
    print("FastXML Results:")
    for metric, value in results_fastxml.items():
        print(f"{metric}: {value:.4f}")
    plot_metrics(results_fastxml, "FastXML")

    # 方法 2：使用 Label Powerset
    print("Training Label Powerset...")

    lp_model = LabelPowerset(base_classifier=RandomForestClassifier(n_estimators=100, random_state=42))
    lp_model.fit(X_train_vec, y_train)
    y_pred_lp = lp_model.predict(X_test_vec)


    results_lp = evaluate_multilabel(y_test, y_pred_lp)
    print("Label Powerset Results:")
    for metric, value in results_lp.items():
        print(f"{metric}: {value:.4f}")
    plot_metrics(results_lp, "Label Powerset")
