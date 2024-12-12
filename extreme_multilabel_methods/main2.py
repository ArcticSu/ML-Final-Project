import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import evaluate_multilabel_custom
from fastxml import FastXML
from label_powerset import LabelPowerset

# 数据加载和预处理
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    if 'title' not in data.columns or 'genres' not in data.columns:
        raise ValueError("CSV 文件需要包含 'title' 和 'genres' 两列！")
    data = data.dropna(subset=['title', 'genres'])
    data['genres'] = data['genres'].str.split('|')
    return data

# 特征提取
def prepare_text_data(data, max_words=10000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_words)
    X = vectorizer.fit_transform(data['title'])
    return X, vectorizer

# 标签编码
def encode_labels(data):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['genres'])
    return y, mlb.classes_

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



# 模型训练与评估（传统方法）
def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测结果
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    results = evaluate_multilabel_custom(y_test, y_pred)
    
    # 输出评估结果
    print(f"{model_name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results

# 主程序入口
if __name__ == "__main__":
    # 加载数据
    filepath = "data/movies_split.csv"
    data = load_and_preprocess_data(filepath)

    # 特征提取与标签编码
    X, vectorizer = prepare_text_data(data)
    y, genre_classes = encode_labels(data)

    # 数据增强
    print("Oversampling data...")
    X, y = oversample_data(X, y)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型对比
    results = {}

    # FastXML
    print("Training FastXML...")
    fastxml_model = FastXML(n_trees=10, max_depth=10, min_samples_split=5)
    results['FastXML'] = train_and_evaluate(X_train, X_test, y_train, y_test, fastxml_model, "FastXML")

    # LabelPowerset
    print("Training LabelPowerset...")
    from sklearn.tree import DecisionTreeClassifier
    base_classifier = DecisionTreeClassifier(max_depth=10, random_state=42)
    label_powerset_model = LabelPowerset(base_classifier=base_classifier)
    results['LabelPowerset'] = train_and_evaluate(X_train, X_test, y_train, y_test, label_powerset_model, "LabelPowerset")

