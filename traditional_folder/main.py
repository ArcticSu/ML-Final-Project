import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation import evaluate_multilabel_custom

# 数据加载和预处理
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    if 'title' not in data.columns or 'genres' not in data.columns:
        raise ValueError("CSV 文件需要包含 'title' 和 'genres' 两列！")
    data = data.dropna(subset=['title', 'genres'])
    data['genres'] = data['genres'].str.split('|')
    return data

# 特征提取
def extract_features(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['title'])
    return X, vectorizer

# 标签编码
def encode_labels(data):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['genres'])
    return y, mlb.classes_

# 特征选择
def select_features(X, y, k=2000):
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

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



# 模型训练与评估
def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = (model.predict_proba(X_test) > 0.5).astype(int)
    results = evaluate_multilabel_custom(y_test, y_pred)
    print(f"{model_name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    # plot_metrics(results, model_name)
    return results

# 结果可视化
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
    # 加载数据
    filepath = "data/movies_split.csv"
    data = load_and_preprocess_data(filepath)

    # 特征提取与标签编码
    X, vectorizer = extract_features(data)
    y, genre_classes = encode_labels(data)

    # 特征选择
    print("Selecting features...")
    X, selector = select_features(X, y, k=2000)

    # 数据增强
    print("Oversampling data...")
    X, y = oversample_data(X, y)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型对比
    results = {}

    # SVM
    print("Training SVM...")
    svm_model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    results['SVM'] = train_and_evaluate(X_train, X_test, y_train, y_test, svm_model, "SVM")

    # Random Forest
    print("Training Random Forest...")
    rf_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    results['Random Forest'] = train_and_evaluate(X_train, X_test, y_train, y_test, rf_model, "Random Forest")

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    results['Logistic Regression'] = train_and_evaluate(X_train, X_test, y_train, y_test, lr_model, "Logistic Regression")

    # XGBoost
    print("Training XGBoost...")
    xgb_model = OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    results['XGBoost'] = train_and_evaluate(X_train, X_test, y_train, y_test, xgb_model, "XGBoost")

    # 结果可视化
    for model_name, model_results in results.items():
        plot_metrics(model_results, model_name)
