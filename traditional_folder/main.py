import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# 可选：导入 XGBoost
from xgboost import XGBClassifier

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

# 模型训练与评估
def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"{model_name} F1 Score: {f1}")
    print(classification_report(y_test, y_pred))
    return f1

# 结果可视化
def plot_results(results):
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    scores = list(results.values())
    plt.bar(models, scores, color='skyblue')
    plt.title("Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("F1 Score")
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
    plot_results(results)
