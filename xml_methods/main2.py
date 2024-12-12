import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dismec import DiSMEC
from xml_cnn import XMLCNN
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

# 模型训练与评估
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    results = evaluate_multilabel_custom(y_test, y_pred)
    print(f"{model_name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results

# 主程序入口
if __name__ == "__main__":
    filepath = "data/movies_split.csv"
    data = load_and_preprocess_data(filepath)

    # 标签编码
    y, genre_classes = encode_labels(data)

    # 数据增强前的文本处理
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(data['title'])

    # 数据增强
    X_tfidf, y = oversample_data(X_tfidf, y)

    # 分割数据集
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # 方法 1：DiSMEC
    print("Training DiSMEC...")
    dismec_model = DiSMEC()
    dismec_results = train_and_evaluate_model(X_train_tfidf, X_test_tfidf, y_train, y_test, dismec_model, "DiSMEC")


    

    # 文本序列化处理
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['title'])

    # 确保数据同步
    data = data.dropna(subset=['title', 'genres'])
    X_seq = pad_sequences(tokenizer.texts_to_sequences(data['title']), maxlen=50)
    y, genre_classes = encode_labels(data)

    # 检查 X 和 y 的一致性
    if X_seq.shape[0] != y.shape[0]:
        raise ValueError(f"Inconsistent sample sizes: X has {X_seq.shape[0]} rows but y has {y.shape[0]} rows")

    # 数据增强
    X_seq, y = oversample_data(X_seq, y)


    # 分割数据集
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

    # 方法 2：XML-CNN
print("Training XML-CNN...")
xml_cnn_model = XMLCNN(vocab_size=5000, max_len=50, num_labels=len(genre_classes))

# 调用 XML-CNN 的训练方法，设置 epochs=10
xml_cnn_model.fit(X_train_seq, y_train, batch_size=32, epochs=10)

# 预测和评估
y_pred_xmlcnn = xml_cnn_model.predict(X_test_seq)
y_pred_xmlcnn = (y_pred_xmlcnn > 0.5).astype(int)
xml_cnn_results = evaluate_multilabel_custom(y_test, y_pred_xmlcnn)

print("XML-CNN Results:")
for metric, value in xml_cnn_results.items():
    print(f"{metric}: {value:.4f}")

