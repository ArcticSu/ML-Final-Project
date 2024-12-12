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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
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
def train_and_evaluate_deep_learning(X_train, X_test, y_train, y_test, model, model_name, batch_size=32, epochs=10):
    # 编译模型
    model.model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy', AUC(name='auc')])

    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 用于记录 Micro F1 分数的列表
    micro_f1_scores = []

    # 自定义训练循环
    for epoch in range(1, epochs + 1):
        # 单个 epoch 训练
        model.model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=1,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stopping])

        # 预测并计算 F1 分数
        y_pred = (model.model.predict(X_test) > 0.5).astype(int)
        results = evaluate_multilabel_custom(y_test, y_pred)
        micro_f1_scores.append(results['Micro F1'])

        # 打印本 epoch 的 Micro F1
        print(f"Epoch {epoch}/{epochs} - Micro F1: {results['Micro F1']:.4f}")

        # 早停条件：连续 3 个 epoch F1 分数下降
        if epoch > 3 and micro_f1_scores[-1] < micro_f1_scores[-2]:
            print("Early stopping due to decreasing Micro F1")
            break

    # 可视化 Micro F1 分数
    # plot_micro_f1(None, micro_f1_scores, model_name)

    # 评估最终模型
    final_results = evaluate_multilabel_custom(y_test, (model.model.predict(X_test) > 0.5).astype(int))

    # 打印最终结果
    print(f"{model_name} Final Results:")
    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")

    return final_results

# 主程序入口
if __name__ == "__main__":
    filepath = "data/movies_split.csv"
    data = load_and_preprocess_data(filepath)

    # 标签编码
    y, genre_classes = encode_labels(data)

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

    # 使用自定义训练方法
    xml_cnn_results = train_and_evaluate_deep_learning(X_train_seq, X_test_seq, y_train, y_test, xml_cnn_model, "XML-CNN")
