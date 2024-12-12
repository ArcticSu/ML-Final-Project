import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from imblearn.over_sampling import SMOTE
from models.textcnn import TextCNN
from models.textrnn import TextRNN
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

# 文本数据处理
def prepare_text_data(data, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['title'])
    sequences = tokenizer.texts_to_sequences(data['title'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

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
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', AUC(name='auc')])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=1)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    results = evaluate_multilabel_custom(y_test, y_pred)

    print(f"{model_name} Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results

# 主程序入口
if __name__ == "__main__":
    # 加载数据
    filepath = "data/movies_split.csv"
    data = load_and_preprocess_data(filepath)

    # 文本数据处理
    max_words = 10000
    max_len = 100
    X, tokenizer = prepare_text_data(data, max_words=max_words, max_len=max_len)

    # 标签编码
    y, genre_classes = encode_labels(data)

    # 数据增强
    print("Oversampling data...")
    X, y = oversample_data(X, y)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型对比
    results = {}

    # TextCNN
    print("Training TextCNN...")
    textcnn_model = TextCNN(max_words, max_len, len(genre_classes))
    results['TextCNN'] = train_and_evaluate_deep_learning(X_train, X_test, y_train, y_test, textcnn_model, "TextCNN")

    # TextRNN
    print("Training TextRNN...")
    textrnn_model = TextRNN(max_words, max_len, len(genre_classes))
    results['TextRNN'] = train_and_evaluate_deep_learning(X_train, X_test, y_train, y_test, textrnn_model, "TextRNN")

    # 打印最终结果
    for model_name, model_results in results.items():
        print(f"{model_name} Results:")
        for metric, value in model_results.items():
            print(f"{metric}: {value:.4f}")
