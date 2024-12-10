import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluation import evaluate_multilabel
from models.textcnn import TextCNN
from models.textrnn import TextRNN

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

# 文本序列处理
def prepare_text_sequences(titles, max_len=50, max_words=5000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(titles)
    sequences = tokenizer.texts_to_sequences(titles)
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=max_len)
    return X, tokenizer, word_index

# 模型训练与评估
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # 预测和评估
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    evaluation_results = evaluate_multilabel(y_test, y_pred_binary)

    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    return model

# 主程序
if __name__ == "__main__":
    filepath = "./data/movies_split.csv"  # 确保路径正确
    data = load_data(filepath)

    # 标签编码和序列处理
    y, genre_classes = encode_labels(data['genres'])
    X, tokenizer, word_index = prepare_text_sequences(data['title'])

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TextCNN
    print("Training TextCNN...")
    textcnn_model = TextCNN(max_words=5000, max_len=50, num_classes=len(genre_classes))
    train_and_evaluate_model(textcnn_model, X_train, y_train, X_test, y_test)

    # TextRNN
    print("Training TextRNN...")
    textrnn_model = TextRNN(max_words=5000, max_len=50, num_classes=len(genre_classes))
    train_and_evaluate_model(textrnn_model, X_train, y_train, X_test, y_test)
