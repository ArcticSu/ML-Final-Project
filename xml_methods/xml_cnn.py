import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Embedding, Dropout

class XMLCNN:
    def __init__(self, vocab_size, max_len, num_labels, embedding_dim=128, dropout_rate=0.5):
        """
        XML-CNN: 基于 CNN 的极端多标签分类方法。
        参数:
        - vocab_size: 词汇表大小
        - max_len: 最大序列长度
        - num_labels: 标签数量
        - embedding_dim: 嵌入维度
        - dropout_rate: Dropout 比例
        """
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_labels, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X, y, batch_size=32, epochs=5):
        """
        训练 XML-CNN 模型。
        参数:
        - X: 输入数据
        - y: 标签
        """
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)

    def predict(self, X):
        """
        使用训练好的模型进行预测。
        参数:
        - X: 输入数据
        返回:
        - 多标签预测结果
        """
        return (self.model.predict(X) > 0.5).astype(int)
