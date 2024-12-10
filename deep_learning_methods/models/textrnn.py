from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def TextRNN(max_words, max_len, num_classes):
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    return model
