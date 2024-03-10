import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from text_utils import preprocess_text


def plot_cf_matrix(test, pred, model_name):
    cf_matrix = confusion_matrix(test, pred)
    cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    cf_matrix_percentage = cf_matrix_normalized * 100

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2f}%'.format(value) for value in cf_matrix_percentage.flatten()]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_normalized, annot=labels, fmt='', cmap='Blues', 
                xticklabels=categories, yticklabels=categories, cbar=False)
    
    plt.xlabel("Predicted values", fontdict={'size':14}, labelpad=10)
    plt.ylabel("Actual values"   , fontdict={'size':14}, labelpad=10)
    plt.title(f"Confusion Matrix - {model_name}", fontdict={'size':18}, pad=20)
    plt.show()

def evaluate(model, X_test_vec, y_test,model_name):
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    plot_cf_matrix(y_test, y_pred, model_name)

def predict_sentiment(vectorizer, model, texts):
    texts_processed = preprocess_text(texts)
    text_data = vectorizer.transform(texts_processed)
    predictions = model.predict(text_data)
    
    sentiment_labels = np.where(predictions==1, "Positive", "Negative")

    df = pd.DataFrame({'text': texts, 'sentiment': sentiment_labels})
    return df

def build_simple_RNN_model(input_length, learning_rate=0.001, vocab_size=10000, embedding_dim=100):
    model = Sequential()
    model.add(Input(shape=(input_length,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(SimpleRNN(64))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model