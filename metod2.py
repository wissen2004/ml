import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("ru_core_news_sm")

def clean_text(text):
    if isinstance(text, str):
        return text.lower()
    return ""


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Review'] = data['Review'].apply(clean_text)
    return data


def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def prepare_data(data):
    X = data['Review']
    y = data['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def train_model(X_train_vec, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy * 100:.2f}%")

def predict_sentiment(model, vectorizer, new_review):
    new_review = preprocess_text(new_review)
    new_review_vec = vectorizer.transform([new_review])
    predicted_rating = model.predict(new_review_vec)[0]
    predicted_rating = int(predicted_rating)
    sentiment = "Положительный" if predicted_rating > 3 else "Отрицательный"
    return sentiment


if __name__ == "__main__":
    file_path = "./data/data.csv"
    data = load_data(file_path)

    if data is not None:
        X_train, X_test, y_train, y_test = prepare_data(data)
        X_train_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_test)
        model = train_model(X_train_vec, y_train)
        evaluate_model(model, X_test_vec, y_test)

        new_review = "Не интересный телефон"
        sentiment = predict_sentiment(model, vectorizer, new_review)
        print(f"Настроение для нового отзыва: {sentiment}")