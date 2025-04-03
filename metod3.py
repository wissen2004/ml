import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv('./data/data.csv')

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

df = df.dropna(subset=['Review', 'Rating'])

df['Rating'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

df['Processed_Review'] = df['Review'].apply(preprocess_text)

X = df['Processed_Review']
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Оценка модели:")
print(classification_report(y_test, y_pred))

def predict_and_comment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review]).toarray()

    prediction = model.predict(review_tfidf)

    if prediction == 1:
        return "Положительный отзыв"
    else:
        return "Отрицательный отзыв"

review = "Не интересный телефон"
comment = predict_and_comment(review)
print(comment)
