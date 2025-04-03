import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import nltk

df = pd.read_csv('./data/data.csv')

df = df.dropna(subset=['Review', 'Rating'])

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

df['Rating'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

nltk.download('punkt')

from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

df['Processed_Review'] = df['Review'].apply(preprocess_text)

X = df['Processed_Review']
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

model = LinearSVC(random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Точность модели:", accuracy_score(y_test, y_pred))

def predict_and_comment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review]).toarray()

    prediction = model.predict(review_tfidf)

    if prediction == 1:
        return "Положительный отзыв"
    else:
        return "Отрицательный отзыв"

review = "Этот продукт плохой!"
comment = predict_and_comment(review)
print(comment)
