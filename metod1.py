import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

df = pd.read_csv("./data/data.csv")

df.drop_duplicates(inplace=True)
df.dropna(subset=["Review", "Rating"], inplace=True)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])
df["Rating"] = df["Rating"].astype(int)
df = df[df["Rating"].between(1, 5)]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [word for i, word in enumerate(words) if word not in stop_words or (i > 0 and words[i - 1] == "не")]
    return " ".join(words)

df["Cleaned_Review"] = df["Review"].apply(clean_text)

df = df[df["Rating"] != 3]

def categorize_rating(rating):
    return "negative" if rating in [1, 2] else "positive"

df["Category"] = df["Rating"].apply(categorize_rating)

X_train, X_test, y_train, y_test = train_test_split(
    df["Cleaned_Review"], df["Category"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_review(review):
    review_cleaned = clean_text(review)
    review_tfidf = vectorizer.transform([review_cleaned])
    prediction = model.predict(review_tfidf)[0]
    return prediction

example_review = "Крутой продукт!"
print(f"Прогноз: {predict_review(example_review)}")
