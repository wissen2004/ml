import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Загрузка данных
try:
    data = pd.read_csv('./data/data.csv', encoding='utf-8')
except Exception as e:
    print(f"Ошибка загрузки файла: {e}")
    exit()

print("Первые 5 строк данных:")
print(data.head())
print("\nТипы данных в столбцах:")
print(data.dtypes)

# Обработка рейтинга
try:
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data = data.dropna(subset=['Rating'])
    data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data[data['Rating'] != 3]
except Exception as e:
    print(f"Ошибка обработки рейтингов: {e}")
    exit()

# Предобработка текста
nltk.download('punkt')
stemmer = SnowballStemmer("russian")
stop_words = {'и', 'в', 'во', 'на', 'с', 'по', 'к', 'у', 'за', 'то', 'что', 'это',
              'как', 'так', 'для', 'а', 'но', 'о', 'же', 'мы', 'вы', 'бы', 'если', 'они'}

def preprocess_text(text):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

data['Processed_Review'] = data['Review'].apply(preprocess_text)
data = data[data['Processed_Review'].str.strip() != '']

# Разделение данных
X = data['Processed_Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Векторизация
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Предсказание
y_pred = model.predict(X_test_vec)

# Оценка качества
print("\nТочность модели:", accuracy_score(y_test, y_pred))
print("\nОтчёт по классификации:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(cm)

# Визуализация матрицы ошибок
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Негативный', 'Позитивный'],
            yticklabels=['Негативный', 'Позитивный'])
plt.xlabel('Предсказано')
plt.ylabel('Истинное')
plt.title('Матрица ошибок')
plt.tight_layout()
plt.show()

# Функция предсказания тональности
def predict_sentiment(text):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    return "Позитивный" if prediction == 1 else "Негативный"

# Примеры предсказаний
test_reviews = [
    "Отличный товар, всем рекомендую!",
    "Ужасное качество, никогда больше не куплю.",
    "Нормально, но могли бы и лучше сделать.",
    "Не интересный товар",
    "Всё прошло нормально, ничего особенного.",
    "Мне понравилось качество, но цена могла быть ниже.",
    "Товар хороший, но доставка была слишком долгой."
]

for review in test_reviews:
    print(f"\nОтзыв: '{review}'")
    print("Тональность:", predict_sentiment(review))

