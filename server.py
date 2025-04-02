from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk

app = Flask(__name__)

nltk.download('punkt')
stemmer = SnowballStemmer("russian")
stop_words = {'и', 'в', 'во', 'на', 'с', 'по', 'к', 'у', 'за', 'то', 'что', 'это', 'как', 'так', 'для', 'а', 'но', 'о', 'же', 'мы', 'вы', 'бы', 'если', 'они'}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    reviews = request.json.get('reviews', [])
    results = []
    for review in reviews:
        processed = preprocess_text(review)
        vec = vectorizer.transform([processed])
        prediction = model.predict(vec)[0]
        results.append({
            'review': review,
            'sentiment': 'Позитивный' if prediction == 1 else 'Негативный'
        })
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
