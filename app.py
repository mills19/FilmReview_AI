from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model & vectorizer
model = joblib.load("logistic_regression_imdb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Route for home page
@app.route('/')
def home():
    return render_template("index.html")

# Route to predict sentiment
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocess_text(review)
    review_vectorized = vectorizer.transform([processed_review])
    prediction = model.predict(review_vectorized)
    sentiment = "Positive review ☺️" if prediction[0] == 1 else "Negative review 🙁"
    return jsonify({'sentiment': sentiment})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
