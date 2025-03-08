from flask import Flask, request, jsonify
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Load pre-trained model and TF-IDF vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# NLTK-based text preprocessing
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Preprocess the message
    cleaned_message = preprocess_text(message)

    # Convert message to TF-IDF features
    message_tfidf = vectorizer.transform([cleaned_message])

    # Predict
    prediction = model.predict(message_tfidf)[0]
    result = "spam" if prediction == 1 else "ham"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
