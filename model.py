# 1. Dependencies
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 2. Load dataset
df = pd.read_csv("email.csv", encoding="latin-1")  # Adjust encoding if needed

df = df.rename(columns={'Category': 'label', 'Message': 'message'})
df = df[['label', 'message']]  # Keep only relevant columns

# Convert labels to binary (spam=1, ham=0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()

# 3. Text Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['processed_message'] = df['message'].apply(preprocess_text)

# 4. Convert Text into numerical form
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = vectorizer.fit_transform(df['processed_message']).toarray()
y = df['label'].values  # Target labels (spam or ham)

# Step 5: Split Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Machine Learning Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Step 8: Test with Custom Messages
def predict_spam(message):
    processed_message = preprocess_text(message)
    vectorized_message = vectorizer.transform([processed_message]).toarray()
    prediction = model.predict(vectorized_message)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test example
print(predict_spam("Congratulations! You won a free iPhone. Click here to claim."))
print(predict_spam("Hey, are we still meeting at 5 PM?"))

# Step 9: Save the Model for Deployment

# Save model and vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load model for future use
loaded_model = joblib.load("spam_classifier.pkl")
loaded_vectorizer = joblib.load("vectorizer.pkl")

# Predict with the loaded model
print(predict_spam("You have won a lottery! Call now!"))
