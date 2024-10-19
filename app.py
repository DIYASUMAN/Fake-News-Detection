from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('fake_or_real_news.csv')

# Feature extraction using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer for later use
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Load model and vectorizer for prediction
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input news text from form
    news_text = request.form['news_text']
    
    # Transform input text using the loaded vectorizer
    input_vector = vectorizer.transform([news_text])
    
    # Predict using the loaded model
    prediction = model.predict(input_vector)
    
    # Determine result
    result = "The news is REAL." if prediction[0] == 1 else "The news is FAKE."
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)