import pandas as pd
import numpy as np
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv(r"C:\Users\aaroh\Downloads\email spam detection\spam.csv\spam.csv", encoding="ISO-8859-1", low_memory=False)
df = df[['v1', 'v2']]  # Selecting only necessary columns
df.columns = ['label', 'message']

# Convert labels to numerical (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def preprocess_text(text):
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['message'] = df['message'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Train the model
vectorizer = TfidfVectorizer()
model = MultinomialNB()
pipeline = make_pipeline(vectorizer, model)
pipeline.fit(X_train, y_train)

# Save the model & vectorizer
with open("spam_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("✅ Model trained and saved successfully!")
