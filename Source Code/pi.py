import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pickle
from scipy.sparse import csr_matrix

# Load the CSV file into a DataFrame
df = pd.read_csv("SpamDetection8thSem\dataset\spam.csv", encoding='latin-1')

# Preprocess the text data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    a=str(text)
    tokens = word_tokenize(a.lower())
    tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['v2'].apply(preprocess_text)

# Split the data into features (X) and labels (y)
X = df['processed_text']
y = df['v1'].map({'ham': 0, 'spam': 1})

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier using TF-IDF vectorization
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', MultinomialNB()),
])
pipeline.fit(X_train, y_train)

# Save TF-IDF vectorizer and model as pickle files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
