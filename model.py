# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset from url.csv
df = pd.read_csv('url.csv')


# Print the column names for inspection
print(df.columns)

# Replace 'column_name' with the actual column name containing labels in your dataset
label_column_name = 'type'

# Check if the label column is present
if label_column_name not in df.columns:
    raise ValueError(f"Column '{label_column_name}' not found in the dataset.")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['url'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df[label_column_name], test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Save the trained model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Save the TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
