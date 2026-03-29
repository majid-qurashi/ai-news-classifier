#!/usr/bin/env python
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model():
    print("Loading data...")
    # Read the dataset
    df = pd.read_csv('train.csv')
    
    # Combine Title and Description for better features
    # Use fillna('') to prevent NaN strings
    df['text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
    
    # Define features and labels
    X = df['text']
    y = df['Class Index']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print("Training the model (this might take a minute)...")
    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Verify performance
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the pipeline using 'with open'
    print("Saving model pipeline to model_pipeline.pkl...")
    with open('model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Success!")

if __name__ == '__main__':
    train_model()
