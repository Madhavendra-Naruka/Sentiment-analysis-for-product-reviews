# funcs.py
import pandas as pd
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import  Pipeline

import joblib
# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Function to preprocess text data: tokenization, lowercasing, removing stopwords,
    punctuation, lemmatization using spaCy, and filtering out single-letter words and spaces.
    """
    # Process text with spaCy
    doc = nlp(text)
    
    # Tokenization, lowercasing, and lemmatization
    tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc]
    
    # Remove stopwords, punctuation, single-letter words, and spaces
    tokens = [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation and len(token) > 1 and not token.isspace()]
    
    return tokens

def json_sampling(file_name,sample_size):
    """
    Function to convert a JSON file to a CSV file with a specified sample size.
    """
    # Read JSON file
    df = pd.read_json(file_name,lines=True)
    
    # Sample data
    df_sample = df.sample(sample_size)
    
    df_sample.to_json(str(sample_size//1000)+"k_"+file_name)

def classify_sentiment(score):
    if score < -0.5:
        return 'strongly negative'
    elif -0.5 <= score < 0:
        return 'negative'
    elif score==0:
        return 'neutral'
    elif 0 <= score < 0.5:
        return 'positive'
    elif score >= 0.5:
        return 'strongly positive'

# Custom feature transformer example
class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Extract custom features from DataFrame X
        custom_features = X[[ 'subjectivity', 'score']].values
        return custom_features
    
#To create Pipelines
def train_and_evaluate_pipeline(X_train, y_train, X_test, y_test, feature_extractor, classifier):
    pipeline = Pipeline([
        ('features', feature_extractor),  
        ('clf', classifier)            
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_evaluate_and_save_pipeline(X_train, y_train, X_test, y_test, feature_extractor,classifier, f_name):
    pipeline = Pipeline([
        ('features', feature_extractor),  
        ('clf', classifier)            
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    # Save the entire pipeline (including feature extractor and classifier)
    f_name = f_name+".pkl"
    joblib.dump(pipeline, f_name)
    print("Pipeline saved as",f_name)
    return accuracy

if __name__ == "__main__":
    text="12 mg is 12 on the periodic table people! Mg for magnesium text"
    print(" ".join(preprocess_text(text)))
