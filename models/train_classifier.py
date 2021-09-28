# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:29:34 2021

@author: snagowsv
"""

import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MultiLabelBinarizer
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target Variables
    """
    
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    #engine.execute("SELECT * FROM InsertDatabaseName").fetchall()
    df = pd.read_sql_table("disaster_messages", engine)
    X = df["message"]
    Y = df.loc[:,~df.columns.isin(["id", "message" ,"original", "genre"])]
    return X,Y
    


def tokenize(text):
    """
    Function to tokenize text.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Creates Model with Pipeline and GridSearch
    
    Returns: cv Randomforest Classifier
    """
    
    pipeline =  pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = parameters = {#'clf__estimator__n_estimators': [10, 20],
                           'vect__ngram_range': ((1, 1), (1, 2)),
                           'vect__max_df': (0.5, 0.75, 1.0)}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return(cv)


def evaluate_model(model, X_test, Y_test):
    
    def test_model(y_test, y_pred):
        """
        Function to iterate through columns and call sklearn classification report on each.
        """
        for index, column in enumerate(y_test):
            print(column, classification_report(y_test[column], y_pred[:, index]))
            
    
    Y_pred = model.predict(X_test)
    test_model(Y_test, Y_pred )


def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()