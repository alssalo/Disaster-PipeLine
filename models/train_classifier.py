import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    This function loads data from a given database.
    
    Input:
        database_filepath: database file path
    Output:
        X: Traininng message list
        Y: Training target
        category names  
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages',engine)
    
    # define features and target
    X = df.message
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    """
    The function process the text data.
    
    Input: Text data(messages)
    
    Output: List of clean tokens 
    
    """    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    # Remove stop words
    words = [w for w in clean_tokens if w not in stopwords.words("english")]
    print(words)
        
    return words


def build_model():
    '''
    input:
        None
    output:
        cv: GridSearch model result.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    #'tfidf__norm':['l2','l1'],
    #'tfidf__smooth_idf':[True, False],
    #'vect__stop_words': ['english',None],
    'clf__estimator__max_depth' :[1, 3]
    #'clf__estimator__min_samples_leaf' : [2,3],
    }

    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data

    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    This fuction saves model file in pickle format
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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