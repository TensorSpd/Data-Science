# import libraries
import sys
import nltk

import pandas as pd

import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import joblib

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Parameters:
    database_filepath (str): Filepath to the SQLite database.

    Returns:
    X (pandas.Series): Series containing the messages.
    Y (pandas.DataFrame): DataFrame containing the categories.
    category_names (list): List of category names.
    """
    # Load data from the specified database file
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)

    # Extract messages from the DataFrame
    X = df['message']

    # Define columns to exclude from categories
    exclude_columns = ['id', 'message', 'original', 'genre']

    # Create a mask to filter out excluded columns
    mask = ~df.columns.isin(exclude_columns)

    # Select only category columns using the mask
    Y = df.loc[:, mask]

    # Extract category names
    category_names = list(Y.columns)

    return X, Y, category_names


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract whether the starting word of a sentence is a verb."""

    def starting_verb(self, text):
        # Tokenize the text into sentences
        sentence_list = nltk.sent_tokenize(text)

        # Iterate through each sentence
        for sentence in sentence_list:
            tokens = tokenize(sentence)
            if not tokens:  # Check if tokens list is empty
                continue  # Skip empty sentences
            pos_tags = nltk.pos_tag(tokens)
            if not pos_tags:  # Check if pos_tags list is empty
                continue  # Skip sentences with no pos_tags
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return True
        return False

    def fit(self, x, y=None):
        """Fit method required for sklearn pipeline compatibility."""
        return self

    def transform(self, X):
        """Transform method required for sklearn pipeline compatibility."""
        # Apply the starting_verb function to each message in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """
    Tokenize the input text by performing the following steps:
    1. Replace non-alphanumeric characters with spaces.
    2. Tokenize the text into individual words.
    3. Lemmatize each word to its base form.
    4. Convert each word to lowercase and remove leading/trailing spaces.

    Args:
    text (str): Input text to be tokenized.

    Returns:
    list: List of clean tokens extracted from the input text.
    """
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word and convert to lowercase
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """Builds and returns a GridSearchCV object for multi-output classification."""

    # Define pipeline with feature union and classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_leaf': [1, 2, 4]
    }

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Prints the evaluation metrics for the model."""

    # Predict labels
    y_pred = model.predict(X_test)

    # Print the metrics for each category
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    Parameters:
    model (object): The trained model object to be saved.
    model_filepath (str): The filepath where the model will be saved.
    """
    try:
        # Save the model as a pickle file
        joblib.dump(model, model_filepath)
        print("Model saved successfully as:", model_filepath)
    except Exception as e:
        print("Error occurred while saving the model:", str(e))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
