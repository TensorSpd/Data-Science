import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals

    # Calculate the counts of messages for each genre
    genre_counts = df.groupby('genre').count()['message']
    # Extract genre names from the index of genre_counts
    genre_names = list(genre_counts.index)

    # Extract the response categories (excluding the first four columns) from the DataFrame
    category = df[df.columns[4:]]
    # Count the occurrences of each response category
    response_count = category.apply(lambda x: (x == 1).sum()).sort_values(ascending=False)[:12]
    # Get the names of the top 10 response categories
    category_cols = response_count.index.tolist()

    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,  # genre names
                    values=genre_counts,  # genre each class counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },
        {
            'data': [
                Bar(
                    x=category_cols,
                    y=response_count,
                    marker=dict(
                        color=['rgb(55, 83, 109)', 'rgb(100, 200, 150)', 'rgb(200, 100, 50)', 'rgb(150, 50, 200)',
                               'rgb(50, 150, 200)', 'rgb(200, 50, 150)', 'rgb(100, 200, 50)', 'rgb(50, 100, 200)',
                               'rgb(150, 200, 50)', 'rgb(200, 150, 100)', 'rgb(50, 100, 150)', 'rgb(150, 100, 50)'])
                )
            ],

            'layout': {
                'title': 'Top 12 Responses',
                'yaxis': {
                    'title': "Count",
                    'gridcolor': 'rgb(200, 200, 200)'  # Add gridlines
                },
                'xaxis': {
                    'title': "Response",
                    'tickangle': -45,  # Rotate x-axis labels for better readability
                    'tickfont': dict(size=10)  # Adjust font size of x-axis labels
                },
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Make plot background transparent
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',  # Make paper background transparent
                'font': {
                    'color': 'rgb(100, 100, 100)',  # Change font color
                    'size': 12  # Adjust font size
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    columns_to_include = [col for col in df.columns[4:] if col != 'child_alone']
    classification_results = dict(zip(columns_to_include, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
