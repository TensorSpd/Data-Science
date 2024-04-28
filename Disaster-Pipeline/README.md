# Disaster Response Pipeline

## Installations
Ensure you have Python 3.X installed on your system. You can quickly set up the required libraries using pip:

```bash
pip install scikit-learn==1.2.1
pip install pandas==1.5.3
pip install flask==3.0.3
```

## Project Overview

This project focuses on building a disaster response pipeline to analyze messages and categorize them into different categories in order to aid emergency response teams during disasters. The pipeline consists of two main components: an ETL (Extract, Transform, Load) process to clean and preprocess the data, and a machine learning pipeline to train a model to classify messages.

## File Structure

```bash
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model


- README.md
```


## Running Python Scripts
To execute the Python scripts, navigate to the project directory and use the following commands:


* Process Data Script
```bash
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
- This script processes the raw message and category data, cleans and merges them, and stores the cleaned data in an SQLite database.


* Train Classifier Script
```bash
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
``` 
- This script trains a machine learning model using the cleaned data from the database and saves the trained model as a pickle file.


 * Running the Web App

To run the Flask web app, follow these steps:

- Open a new terminal window.
- Navigate to the project directory.
- Run the following command:

```bash
python run.py
```

Once the server is running without errors, open a web browser and go to the URL provided by the Flask app.


## Additional Information

The web app provides a user interface for classifying new messages using the trained model.
Ensure that the required dependencies are installed before running the scripts or the web app.


## Licensing, Authors, Acknowledgements

* The data used in this project was acquired from Appen (formerly Figure 8).
* This project is part of the Data Scientist Nanodegree program offered by Udacity.
* For further information on the use of Udacity's platform and resources, please refer to the Udacity Terms of Service.
