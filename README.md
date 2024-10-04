# Data Science Portfolio

Welcome to my Data Science Portfolio! This repository showcases three comprehensive projects undertaken as part of the Udacity Data Scientist Nanodegree program. Each project demonstrates my proficiency in data analysis, machine learning, and the application of data science methodologies to solve real-world problems.

## **Projects**

### **1. [Rio de Janeiro Airbnb Activity (2023 and 2018)](https://medium.com/@sahanipradeep5529/dreaming-of-rio-1a9b7e92b54e)**

- **Motivation**: This project focuses on exploring Airbnb listings in Rio de Janeiro, analyzing pricing trends, availability, and neighborhood attractiveness. The project compares datasets from 2018 and 2023 to derive insights for travelers and hosts.
  
- **Data Sources**: 
  - **2023 Data**: [Inside Airbnb](http://insideairbnb.com/rio-de-janeiro)
  - **2018 Data**: [Kaggle](https://www.kaggle.com/datasets/allanbruno/airbnb-rio-de-janeiro/data)

- **Key Insights**:
  - **Seasonal Pricing Trends**: February to April are the most affordable months to book Airbnb listings, with prices peaking towards the end of the year.
  - **Neighborhood Analysis**: São Cristóvão, Estácio, and Joá are the most expensive neighborhoods, while Mangueira offers the most affordable options.
  - **Price Determinants**: Location, room type, and number of occupants are major influencers on Airbnb pricing in both datasets.

- **Technologies**: Python, Pandas, Seaborn, Matplotlib, Scikit-learn.

- **Read More**: [Full Medium Post](https://medium.com/@sahanipradeep5529/dreaming-of-rio-1a9b7e92b54e)

---

### **2. [Disaster Response Pipeline](https://github.com/yourusername/disaster-response-pipeline)**

- **Motivation**: This project involves building a machine learning pipeline to classify disaster-related messages into categories that could assist emergency response teams. The project applies Natural Language Processing (NLP) to streamline communication during disaster situations.

- **Components**:
  - **ETL Pipeline**: Extract, clean, and preprocess data.
  - **Machine Learning Pipeline**: Train a multi-output classifier to predict the categories of messages (e.g., food, water, shelter).
  - **Web Application**: A Flask-powered web interface for users to input messages and receive classification results.

- **Technologies**: Python, Flask, Pandas, Scikit-learn, SQLite, Natural Language Toolkit (NLTK).

- **Running the Web App**:
  1. Execute `python run.py` to launch the Flask app.
  2. Open the web browser and access the provided URL to classify disaster messages.

---

### **3. [Arvato - Customer Segmentation and Predictive Modeling for Mail-Order Campaign](https://medium.com/@sahanipradeep5529/leveraging-data-science-for-targeted-marketing-insights-from-arvatos-customer-segmentation-31a688ca58dd)**

- **Motivation**: This capstone project for Arvato Financial Services focuses on customer segmentation and predictive modeling for targeted marketing campaigns. By analyzing demographic data, the goal is to identify high-value customers for future marketing efforts.

- **Data**: Private datasets from Arvato Financial Services used for educational purposes in the Udacity Data Science Nanodegree.

- **Project Highlights**:
  - **Customer Segmentation**: Applied PCA and K-means clustering to identify distinct customer groups.
  - **Predictive Modeling**: Built an XGBoost model to predict customer responses to marketing campaigns.
  - **Results**: Achieved high ROC AUC scores, reflecting the model’s ability to identify potential customers for targeted marketing.

- **Technologies**: Python, Pandas, Seaborn, Matplotlib, XGBoost, Scikit-learn.

- **Read More**: [Full Medium Post](https://medium.com/@sahanipradeep5529/leveraging-data-science-for-targeted-marketing-insights-from-arvatos-customer-segmentation-31a688ca58dd)

---

## **Installation and Setup**

### **Environment Setup**
Ensure you have Python 3.x installed. The following libraries are required for all projects:

```bash
pip install scikit-learn==1.2.1 pandas==1.5.3 numpy==1.23.5 matplotlib==3.7.0 seaborn==0.12.2 flask==3.0.3 xgboost==2.0.3
```
## **How to Run Each Project**

**1. Rio de Janeiro Airbnb Activity:**
```bash
jupyter notebook Rio-de-Janeiro.ipynb
```
**2. Disaster Response Pipeline:**
- **Run the ETL pipeline to clean and preprocess the data:**
```bash
python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
- **Train the machine learning model:**
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
- **Run the web app:**
```bash
python run.py
```
**3. Arvato Customer Segmentation:**
```bash
jupyter notebook Arvato_Project.ipynb
```

##**Key Learnings and Techniques**
Throughout these projects, key skills developed include:

- **Data Wrangling:** Cleaning and transforming data for machine learning models.
- **Exploratory Data Analysis (EDA):** Visualizing and interpreting trends and insights from data.
- **Model Building:** Implementing supervised learning techniques like classification, clustering, PCA, and XGBoost.
- **Natural Language Processing (NLP):** Text processing and classification for disaster-related messages.
- **Web Development:** Using Flask to build web interfaces for interactive machine learning applications.
- **Model Evaluation:** Using metrics like ROC AUC to assess model performance.

## **Licensing, Authors, and Acknowledgements**
**Data Sources:**
- **Airbnb Dataset:**
Inside Airbnb (2023)
Kaggle (2018)

- **Disaster Response Data:** Appen (formerly Figure 8).

- **Arvato Dataset:** Private datasets used for educational purposes in the Udacity Data Science Nanodegree.

## **Acknowledgements:**
- This repository is part of the Udacity Data Science Nanodegree program. Special thanks to Udacity and the instructors for their guidance. For further information on the use of Udacity's platform and resources, please refer to the [Udacity Terms of Service](https://www.udacity.com/legal)
