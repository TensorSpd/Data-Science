# Data-Science-Nanodegree

## Project: Rio-de-Janeiro Airbnb activity (2023 and 2018)

## Installations
Ensure you have Python 3.X installed on your system. You can quickly set up the required libraries using pip:

```bash
pip install scikit-learn==1.2.1
pip install pandas==1.5.3
pip install numpy==1.23.5
pip install matplotlib==3.7.0
pip install seaborn==0.12.2
```

Consider using [Anaconda](https://www.anaconda.com/download) , a comprehensive Python distribution that comes pre-packaged with all the necessary tools and libraries for this project.

## Project Motivation
Embarking on the Udacity Data Scientist Nanodegree project, I was presented with an opportunity to leverage the CRISP-DM framework for a data science blog post. Fascinated by the intersection of data science and real-world applications, I embarked on a journey to explore Airbnb listings in Rio de Janeiro. Motivated by the desire to understand the dynamics of the hospitality market in one of the world's most iconic destinations, I sought to uncover valuable insights that could inform both travelers and hosts alike. This project provided an exciting opportunity to apply data science techniques to a domain I'm passionate about while contributing to a deeper understanding of the factors shaping Airbnb pricing and trends in Rio de Janeiro. To read more about my findings and analysis, check out my Medium blog post [here](https://medium.com/@sahanipradeep5529/dreaming-of-rio-1a9b7e92b54e).

* Data for 2023: The dataset for 2023 was obtained from Inside Airbnb, a website that provides publicly available Airbnb data for various cities. You can access the Rio de Janeiro dataset specifically [here](http://insideairbnb.com/rio-de-janeiro).
* Data for 2018: The dataset for 2018 was sourced from Kaggle, a platform for hosting datasets and machine learning competitions. The specific dataset for Airbnb listings in Rio de Janeiro can be found [here](https://www.kaggle.com/datasets/allanbruno/airbnb-rio-de-janeiro/data).


## Analysis Summary

### Part I: Most Expensive Month to Visit
We uncover a fascinating trend in pricing dynamics, with availability of rental properties significantly impacting average prices.
Seasonal patterns reveal instability at the beginning of the year, with a notable spike in average monthly prices towards year-end. The months between February and April emerge as the most affordable for booking listings.

### Part II: Neighborhood Pricing
Analysis of neighborhood pricing variations identifies São Cristóvão, Estácio, and Joá as the most expensive areas, while Mangueira emerges as the least expensive neighborhood.

### Part III: Factors Influencing Prices

2023 Dataset:
Location, number of occupants, and room type are identified as key influencers of listing prices.

2018 Dataset:
Similar trends are observed, with location variables and number of occupants playing significant roles in pricing.


## File Description
Rio-de-Janeiro.ipynb : contains the detail analysis of above information using CRISP-DM.


## RUN
To run this Jupyter notebook, navigate to its directory in your terminal or command prompt, then execute the following command:
```bash
jupyter Rio-de-Janeiro.ipynb
```
This command will open the Jupyter notebook in your default web browser, allowing you to interactively explore and run the code cells within the notebook.


## Licensing, Authors, Acknowledgements

* The data used in this project was acquired from the Inside Airbnb repository, providing publicly available Airbnb data for various cities.
* This project is part of the Data Scientist Nanodegree program offered by Udacity.
* For further information on the use of Udacity's platform and resources, please refer to the [Udacity Terms of Service](https://www.udacity.com/legal).





