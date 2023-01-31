# Diamond Price Prediction
Using Data Science with Machine Learning to detect the price of diamonds using significant features given by the most linked features that are taken into consideration when evaluating price by diamond sellers.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction
This project is a part of my training at SHAI FOR AI.
In a world of speed and development, and the great expansion of technology based on artificial intelligence, machine learning and its uses in many scientific and practical fields, academic and professional, as an example professionally in financial forecasts, which we find its importance based on the correct and accurate prediction of the problem and determining the possibility of addressing it and solving it in the most accurate ways, and scientific methods and evaluating it on the best possible standards.\
Based on this introduction, I present to you my project in solving the problem of diamond price prediction, and my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning.\
Hoping to improve it gradually in the coming times.

## Dataset General info
**General info about the dataset:**

Context

This is a classic dataset contains the prices and other attributes of almost 54,000 diamonds, but in SHAI competition project we use a subset of the full data which contain only 43040 diamonds.

* Content price price in US dollars (\$326--\$18,823)

* carat weight of the diamond (0.2--5.01)

* cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

* color diamond colour, from J (worst) to D (best)

* clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

* x length in mm (0--10.74)

* y width in mm (0--58.9)

* z depth in mm (0--31.8)

* depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

* table width of top of diamond relative to widest point (43--95)

## Technologies
* Programming language: Python.
* Libraries: Numpy, Matplotlib, Pandas, Seaborn, tabulate, sklearn, xgboost. 
* Application: Jupyter Notebook.

## Setup
To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:

'''

* pip install numpy
* pip install matplotlib
* pip install pandas
* pip install seaborn
* pip install tableau
* pip install scikit-learn
* pip install xgboost

'''

To install these packages with conda run:

'''

* conda install -c anaconda numpy
* conda install -c conda-forge matplotlib
* conda install -c anaconda pandas
* conda install -c anaconda seaborn
* conda install -c conda-forge tableauserverclient
* conda install -c anaconda scikit-learn
* conda install -c anaconda py-xgboost

'''

## Features
* I present to you my project solving the problem of diamond price prediction using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it in the best possible ways and the current capabilities using Machine Learning.

### To Do:
Briefly about the process of the project work:
* Take a comprehensive view of the data contained within the data set.
* Structuring problem.
* Choosing a performance measure algorithm: here RMSE algorithm was chosen.
* Hypothesis testing: all of my hypotheses here were in the Machine Learning field.
* Preparing the work environment to deal with the data and solving the problem.
* Download the dataset.
* Do a quick look at the data structure
* Build the test set.
* Explore and display the data to get ideas from it: this stage aims to extract ideas and a deeper understanding of the data and the goal of the problem.
* Searching for correlation: correlation test resulting from merging a set of descriptors with each other, and manipulate with features.
* Data cleaning.
* Dealing with texts and categorical data.
* Build custom transformers and value converters.
* Feature standardization.
* Model selection and training: choosing the optimal model for training data, and at this stage a lot of machine learning algorithms related to Regression were tested and the analysis was based on the optimum ones: Decision Tree, Random Forest, Grandient Boosting, XGBoosting, Extra Trees, SVR, NuSVR, Bagging, Hist Gradient Boosting, Linear Regression, Lasso, Ridge, AdaBoost, SGD, Tweedie, PLS.
* Training and evaluation on the dataset.
* Better evaluation with cross-validation, and learning curve.
* Get the optimal setting of the model: setting the model best parameters using GridSearchCV, RandomizedSearchCV.
* Analyze the best model and analyze the errors.
* Evaluate the model on the test data.

## Run Example
To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset.

3. Select which cell you would like to run and show its output.

4. Run Selection/Line in Python Terminal command (Shift+Enter).

## Sources
This data was taken from SHAI competition (Diamond Price Prediction competition)\
(https://www.kaggle.com/t/0aa9c9c6994548aba1f257a94e1c59cc)
