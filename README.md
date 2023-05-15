# zillow_project
Predictive models for house prices in the LA area

# Project Goals

To create a predictive model for Single Family Residential houses that performs better than the baseline.

# Project Description

This project uses the Zillow dataset from 2017 for the LA, Orange, and Ventura counties. The predictive modeling is based on the number of bedrooms and bathrooms, as well as the square footage to provide an estimate for a given property's value (in 2017 dollars). 

# Project Planning

Acquire
- Retrieve data from CodeUp server to include:
    - bedrooms
    - bathrooms
    - square footage
- Data is limited to:
    - 2017 properties
    - properties that had a transaction
    - Single Family Residential homes

Preparation
- Remove unnecessary or duplicate columns (N/A)
- Rename columns
- Handle nulls
- Verify datatypes
- Visualize each variable to check for outliers and handle outliers
- Split data into train, validate, test datasets

(Pre-processing)
- Encode object variables
- Scale numeric variables

Explore
- Plot visualizations
    - Create pairplot for all variables
        - Create heatmap for feature correlation
        - Check for multicollinearity
    - Engineer additional features as necessary
- Hypothesize
- Run stats tests
- Summarize

Model
- Establish baseline
    - Property value average
- Split train, validate, and test into X_train and y_train
- Run train on different models
    - Ordinary Least Squares
    - LASSO + LARS
    - Generalized Linear Model
    - Polynomial Regression
- Check models' performance using validate
    - Adjust hyperparameters as necessary
- Final verification using test on best model

# Initial Hypothesis

Bedrooms, bathrooms, and square footage are the best predictors for house prices.

# Data Dictionary

Column Name | Description | Key
--- | --- | ---
bedrooms | Number of bedrooms | Integer
bathrooms | Number of bathrooms | Float
sqft | Square footage of property | Integer
property_value | Value of property | In dollars

# How to Reproduce

Ensure that you have a env.py file with your own username and password to connect to the CodeUp server. You will be able to run through the notebook using your own credentials.

# Key Findings



# Recommendations

