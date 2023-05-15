import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from env import get_db_url

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

import os

link = os.getcwd()+'/'

def acquire_zillow():
    '''Retrieves the data from zillow database on CodeUp server.
    
    Returns bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, and taxvaluedollarcnt for single family residential houses. Only includes properties that had a transaction in 2017.'''

    #Obtain filepath to connect to zillow db on CodeUp 
    z = get_db_url('zillow')

    # Checking to see if file exists in local directory. 

    if os.path.exists(link + 'zillow.csv'):
        zillow = pd.read_csv('zillow.csv')
        return zillow

    # Write to a local csv file if it doesn't exist. Includes query for requested data for Single Family Residential households.

    else:
        zillow = pd.read_sql('''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, fips FROM properties_2017
	LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    RIGHT JOIN predictions_2017 USING (parcelid)
    WHERE propertylandusedesc = 'Single Family Residential';''', z)
        zillow.to_csv('zillow.csv', index=False)
        return zillow

def prep_zillow(zillow=acquire_zillow()):
    '''Cleans zillow data to eliminate nulls and removes data outliers based on comments below.'''

    #Rename columns
    zillow = zillow.rename(columns={'bedroomcnt': 'bedrooms'
                       ,'bathroomcnt': 'bathrooms'
                       ,'calculatedfinishedsquarefeet': 'sqft'
                       ,'taxvaluedollarcnt': 'prop_value'
                      })

    #Replace numerical values in county with their respective strings.
    #
    #zillow.county = zillow.county.replace([6037.0, 6059.0, 6111.0], ['LA', 'Orange', 'Ventura'])

    #Remove rows with NaNs from dataset
    zillow = zillow.dropna()

    #Handling outliers
    zillow.bedrooms = zillow.bedrooms[zillow.bedrooms.between(1,7)] #limiting properties to those between 1 and 7 bedrooms
    zillow.bathrooms = zillow.bathrooms[zillow.bathrooms.between(1,7)] #limiting properties to those between 1 and 7 bathrooms
    zillow.sqft = zillow.sqft[zillow.sqft.between(320, 5000)] #removing top 1% of highest square footage
    zillow.prop_value = zillow.prop_value[zillow.prop_value.between(zillow.prop_value.quantile(.01),zillow.prop_value.quantile(.99))] #removing top and bottom 1% of value
    zillow.yearbuilt = zillow.yearbuilt[zillow.yearbuilt >= 1875] #removing properties built before 1875

    #Removing those newly-created nulls as well
    zillow = zillow.dropna()

    #Adjusting dtypes so dataframe is easier to read
    zillow[['bedrooms', 'sqft', 'prop_value', 'yearbuilt']] = zillow[['bedrooms', 'sqft', 'prop_value', 'yearbuilt']].astype(int)

    #Encoding 'county' for future modeling
#    dummy_df = pd.get_dummies(zillow[['county']], dummy_na=False, drop_first=[True])
#    dummy_df.head()
#    zillow = pd.concat([zillow, dummy_df], axis=1)

    zillow = zillow[zillow.fips == 6037.0]

    return zillow    

def split_data(df):
    '''
    Takes in a DataFrame and returns train, validate, and test DataFrames; stratifies on target argument.
    
    Train, Validate, Test split is: 60%, 20%, 20% of input dataset, respectively.
    '''
    # First round of split (train+validate and test)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # Second round of split (train and validate)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123)
    return train, validate, test

def wrangle_zillow():
    '''Combines acquisition, preparation, and split of zillow data into one function.'''

    zillow = acquire_zillow()

    zillow = prep_zillow()

    train, validate, test = split_data(zillow)

    return train, validate, test

def scale_data(train, 
               validate, 
               test, 
               cols):
    
    #make copies for scaling
    train_scaled = train.copy() #Ah, making a copy of the df and then overwriting the data in .transform()
    validate_scaled = test.copy()
    test_scaled = test.copy()

    #scale them!
    #make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    #fit the thing
    scaler.fit(train[cols])

    #use the thing
    train_scaled[cols] = scaler.transform(train[cols])
    validate_scaled[cols] = scaler.transform(validate[cols])
    test_scaled[cols] = scaler.transform(test[cols])
    
    return train_scaled, validate_scaled, test_scaled