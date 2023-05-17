import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

def baseline_mean_errors(y):
    '''Computes the RMSE for the baseline model of a given target variable.'''

    #Set the baseline
    baseline = y.mean()

    #SSE for baseline
    SSE_baseline = ((np.full(len(y), baseline) - y) ** 2).sum() #Creates an array with the mean value that is the length of the df, squares the residuals, and sums it

    #MSE for baseline
    MSE_baseline = SSE_baseline / len(y)

    #RMSE for baseline
    RMSE_baseline = MSE_baseline ** 0.5
    print(f'RMSE for baseline = {RMSE_baseline}')

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def create_models(X_train, X_validate, features, target, target_validate):
    '''Takes in train, validate, list of features to send in, train target, and validate target, and runs both through
    all four models (MLR, LASSO, GLM, Poly) and provides their respective RMSE and R2 scores.'''

    #Assign train and validate to the dataset+features to send in
    train = X_train[features]
    validate = X_validate[features]

    #Setting baseline and calculating RMSE for baseline

    baseline = target.mean()

    RMSE_baseline = ((((np.full(len(target), baseline) - target) ** 2).sum())/len(target)) ** 0.5

    #Multiple Linear Regression

    lr = LinearRegression(normalize=True) #Initiate Linear Regression
    
    lr.fit(train, target) #Fit LR on train only

    #RMSE scores for train and validate
    train_rmse_lr = mean_squared_error(target, lr.predict(train), squared=False)
    train_rmse_lr_val = mean_squared_error(target_validate, lr.predict(validate), squared=False)

    #R2 scores for train and validate
    train_r2_lr = r2_score(target, lr.predict(train))
    train_r2_lr_val = r2_score(target_validate, lr.predict(validate))

    #LASSO LARS Regression

    lars = LassoLars(alpha=2.0) #Initiate LASSO LARS Regression

    lars.fit(train, target) #Fit LARS on train only

    #RMSE scores for train and validate
    train_rmse_lars = mean_squared_error(target, lars.predict(train), squared=False)
    train_rmse_lars_val = mean_squared_error(target_validate, lars.predict(validate), squared=False)

    #R2 scores for train and validate
    train_r2_lars = r2_score(target, lars.predict(train))
    train_r2_lars_val = r2_score(target_validate, lars.predict(validate))

    #Generalized Linear Model 

    glm = TweedieRegressor(power=1, alpha=0) #Initiate GLM regression. Power=1 = Poisson distribution

    glm.fit(train, target) #Fit on train only

    #RMSE scores for train and validate
    train_rmse_glm = mean_squared_error(target, glm.predict(train), squared=False)
    train_rmse_glm_val = mean_squared_error(target_validate, glm.predict(validate), squared=False)

    #R2 scores for train and validate
    train_r2_glm = r2_score(target, glm.predict(train))
    train_r2_glm_val = r2_score(target_validate, glm.predict(validate))

    #Polynomial Regression

    pf = PolynomialFeatures(degree=3) #Initialize Polynomial 

    train_degree2 = pf.fit_transform(train) #Fit and transform on train only
    validate_degree2 = pf.transform(validate) #Transform validate

    lr2 = LinearRegression(normalize=True) #Initiate Linear Regression for modeling on transformed variables

    lr2.fit(train_degree2, target) #Fit 

    #Finding RMSE and R2 values for both train and validate sets.
    train_rmse_poly = mean_squared_error(target, lr2.predict(train_degree2), squared=False)
    train_rmse_poly_val = mean_squared_error(target_validate, lr2.predict(validate_degree2), squared=False)

    train_r2_poly = r2_score(target, lr2.predict(train_degree2))
    train_r2_poly_val = r2_score(target_validate, lr2.predict(validate_degree2))

    #Create dataframe that provides RMSE and R2 scores for train and validate across all models.

    metrics_df = pd.DataFrame({'model': ['baseline', 'MLR', 'LASSO', 'GLM', 'Poly'],
                               'rmse_train': [RMSE_baseline, train_rmse_lr, train_rmse_lars, train_rmse_glm, train_rmse_poly],
                               'rmse_validate': [RMSE_baseline, train_rmse_lr_val, train_rmse_lars_val, train_rmse_glm_val, train_rmse_poly_val],
                               'r2_train': [0, train_r2_lr, train_r2_lars, train_r2_glm, train_r2_poly],
                               'r2_validate': [0, train_r2_lr_val, train_r2_lars_val, train_r2_glm_val, train_r2_poly_val]})
    
    return metrics_df

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

def test_model(train_scaled, test_scaled, features, target_train, target_test):
    '''Takes in train_scaled, test_scaled, list of features to send in, target_train and target test, and produces
    RMSE and R2 scores for the test dataset.'''

    #Initialize the Polynomial Features
    pf = PolynomialFeatures(degree=3)

    #Fit and transform on train only, transform the test dataset based on the fit on train.

    train_degree2 = pf.fit_transform(train_scaled[features])
    test_degree2 = pf.transform(test_scaled[features])

    #Initialize Linear Regression.
    lr2 = LinearRegression(normalize=True)

    #Fit Linear regression based on the transformed train dataset.
    lr2.fit(train_degree2, target_train)

    #RMSE score for test.
    rmse_test = mean_squared_error(target_test, lr2.predict(test_degree2), squared=False)
    
    #R2 score for test.
    r2_test = r2_score(target_test, lr2.predict(test_degree2))

    #Creates dataframe to show RMSE and R2 scores for test.
    metrics_df = pd.DataFrame({'model': ['Poly'],
                               'rmse_test': [rmse_test],
                               'r2_test': [r2_test]})
    
    return metrics_df

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

def get_act_pred_viz(train_scaled, test_scaled, features, target_train, target_test):
    '''Takes in train_scaled, test_scaled, list of features to send in, target_train and target test, and produces
    a regression plot for actual vs predicted based on Polynomial model.'''

    #Initialize the Polynomial Features
    pf = PolynomialFeatures(degree=3)

    #Fit and transform on train only, transform the test dataset based on the fit on train.

    train_degree2 = pf.fit_transform(train_scaled[features])
    test_degree2 = pf.transform(test_scaled[features])

    #Initialize Linear Regression.
    lr2 = LinearRegression(normalize=True)

    #Fit Linear regression based on the transformed train dataset.
    lr2.fit(train_degree2, target_train)

    #Assign variables to put into sns.regplot's x and y values.
    act = target_test
    pred = lr2.predict(test_degree2)

    #Creates regression plot for actual vs. predicted
    sns.regplot(x=act, y=pred, line_kws={'color':'red'}, scatter_kws={'alpha': 0.1})
    plt.xlabel('Actual Property Value, in Millions')
    plt.ylabel('Predicted Property Value, in Millions')
    plt.title('Visualization of Polynomial Model, Actual vs Predicted')
    plt.axhline(target_train.mean(), c='black', linestyle='--') #Creates black dashed line that shows baseline
    plt.text(x=2500000, y=500000, s='Baseline')
    plt.show()
