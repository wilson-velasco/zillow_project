import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_residuals(y, yhat):
    '''Plots the residuals for a given target variable against the target variable.'''

    sns.scatterplot(x=y, y=(yhat - y))
    plt.title('Residuals for Target Variable')
    plt.xlabel(f'Target Variable')
    plt.ylabel('Residual Values')
    plt.show()

def regression_errors(y, yhat):
    '''Takes in an actual value and a predicted value and outputs summary of regression errors.
    
    Regression errors included: SSE, ESS, TSS, MSE, RMSE.'''

    #SSE for model

    SSE_model = ((yhat - y) ** 2).sum()
    print(f'SSE for model = {SSE_model}')

    #ESS for model

    ESS_model = sum((yhat - y.mean())**2)
    print(f'ESS for model = {ESS_model}')

    #TSS for model

    TSS_model = ESS_model + SSE_model
    print(f'TSS for model = {TSS_model}')

    #MSE for model

    MSE_model = SSE_model / len(y)
    print(f'MSE for model = {MSE_model}')

    #RMSE for model

    RMSE_model = MSE_model ** 0.5
    print(f'RMSE for model = {RMSE_model}')

def baseline_mean_errors(y):
    '''Computes the SSE, MSE, and RMSE for the baseline model of a given target variable.'''

    #Set the baseline
    baseline = y.mean()

    #SSE for baseline

    SSE_baseline = ((np.full(len(y), baseline) - y) ** 2).sum()
    print(f'SSE for baseline = {SSE_baseline}')

    #MSE for baseline

    MSE_baseline = SSE_baseline / len(y)
    print(f'MSE for baseline = {MSE_baseline}')

    #RMSE for baseline

    RMSE_baseline = MSE_baseline ** 0.5
    print(f'RMSE for baseline = {RMSE_baseline}')

def better_than_baseline(y, yhat):
    '''Returns SSE values for actual vs. predicted, states whether the predictive model performed better, and returns True if so.'''

    #Set the baseline
    baseline = y.mean()

    SSE_model = ((yhat - y) ** 2).sum()
    SSE_baseline = ((np.full(len(y), baseline) - y) ** 2).sum()

    print(f'The SSE for the model is: {SSE_model}')
    print(f'The SSE for the baseline is: {SSE_baseline}')

    if SSE_model < SSE_baseline:
        print('The model outperforms the baseline.')
        return True
    else:
        print('The model did not perform better than the baseline.')
        return False

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def create_models(X_train, X_validate, features, target, target_validate):

    train = X_train[features]
    validate = X_validate[features]

    #Setting baseline and calculating RMSE for baseline

    baseline = target.mean()

    RMSE_baseline = ((((np.full(len(target), baseline) - target) ** 2).sum())/len(target)) ** 0.5

    #Multiple Linear Regression

    lr = LinearRegression(normalize=True)
    
    lr.fit(train, target)

    train_rmse_lr = mean_squared_error(target, lr.predict(train), squared=False)
    train_rmse_lr_val = mean_squared_error(target_validate, lr.predict(validate), squared=False)

    #LASSO LARS Regression

    lars = LassoLars(alpha=2.0)

    lars.fit(train, target)

    train_rmse_lars = mean_squared_error(target, lars.predict(train), squared=False)
    train_rmse_lars_val = mean_squared_error(target_validate, lars.predict(validate), squared=False)

    #Generalized Linear Model 

    glm = TweedieRegressor(power=1, alpha=0)

    glm.fit(train, target)

    train_rmse_glm = mean_squared_error(target, glm.predict(train), squared=False)
    train_rmse_glm_val = mean_squared_error(target_validate, glm.predict(validate), squared=False)

    #Polynomial Regression

    pf = PolynomialFeatures(degree=2)

    train_degree2 = pf.fit_transform(train)
    validate_degree2 = pf.transform(validate)

    lr2 = LinearRegression(normalize=True)

    lr2.fit(train_degree2, target)

    train_rmse_poly = mean_squared_error(target, lr2.predict(train_degree2), squared=False)


    train_rmse_poly_val = mean_squared_error(target_validate, lr2.predict(validate_degree2), squared=False)

    metrics_df = pd.DataFrame({'model': ['baseline', 'MLR', 'LASSO', 'GLM', 'Poly'],
                               'rmse_train': [RMSE_baseline, train_rmse_lr, train_rmse_lars, train_rmse_glm, train_rmse_poly],
                               'rmse_validate': [RMSE_baseline, train_rmse_lr_val, train_rmse_lars_val, train_rmse_glm_val, train_rmse_poly_val]})
    
    return metrics_df

