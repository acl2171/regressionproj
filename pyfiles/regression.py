"""
DOCSTRING: This module contains all the functions relevant to our regression process, including feature selection and model analysis."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import statsmodels.stats.outliers_influence as smd

def scale_dataset(x_dataframe):
    scaler = StandardScaler()
    scaler = scaler.fit_transform(x_dataframe)
    scaled_df = pd.DataFrame(scaler, columns = x_dataframe.columns)
    return scaled_df

def cooksd(dataframe, columns_list):
    """This function returns a dataframe of the variables and their cook's D distance."""
    new_variables_dict = {}
    for variable in columns_list:
        X = np.array(dataframe[variable]).reshape(-1,1)
        y = dataframe.AVGDV
        lr = LinearRegression(fit_intercept= False)
        lr.fit(X,y)
        mod = sm.OLS(y, X)
        res = mod.fit()
        cooksd = smd.OLSInfluence(res)
        new_variables_dict.update({variable: cooksd.cooks_distance[0]})
    df = pd.DataFrame(new_variables_dict)
    return df

def dropping_outliers(columns_list, df1, cooksddf):
    """This function drops rows with outlier values according to Cook's distance above 3 * the mean."""
    for column in columns_list:
        for value in cooksddf[column]:
            if value > (3 * df1[column].mean()):
                df1 = df1[df1[column] != value]
    return df1

def recursive_feature_elimination(X, y, number):
    """This function runs a recursive feature elimination for feature selection based on the coefficients of a linear 
    regression model, and returns a dataframe of the resulting features."""
    
    # run rfe
    lr = LinearRegression(fit_intercept=True)
    rfe = RFE(estimator=lr, n_features_to_select=number, step=1)
    rfe.fit(X, y)
    features = rfe.support_
    
    # Create dataframe from rfe results
    new =[]
    for i in range(0, len(X.columns)):
        if features[i] == True:
            new.append(X.columns[i])
    recursive_df = X.loc[:, new]
    return recursive_df

def run_lasso(cleaned_df, X, y):
    """This function runs a lasso regression and outputs a dataframe with the resulting variables.
    param_orig_dataframe: The original dataframe before scaling
    dataframe: the scaled dataframe
    y: the dependent variable. """
    clf = Lasso(alpha = 0.10, normalize = True, positive = True)
    clf.fit(X, y)
    coef = clf.coef_
    cols = X.columns
    feat1 = pd.DataFrame(zip(cols, coef))
    feat1 = feat1[(feat1[1] > 0.0)]
    feat1_cols = list(feat1[0])
    feat1_df = X.loc[:,feat1_cols]
    return feat1_df

def lasso_for_predict(dataframe, y):
    clf = Lasso()
    clf.fit(dataframe, y)
    return clf

def run_model(dataframe, y):
    """This function runs an OLS model given a dataframe and a dependent variable, and returns as summary of the results."""
    X = dataframe
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res