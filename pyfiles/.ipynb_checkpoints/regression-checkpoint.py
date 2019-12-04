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

def scale_dataset(dataframe):
    scaler = StandardScaler()
    scaler = scaler.fit(dataframe)
    then = scaler.transform(dataframe)
    scaled_df = pd.DataFrame(then, columns = dataframe.columns)
    return scaled_df

def recursive_feature_elimination(scaled_dataframe, dataframe, y):
    """This function runs a recursive feature elimination for feature selection based on the coefficients of a linear 
    regression model, and returns a dataframe of the resulting features."""
    
    # run rfe
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=6, step=1)
    rfe.fit(scaled_dataframe, y)
    features = rfe.support_
    
    # Create dataframe from rfe results
    new =[]
    for i in range(0, len(dataframe.columns)):
        if features[i] == True:
            new.append(dataframe.columns[i])
    recursive_df = dataframe.loc[:, new]
    return recursive_df

def run_lasso(orig_dataframe, dataframe, y):
    clf = Lasso()
    clf.fit(dataframe, y)
    coef = clf.coef_
    cols = dataframe.columns
    feat1 = pd.DataFrame(zip(cols, coef))
    feat1 = feat1[feat1[1] > 0]
    feat1_cols = list(feat1[0])
    feat1_df = orig_dataframe.loc[:,feat1_cols]
    return feat1_df

def lasso_remove_multicollinearity(lasso_df):
    """This function removes variables with high multicollinearity from the results of the lasso feature selection."""
    lasso_df.drop(['AVGADW0001', 'AGE_5_17', 'ARR_0506', 'VACANT', 'ADWWEEKEND', 'ADWWEEKNIGHT'], axis = 1, inplace = True)
    return lasso_df


def lasso_for_predict(dataframe, y):
    clf = Lasso()
    clf.fit(dataframe, y)
    return clf

def run_model(dataframe, y):
    X = dataframe
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())