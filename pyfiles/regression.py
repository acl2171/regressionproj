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

def scale_dataset(dataframe):
    scaler = StandardScaler()
    scaler = scaler.fit(dataframe)
    then = scaler.transform(dataframe)
    scaled_df = pd.DataFrame(then, columns = dataframe.columns)
    return scaled_df

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

def run_model(dataframe, y):
    X = dataframe
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())