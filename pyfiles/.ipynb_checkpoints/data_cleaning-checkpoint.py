"""

DOCSTRING: This module contains data pre-processing functions which clean the original dataset and remove irrelevant features. It also contains functions to develop visualizations.

"""
import seaborn as sns
import numpy as np

def clean_orig_dataset(dataframe):
    """This function creates creates a new column and removes columns that are not of interest from the original 
    dataset."""
    # Create new column: alcohol availability per popn_square mile
    dataframe['AA_POP'] = (dataframe['STORE_BG'] + dataframe['NIGHT_BG'] + dataframe['TAVERN_BG'] + dataframe['REST_BG'])/dataframe['POP2004']
    
    #drop columns
    dataframe.drop(['AGE_UNDER5', 'SQMI','AGE_5_17', 'AGE_18_21', 'AGE_22_29', 'AGE_30_39', 'AGE_40_49', 'AGE_50_64', 'AGE_65_UP','STORE_BG', 'NIGHT_BG','TAVERN_BG','REST_BG','POP04_SQMI','OBJECTID', 'TRACT', 'STATE_FIPS','BLKGRP', 'DV05', 'DV06', 'DVWEEKEND', 'DVWEEKNIGHT', 'DVENDNIGHT', 'METRODUMMY', 'PUBHOUSINGDUMMY', 'AVGADW0506','MPDDIS06','UIWEEKNIGH','UIWEEKEND','UIWEEKNIGH','UIENDNIGHT','MPDWEEKNIG', 'ADW06','AVGUIDISOR','AGE_UNDER5','MPDWENDNIG', 'MPDWEEKEND','UIDIS05', 'UIDIS06', 'MPDDIS05','SHAPE_LENG','SHAPE_AREA', 'POP2000', 'POP2004','POP00_SQMI', 'PCTNOTALLOW', 'AVGMPDDIS','AVGADW0001','ADWWEEKEND','ADWWEEKNIGHT','INVMILEDIST','ADW05', 'STFID','ADW00', 'ADW01', 'BUSSTOP_SQMI', 'ASIAN', 'HISPANIC', 'BLACK','ARR_0506','ADWNIGHTS', 'WHITE','MALE'], axis = 1, inplace = True)

    return dataframe


def scatter(X, y):
    sns.set_context('talk')
    return sns.scatterplot(X, y);

def barplot(df):
    viz = df.loc[:,['DVWEEKEND', 'DVENDNIGHT', 'DVWEEKNIGHT']]
    viz2 = []
    for column in viz.columns:
        viz2.append(viz[column].sum())
    viz_final = pd.DataFrame(zip(viz.columns, viz2))
    return sns.barplot(x = viz_final[0], y = viz_final[1]); 
    