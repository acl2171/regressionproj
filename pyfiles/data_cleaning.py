"""

DOCSTRING: This module contains data pre-processing functions which clean the original dataset and remove irrelevant features. 

"""

def clean_orig_dataset(dataframe):
    """This function removes irrelevant columns from the original dataset."""
    dataframe.drop(['OBJECTID', 'TRACT', 'STATE_FIPS','BLKGRP', 'DV05', 'DV06', 'DVWEEKEND', 'DVWEEKNIGHT', 'DVENDNIGHT', 'METRODUMMY', 'PUBHOUSINGDUMMY', 'AVGADW0506','MPDDIS06','UIWEEKNIGH','UIWEEKEND','UIWEEKNIGH','UIENDNIGHT','MPDWEEKNIG', 'ADW06','AVGUIDISOR','AGE_UNDER5','MPDWENDNIG', 'MPDWEEKEND','UIDIS05', 'UIDIS06', 'MPDDIS05','SHAPE_LENG','SHAPE_AREA', 'POP2000', 'POP00_SQMI', 'ADW05', 'STFID','ADW00', 'ADW01', 'BUSSTOP_SQMI'], axis = 1, inplace = True)
    return dataframe


def recursive_feature_elimination(X, y):
    """This function removes the features that were selected from backwards selection process."""
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    return ranking