"""
DOCSTRING: 

This module contains the functions used to collect data from Netflix (webscraping using BeautifulSoup) and the NYT Bestseller Lists (using the NYT API).

"""
from bs4 import BeautifulSoup
import pandas as pd
import itertools
import numpy as np

def get_titles_genres_from_netflix(dictionary, genres):
    """This function uses BeautifulSoup to extract the titles and genres of different categories of films based on books 
    from Netflix.
    param_dictionary: a dictionary of genre and associated html webpage
    param_genres: a list of genres"""
    soup_objects = {}
    list_titles = []
    list_gen = []
    
    # Create a dictionary of soup objects:genre
    for k, v in dictionary.items():
        with open(v) as f:
            soup = BeautifulSoup(f, 'html.parser')
            soup_objects.update({k:soup})
    
    # Create a dictionary of list of titles: genre
    for a, b in soup_objects.items():
        list_gen.append(b.find_all('a', attrs = {'aria-hidden':'false'}))

    for x in range(0, len(list_gen)):
        for i in range(len(list_gen[x])):
            list_titles.append('{}, {}'.format(genres[x], list_gen[x][i].p.contents))

    # Change dictionary to dataframe
    df = pd.DataFrame(np.array(list_titles))
    return df


def make_netflix_dataframe(df):
    """This function cleans up the data from Netflix into a neater dataframe with column titles and appropriate 
    datatypes."""
    new = df[0].str.split(pat = ',', n = 1, expand = True)
    df['genre'] = new[0]
    df['movie'] = new[1]
    df.drop(0, axis = 1, inplace = True)
    df['movie'] = df['movie'].apply(lambda x: x[2:-1])
    return df