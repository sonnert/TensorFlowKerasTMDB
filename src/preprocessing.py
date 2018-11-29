import pandas as pd
import tensorflow as tf

import est_rf

import json
from sklearn.model_selection import train_test_split
from datetime import datetime

DATA_PATH="res/tmdb_5000_movies.csv"

def prepare_data_s():
    """
    Prepares data to be used for the single-label regression task
    of predicting a movie's generated revenue. 
    
    A created corpus is to be used together with the genres column 
    to create a TensorFlow categorical multi-value feature.

    Creates "train.csv", "test.csv", and "corpus.txt" data files.
    """

    data = pd.read_csv(DATA_PATH, usecols=
    ['budget', 'genres', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count'])

    df_raw = data[['revenue', 'budget', 'runtime', 'vote_average', 'vote_count']]

    dates_to_years = [int(s.split('-', 1)[0]) if isinstance(s, str) else 0 for s in data['release_date']]
    df_raw_years = pd.concat([df_raw, pd.Series(dates_to_years, name="year")], axis=1)

    list_of_genres_per_sample = [[e['name'] for e in json.loads(le)] for le in data['genres']]
    flat_list_of_genres_per_sample = [item for sublist in list_of_genres_per_sample for item in sublist]

    df = pd.concat([df_raw_years, pd.Series(list_of_genres_per_sample, name='genres')], axis=1)
    df = df[df['revenue'] != 0]
    train, test = train_test_split(df, test_size=0.1)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    with open("corpus.txt", mode="w+") as f:
        for s in list(set(flat_list_of_genres_per_sample)):
            f.write("%s\n" % s)

if __name__ == "__main__":
    prepare_data_s()