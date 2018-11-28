import pandas as pd
import tensorflow as tf

import est_rf

import json
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from datetime import datetime

DATA_PATH="res/tmdb_5000_movies.csv"
MODEL_DIR="models/"

def prepare_data_s():
    """
    Prepares data to be used for the single-label regression task
    of predicting a movie's generated revenue.

    Creates "train.csv" and "test.csv" data files.
    """

    data = pd.read_csv(DATA_PATH, usecols=
    ['budget', 'genres', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count'])

    # 'Revenue' single-label regression data set
    dfraw = data[['revenue', 'budget', 'runtime', 'vote_average', 'vote_count']]

    dates_to_years = [int(s.split('-', 1)[0]) if isinstance(s, str) else 0 for s in data['release_date']]
    Xra = pd.concat([dfraw, pd.Series(dates_to_years, name="year")], axis=1)

    list_of_genres_per_sample = [[e['name'] for e in json.loads(le)] for le in data['genres']]
    genres_df = pd.DataFrame(list_of_genres_per_sample).iloc[:,:3]
    genres_df.columns = ['genre1', 'genre2', 'genre3']

    df = pd.concat([Xra, genres_df], axis=1)
    df = df[df['revenue'] != 0]
    train, test = train_test_split(df, test_size=0.1)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)