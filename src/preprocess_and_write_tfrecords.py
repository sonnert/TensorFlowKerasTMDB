import pandas as pd
import tensorflow as tf

import json, csv, sys, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

DATA_PATH="../res/tmdb_5000_movies.csv"

def prepare_data_s():
    """
    Prepares data to be used for the single-label regression task
    of predicting a movie's generated revenue. 
    
    A created corpus is to be used together with the genres column 
    to create a TensorFlow categorical multi-value feature.

    Creates "train.csv", "test.csv", and "corpus.txt" data files.
    """

    # Read and shuffle data
    data = pd.read_csv(DATA_PATH, usecols=
    ['budget', 'genres', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count'])
    data = data.sample(frac=1)

    df_raw = data[['vote_average', 'budget', 'runtime', 'revenue', 'vote_count', 'genres']]

    # Get years from dates and add to df
    dates_to_years = [int(s.split('-', 1)[0]) if isinstance(s, str) else 0 for s in data['release_date']]
    df_raw_years = pd.concat([df_raw, pd.Series(dates_to_years, name="year")], axis=1)

    # Clean data of bad entries
    df_raw_years = df_raw_years[df_raw_years['budget'] > 1000]
    df_raw_years = df_raw_years[df_raw_years['revenue'] > 1000]
    df_raw_years = df_raw_years[df_raw_years.astype(str)['genres'] != '[]']

    # Re-format genre column
    list_of_genres_per_sample = [[e['name'] for e in json.loads(le)] for le in df_raw_years['genres']]

    # Separate numeric features and scale
    df_features_n = df_raw_years[['budget', 'runtime', 'revenue', 'vote_count', 'year']]
    scaler = MinMaxScaler()
    df_nolabel = pd.DataFrame(scaler.fit_transform(df_features_n))

    # Re-introduce label and add genres
    df = pd.concat([df_raw_years['vote_average'].reset_index(drop=True), 
        df_nolabel.reset_index(drop=True), 
        pd.Series(list_of_genres_per_sample).reset_index(drop=True)], 
        axis=1, 
        ignore_index=True)
    df = df.dropna()
    df.columns = ['vote_average', 'budget', 'runtime', 'revenue', 'vote_count', 'year', 'genres']

    # Split to train and test sets
    train, test = train_test_split(df, test_size=0.1)
    _, pred = train_test_split(df_nolabel, test_size=0.1)

    def _bytes_feature(value):
        byteslist = [bytes(str(e), 'utf8') for e in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=byteslist))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    tftrain = "train.tfrecords"
    
    with tf.python_io.TFRecordWriter("train.tfrecords") as tf_writer:
        header = True
        for _, row in train.iterrows():
            if header:
                header = False
                continue

            feature = {
                'vote_average': _float_feature(row['vote_average']),
                'budget':       _float_feature(row['budget']),
                'runtime':      _float_feature(row['runtime']),
                'revenue':      _float_feature(row['revenue']),
                'vote_count':   _float_feature(row['vote_count']),
                'year':         _float_feature(row['year']),
                'genres':       _bytes_feature(row['genres'])
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            tf_writer.write(example.SerializeToString())

    # Output data file for genre corpus
    flat_list_of_genres_per_sample = [item for sublist in list_of_genres_per_sample for item in sublist]
    with open("corpus.txt", mode="w+") as f:
        for s in list(set(flat_list_of_genres_per_sample)):
            f.write("%s\n" % s)

if __name__ == "__main__":
    prepare_data_s()