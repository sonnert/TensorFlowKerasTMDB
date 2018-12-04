import tensorflow as tf
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

TRAIN_DATA="train.csv"
TEST_DATA="test.csv"
PREDICT_DATA="test_nolabel.csv"
CORPUS="corpus.txt"
MODEL_DIR="models/"

def input_fn_train():
    dataset = tf.data.TextLineDataset(filenames=TRAIN_DATA).skip(1)
    HEADERS = ['vote_average', 'budget', 'runtime', 'revenue', 'vote_count', 'year']
    FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    genres = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_file(
        key='genres', vocabulary_file=CORPUS))

    def parse_line(line):
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
        features = dict(zip(HEADERS, fields))
        label = features.pop('vote_average')
        return features, label

    dataset = dataset.batch(32).map(parse_line)
    dataset_z = tf.data.Dataset.zip((dataset, genres))
    return dataset_z

def input_fn_eval():
    dataset = tf.data.TextLineDataset(filenames=TEST_DATA).skip(1)
    HEADERS = ['vote_average', 'budget', 'runtime', 'revenue', 'vote_count', 'year']
    FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    def parse_line(line):
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
        features = dict(zip(HEADERS, fields))
        label = features.pop('vote_average')
        return features, label

    dataset = dataset.batch(32).map(parse_line)
    return dataset

def input_fn_pred():
    dataset = tf.data.TextLineDataset(filenames=PREDICT_DATA).skip(1)
    HEADERS = ['budget', 'runtime', 'revenue', 'vote_count', 'year']
    FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0]]

    def parse_line(line):
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
        features = dict(zip(HEADERS, fields))
        return features

    dataset = dataset.batch(32).map(parse_line)
    return dataset

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    
    feature_columns = [
        tf.feature_column.numeric_column("budget"),
        tf.feature_column.numeric_column("runtime"),
        tf.feature_column.numeric_column("revenue"),
        tf.feature_column.numeric_column("vote_count"),
        tf.feature_column.numeric_column("year")]

    est = tf.estimator.DNNRegressor(
        model_dir=MODEL_DIR, 
        feature_columns=feature_columns,
        hidden_units=[20, 40, 80],
        optimizer=lambda: tf.train.AdamOptimizer(
            learning_rate=tf.train.exponential_decay(
                learning_rate=0.1,
                global_step=tf.train.get_global_step(),
                decay_steps=10000,
                decay_rate=0.96)
                )
            )

    n_epochs = 1
    for i in range(n_epochs):
        est.train(input_fn=input_fn_train)

    """
    est.evaluate(input_fn=input_fn_eval)
    
    labels_l = pd.read_csv("test.csv", usecols=['vote_average'])['vote_average'].tolist()
    pred_g = est.predict(input_fn=input_fn_pred)
    pred_l = [e['predictions'][0] for e in pred_g]
    
    print("")
    print("Predicting..")
    mae = tf.losses.absolute_difference(
        labels=labels_l,
        predictions=pred_l)

    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        print("MAE: ", mae.eval())
    """