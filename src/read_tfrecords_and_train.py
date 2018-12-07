import tensorflow as tf
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

TRAIN_DATA_N="train_n.csv"
TEST_DATA_N="test_n.csv"
TRAIN_DATA_C="train_c.csv"
TEST_DATA_C="test_c.csv"
PREDICT_DATA="test_n_nolabel.csv"
CORPUS="corpus.txt"
MODEL_DIR="models/"

def input_fn_eval():
    dataset = tf.data.TextLineDataset(filenames=TEST_DATA_N).skip(1)
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

def input_fn_train():
    dataset = tf.data.TFRecordDataset(filenames="train.tfrecords")
    
    def parse_example(example):
        features = {
            'vote_average': tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'budget':       tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'runtime':      tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'revenue':      tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'vote_count':   tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'year':         tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'genres':       tf.io.VarLenFeature(tf.string)
        }
        parsed_features = tf.io.parse_single_example(example, features)
        label = parsed_features.pop('vote_average')
        return parsed_features, label

    dataset = dataset.map(parse_example).batch(32)
    return dataset

def input_fn_test():
    dataset = tf.data.TFRecordDataset(filenames="test.tfrecords")
    
    def parse_example(example):
        features = {
            'vote_average': tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'budget':       tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'runtime':      tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'revenue':      tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'vote_count':   tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'year':         tf.FixedLenFeature([], tf.float32, default_value=0.0),
            'genres':       tf.io.VarLenFeature(tf.string)
        }
        parsed_features = tf.io.parse_single_example(example, features)
        label = parsed_features.pop('vote_average')
        return parsed_features, label

    dataset = dataset.map(parse_example).batch(32)
    return dataset

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    
    cat_col = tf.feature_column.categorical_column_with_vocabulary_file(
        key='genres',
        vocabulary_file="corpus.txt",
        vocabulary_size=19
    )
    feature_columns = [
        tf.feature_column.numeric_column("budget"),
        tf.feature_column.numeric_column("runtime"),
        tf.feature_column.numeric_column("revenue"),
        tf.feature_column.numeric_column("vote_count"),
        tf.feature_column.numeric_column("year"),
        tf.feature_column.indicator_column(cat_col)
        ]

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

    n_epochs = 10
    print("")
    print("Training...")
    for i in range(n_epochs):
        est.train(input_fn=input_fn_train)

    print("")
    print("Evaluating...")
    est.evaluate(input_fn=input_fn_test)
    
    """
    labels_l = pd.read_csv("test_n.csv", usecols=['vote_average'])['vote_average'].tolist()
    pred_g = est.predict(input_fn=input_fn_test)
    pred_l = [e['predictions'][0] for e in pred_g]
    
    print("")
    print("Predicting..")
    mae = tf.losses.absolute_difference(
        labels=labels_l,
        predictions=pred_l)

    with tf.Session() as sess:
        print("MAE: ", mae.eval())
    """