def create_sequences(features, labels, seq_length=100):
    """ 
    Takes the time series input and turns it into a TensorFlow dataset with variable sequence length. 
    Uses the label corresponding to the last time index as the label of the sequence by default.
    """

    import tensorflow as tf
    import pandas as pd
    import numpy as np

    def gen():
        for i in range(len(features) - seq_length):
            yield features[i : i + seq_length], labels[i + seq_length - 1]

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_length, features.shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    return dataset

def split_data(df, size: float = 0.5, test_pct = 0.1):
    """
    Splits the dataset into training and testing sets while maintaining time order.
    The latest fraction of the data (test_pct) is taken as the test set every time by default, to train the models for the latest market dynamics.
    """

    import tensorflow as tf
    import pandas as pd
    import numpy as np

    test_size = int(len(df) * test_pct)
    train_size = int(len(df) * size)

    train_features = df.iloc[:train_size].drop(columns=["timestamp", "price_change_BTCUSDT"]).values
    train_labels = df.iloc[:train_size]["price_change_BTCUSDT"].values                    
    
    test_features = df.iloc[train_size:train_size+test_size].drop(columns=["timestamp", "price_change_BTCUSDT"]).values
    test_labels = df.iloc[train_size:train_size+test_size]["price_change_BTCUSDT"].values

    return train_features, train_labels, test_features, test_labels
