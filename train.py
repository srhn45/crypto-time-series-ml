def train(df, model, initial_size=0.5, test_pct=0.1, seq_length=100, patience=5, epochs=100):
    """
    Trains the model, with an expanding window of time (increasing amount of training data) to simulate the real world.
    Implemented early stopping to stop fitting the model with the current set of data if the validation accuracy starts increasing.
    """

    import tensorflow as tf
    from data import create_sequences, split_data

    size = initial_size

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    while size + test_pct <= 1:
        train_features, train_labels, test_features, test_labels = split_data(df, size=size, test_pct=test_pct)
 
        train_dataset = create_sequences(train_features, train_labels, seq_length).shuffle(10000).batch(32).repeat().prefetch(tf.data.AUTOTUNE)
        test_dataset = create_sequences(test_features, test_labels, seq_length).batch(32).prefetch(tf.data.AUTOTUNE)
        
        print(f"Training on {int(size*len(df))} samples, testing on {int(test_pct*len(df))} samples (total size = {len(df)})")

        history = model.fit(train_dataset, validation_data=test_dataset,
            epochs=epochs, steps_per_epoch=len(train_features) // 32,
            callbacks=[early_stopping],
            verbose=1
        )

        size += test_pct

        if "val_accuracy" in history.history:
            val_acc = max(history.history["val_accuracy"])
        else:
            val_acc = 0

    return model, val_acc