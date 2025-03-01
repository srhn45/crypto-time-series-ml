import tensorflow as tf

class rnn_cnn_layer(tf.keras.Model):
    """
    Custom layer that combines an LSTM and a 1D-CNN. 
    My hope is that the convolutional layer can learn mid-term trends while the recurrent layer learns short-term trends.
    I pass the entire output of the RNN layer to the CNN to make use of the kernel.
    Using padding = "valid" also allows the sequence to be shortened over multiple layers.
    Tinkering with the stride to decrease the sequence length to have faster convergence is also an option.
    Or multiple CNN's can be stacked on top with dilation_rate > 1 for a WaveNet-like approach.

    I also utilized dropout and recurrent_dropout since there isn't a lot of historical data available (to combat overfitting).
    """

    def __init__(self, rnn_neurons=32, cnn_filters=32, kernel_size=5, dropout_rate=0.2, recurrent_dropout_rate=0.1, activation="swish"):
        super().__init__()

        self.rnn_neurons = rnn_neurons
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.rnn_layer = tf.keras.layers.LSTM(self.rnn_neurons, return_sequences=True, 
                                           dropout=self.dropout_rate, recurrent_dropout=self.recurrent_dropout_rate,
                                           activation=self.activation)
        self.cnn_layer = tf.keras.layers.Conv1D(filters=self.cnn_filters, kernel_size=self.kernel_size, activation=self.activation, padding="valid")
        
        super().build(input_shape)

    def call(self, inputs, training=True):
        Z = self.cnn_layer(self.rnn_layer(inputs))
        return Z


class predictor_model(tf.keras.Model):
    """
    Stacks multiple rnn_cnn_layers on top of each other, before passing the output to a final LSTM layer.
    The final LSTM layer only passes the final output to the dense layer(s), before the prediction is made.
    The number of layers and neurons/filters, along with other hyperparameters is modular.
    """

    def __init__(self, rnn_cnn_layers=3, rnn_neurons=32, cnn_filters=32, kernel_size=5, dense_layers=2, dense_neurons=16, output_neurons=5, dropout_rate=0.2, recurrent_dropout_rate=0.1, activation="swish"):
        super(predictor_model, self).__init__()
        
        self.rnn_cnn_layers = []
        self.dense_layers = []
        
        for i in range(rnn_cnn_layers):
            self.rnn_cnn_layers.append(
                rnn_cnn_layer(rnn_neurons=rnn_neurons, cnn_filters=cnn_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, recurrent_dropout_rate=recurrent_dropout_rate, activation=activation)
            )

        self.lstm = tf.keras.layers.LSTM(rnn_neurons, return_sequences=False, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                                      activation=tf.keras.activations.get(activation))

        for i in range(dense_layers):
            self.dense_layers.append(tf.keras.layers.Dense(dense_neurons, activation=None, use_bias=False))  # no bias since BatchNorm will have its own bias term.
            self.dense_layers.append(tf.keras.layers.BatchNormalization())
            self.dense_layers.append(tf.keras.activations.get(activation))  # BatchNorm applied before the activation function.
            self.dense_layers.append(tf.keras.layers.Dropout(dropout_rate))  # Dropout to combat overfitting.
                
        self.output_layer = tf.keras.layers.Dense(output_neurons, activation="softmax")  # Using softmax here since predictions are categorical.
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.rnn_cnn_layers:
            x = layer(x, training=training)

        x = self.lstm(x)

        for layer in self.dense_layers:
            x = layer(x, training=training) if isinstance(layer, tf.keras.layers.Dropout) else layer(x)

        return self.output_layer(x)