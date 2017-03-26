from controller import *


class LSTM(Controller):
    def __init__(self, batch_size, input_size, output_size, memory_size, seq_length):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.max_seq_length = seq_length
        self.total_seq_length = 2 * seq_length + 2
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.memory_size)

        # Extra layer of neural network
        # The output is a vector of dimension memory_size, but we want it to be output_size, so we multiply it by some W
        self.weights = tf.Variable(tf.random_normal([self.memory_size, self.output_size], stddev=0.1),
                                   name="output_weights")
        self.biases = tf.Variable(tf.zeros([self.output_size]), name="output_biases")

    def __call__(self, x, y):
        x_list = tf.split(x, self.max_seq_length, axis=2)
        x_list = [tf.squeeze(i, 2) for i in x_list]

        lstm_outputs, states = tf.contrib.rnn.static_rnn(self.lstm_cell, x_list, dtype=tf.float32)
        outputs = [tf.matmul(o, self.weights) + self.biases for o in lstm_outputs]
        outputs = tf.transpose(outputs, [1, 2, 0])
        cost = tf.reduce_mean(tf.square(y - outputs))

        # Before returning cost, adding all the tf summaries!

        states = tf.reshape(tf.transpose(states, [2, 3, 1, 0]),
                            [self.batch_size, 2 * self.memory_size, self.max_seq_length, 1])
        tf.summary.image("LSTM hidden state (two of them)", states, max_outputs=2 * self.batch_size)
        tf.summary.image("Input", tf.expand_dims(x, axis=3), max_outputs=self.batch_size)
        tf.summary.image("Output", tf.expand_dims(outputs, axis=3), max_outputs=self.batch_size)

        weights_expanded = tf.expand_dims(tf.expand_dims(self.weights, axis=0), axis=3)
        tf.summary.image("LSTM output weights", tf.transpose(weights_expanded), max_outputs=self.batch_size)

        inner_weights = [v for v in tf.global_variables() if v.name.startswith("rnn/basic_lstm_cell/weights")][0]
        inner_weights_expanded = tf.expand_dims(tf.expand_dims(inner_weights, axis=0), axis=3)
        tf.summary.image("LSTM inner weights", tf.transpose(inner_weights_expanded), max_outputs=self.batch_size)

        tf.summary.scalar('Cost', cost)

        return cost
