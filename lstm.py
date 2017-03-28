from controller import *


class LSTM(Controller):
    def __init__(self, batch_size, input_size, output_size, memory_size, seq_length):
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.max_seq_length = seq_length
        self.total_seq_length = 2 * seq_length + 2
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.memory_size)

        # Extra layer of neural network
        # The output is a vector of dimension memory_size, but we want it to be output_size, so we multiply it by some W
        self.weights = tf.Variable(tf.random_normal([self.memory_size, self.output_size], stddev=0.1),
                                   name="output_weights")
        self.biases = tf.Variable(tf.zeros([self.output_size]), name="output_biases")

    def __call__(self, x):
        """
        This call should not be used in DNC; it processes all inputs for all time steps!
        This call would usually iterate through the step function but tensorflow already has this implementation.
        This breaks my idea of how code should be organized in this project but oh well.
        DNC calls the step function for every time step.
        
        :param x: inputs for all time steps
        :return: 
        """
        x_list = tf.split(x, self.max_seq_length, axis=2)
        x_list = [tf.squeeze(i, 2) for i in x_list]

        # I changed the tensorflow source code here, it returns states instead of just the last state
        lstm_outputs, states = tf.contrib.rnn.static_rnn(self.lstm_cell, x_list, dtype=tf.float32)
        outputs = [tf.matmul(o, self.weights) + self.biases for o in lstm_outputs]
        outputs = tf.transpose(outputs, [1, 2, 0])

        # Before returning outputs, adding all the tf summaries!

        states = tf.reshape(tf.transpose(states, [2, 3, 1, 0]),
                            [self.batch_size, 2 * self.memory_size, self.max_seq_length, 1])
        tf.summary.image("LSTM cell state and 'hidden' state", states, max_outputs=2 * self.batch_size)
        tf.summary.image("Input", tf.expand_dims(x, axis=3), max_outputs=self.batch_size)
        tf.summary.image("Output", tf.expand_dims(outputs, axis=3), max_outputs=self.batch_size)

        weights_expanded = tf.expand_dims(tf.expand_dims(self.weights, axis=0), axis=3)
        tf.summary.image("LSTM output weights", tf.transpose(weights_expanded), max_outputs=self.batch_size)

        inner_weights = [var for var in tf.global_variables() if var.name.startswith("rnn/basic_lstm_cell/weights")][0]
        inner_weights_expanded = tf.expand_dims(tf.expand_dims(inner_weights, axis=0), axis=3)
        tf.summary.image("LSTM inner weights", tf.transpose(inner_weights_expanded), max_outputs=self.batch_size)

        return outputs

    def step(self, x):
        # State should be remembered as a class variable?
        return self.lstm_cell(x, state)
