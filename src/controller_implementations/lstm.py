from controller import *


class LSTM(Controller):
    def __init__(self, batch_size, inp_vector_size, memory_size, n_layers, out_vector_size=None):
        self.batch_size = batch_size
        self.inp_vector_size = inp_vector_size
        self.memory_size = memory_size
        self.out_vector_size = self.memory_size
        self.n_layers = n_layers
        if out_vector_size is not None:
            self.out_vector_size = out_vector_size

        one_cell = tf.contrib.rnn.BasicLSTMCell
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell([one_cell(self.memory_size) for _ in range(self.n_layers)])

        self.state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        # Extra layer of neural network
        # The output is a vector of dimension memory_size, but we want it to be output_size, so we multiply it by some W
        self.weights = tf.Variable(tf.random_normal([self.memory_size, self.out_vector_size], stddev=0.1),
                                   name="output_weights")
        self.biases = tf.Variable(tf.zeros([self.out_vector_size]), name="output_biases")

    def __call__(self, x, sequence_length):
        """
        This call should not be used in DNC; it processes all inputs for all time steps!
        It's not a matter of the implementation of this function, it's just that DNC never processes the inputs for 
        all time steps in its controller.
        
        
        :param x: inputs for all time steps
        :return: outputs for all time steps
        """
        # x shape is [max_time, batch_size, vector_size]
        x = tf.transpose(x, [0, 2, 1])
        sequence_length = tf.stack([sequence_length for _ in range(self.batch_size)])
        outputs, states = tf.nn.dynamic_rnn(self.lstm_cell, x, dtype=tf.float32, sequence_length=sequence_length)

        # have to do the einsum this way and transpose later because of the way tf broadcasts biases
        outputs = tf.einsum("btm,mo->bto", outputs, self.weights) + self.biases
        outputs = tf.transpose(outputs, [0, 2, 1])

        return outputs, states

    def notify(self, states):
        pass

    def step(self, x, step):
        with tf.variable_scope("LSTM_step") as scope:
            hidden, self.state = self.lstm_cell(x, self.state)

            # Here we have an extra affine transformation after the standard LSTM cell
            # Because we need to have the shapes of the y and the output match
            # DNC solves this, but if we're testing against baseline LSTM, we need to have this extra layer.
            # But probably it wouldn't hurt to have it in DNC as well?
            # output = tf.matmul(hidden, self.weights) + self.biases

        return hidden, self.state
