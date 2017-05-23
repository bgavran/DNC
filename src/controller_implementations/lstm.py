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

            # Extra layer of neural network
            # The output is a vector of dimension memory_size, but we want it to be output_size
            # so we multiply it by some W
            # initializer = tf.contrib.layers.xavier_initializer()
            initializer = tf.random_normal
            self.weights = tf.Variable(initializer([self.memory_size, self.out_vector_size], stddev=0.1),
                                       name="output_weights")
            self.biases = tf.Variable(tf.zeros([1, self.out_vector_size, 1]), name="output_biases")

        one_cell = tf.contrib.rnn.BasicLSTMCell
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell([one_cell(self.memory_size) for _ in range(self.n_layers)])

        self.initial_state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

    def __call__(self, x, sequence_length):
        """
        This call should not be used in DNC; it processes all inputs for all time steps!
        It's not a matter of the implementation of this function, it's just that DNC never processes the inputs for 
        all time steps in its controller.
        
        
        :param x: inputs for all time steps, shape [batch_size, vector_size, time_steps]
        :return: outputs for all time steps
        """
        x = tf.transpose(x, [0, 2, 1])
        sequence_length = tf.stack([sequence_length for _ in range(self.batch_size)])
        # input should be of shape [batch_size, max_time, ...]
        outputs, states = tf.nn.dynamic_rnn(self.lstm_cell, x, dtype=tf.float32, sequence_length=sequence_length)
        # output is of shape [batch_size, max_time, output_size]

        # have to do the einsum this way and transpose later because of the way tf broadcasts biases
        outputs = tf.einsum("btm,mo->bot", outputs, self.weights) + self.biases

        # TODO this returns just the final state and not all of them?
        return outputs, states

    def step(self, x, state, step):
        with tf.variable_scope("LSTM_step"):
            hidden, new_state = self.lstm_cell(x, state)

        return hidden, new_state

    def notify(self, summaries):
        pass
        # summ = tf.reshape(tf.transpose(summaries, [2, 1, 3, 0]), [self.batch_size, 2 * self.memory_size, -1, 1])
        # tf.summary.image("LSTM_state", summ, max_outputs=Controller.max_outputs)
