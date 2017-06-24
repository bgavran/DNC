from controller import *


class LSTM(Controller):
    def __init__(self, inp_vector_size, memory_size, n_layers, out_vector_size=None, initializer=tf.random_normal,
                 initial_stddev=None):
        self.inp_vector_size = inp_vector_size
        self.memory_size = memory_size
        self.out_vector_size = self.memory_size
        self.out_layer_exists = False
        self.n_layers = n_layers
        if out_vector_size is not None:
            self.out_vector_size = out_vector_size
            self.out_layer_exists = True

            # Extra layer of neural network
            # The output is a vector of dimension memory_size, but we want it to be output_size
            # so we multiply it by some W
            self.weights = tf.Variable(initializer([self.memory_size, self.out_vector_size], stddev=initial_stddev),
                                       name="output_weights")
            self.biases = tf.Variable(tf.zeros([self.out_vector_size]), name="output_biases")

        one_cell = tf.contrib.rnn.BasicLSTMCell
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell([one_cell(self.memory_size) for _ in range(self.n_layers)])

    def initial_state(self, batch_size):
        return self.lstm_cell.zero_state(batch_size, dtype=tf.float32)

    def __call__(self, x, sequence_lengths):
        """
        This call should not be used in DNC; it processes all inputs for all time steps!
        It's not a matter of the implementation of this function, it's just that DNC never processes the inputs for 
        all time steps in its controller.
        
        
        :return: outputs for all time steps
        """
        # input should be of shape [batch_size, max_time, ...]
        outputs, states = tf.nn.dynamic_rnn(self.lstm_cell,
                                            x,
                                            dtype=tf.float32,
                                            sequence_length=sequence_lengths,
                                            swap_memory=True)
        # output is of shape [batch_size, max_time, output_size]
        if self.out_layer_exists:
            # TODO optimize this einsum
            outputs = tf.einsum("btm,mo->bto", outputs, self.weights) + self.biases

        # TODO this returns just the final state and not all of them? (needed for visualization purposes)
        return outputs, states

    def step(self, x, state, step):
        with tf.variable_scope("LSTM_step"):
            hidden, new_state = self.lstm_cell(x, state)

        return hidden, new_state

    def notify(self, summaries):
        pass
        # summ = tf.reshape(tf.transpose(summaries, [2, 1, 3, 0]), [self.batch_size, 2 * self.memory_size, -1, 1])
        # tf.summary.image("LSTM_state", summ, max_outputs=Controller.max_outputs)
