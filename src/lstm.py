from controller import *


class LSTM(Controller):
    def __init__(self, batch_size, inp_vector_size, out_vector_size, memory_size, total_output_length):
        self.batch_size = batch_size
        self.inp_vector_size = inp_vector_size
        self.out_vector_size = out_vector_size
        self.memory_size = memory_size
        self.total_output_length = total_output_length

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.memory_size)

        self.state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        # Extra layer of neural network
        # The output is a vector of dimension memory_size, but we want it to be output_size, so we multiply it by some W
        self.weights = tf.Variable(tf.random_normal([self.memory_size, self.out_vector_size], stddev=0.1),
                                   name="output_weights")
        self.biases = tf.Variable(tf.zeros([self.out_vector_size]), name="output_biases")

    def __call__(self, x):
        """
        This call should not be used in DNC; it processes all inputs for all time steps!
        It's not a matter of the implementation of this function, it's just that DNC never processes the inputs for 
        all time steps in its controller.
        
        This call would usually iterate through the step function but tensorflow already has this implementation.
        This breaks my idea of how code should be organized in this project but oh well.
        DNC calls the step function for every time step.
        
        :param x: inputs for all time steps
        :return: 
        """
        raise RuntimeError("This function shouldn't be called, yet.")
        x_list = tf.split(x, self.max_seq_length, axis=2)
        x_list = [tf.squeeze(i, 2) for i in x_list]

        # I changed the tensorflow source code here, it returns states instead of just the last state
        from core_rnn import static_rnn
        lstm_outputs, states = static_rnn(self.lstm_cell, x_list, dtype=tf.float32)
        outputs = [tf.matmul(o, self.weights) + self.biases for o in lstm_outputs]

        # Before returning outputs, adding all the tf summaries!

        # states = tf.reshape(tf.transpose(states, [2, 3, 1, 0]),
        #                     [self.batch_size, 2 * self.memory_size, self.max_seq_length, 1])
        # tf.summary.image("LSTM cell state and 'hidden' state", states, max_outputs=2 * self.memory_size)
        # tf.summary.image("Input", tf.expand_dims(x, axis=3), max_outputs=self.batch_size)
        # tf.summary.image("Output", tf.expand_dims(outputs, axis=3), max_outputs=self.batch_size)
        #
        # weights_expanded = tf.expand_dims(tf.expand_dims(self.weights, axis=0), axis=3)
        # tf.summary.image("LSTM output weights", tf.transpose(weights_expanded), max_outputs=self.batch_size)
        #
        # inner_weights = [var for var in tf.global_variables() if var.name.startswith("rnn/basic_lstm_cell/weights")][0]
        # inner_weights_expanded = tf.expand_dims(tf.expand_dims(inner_weights, axis=0), axis=3)
        # tf.summary.image("LSTM inner weights", tf.transpose(inner_weights_expanded), max_outputs=self.batch_size)

        return outputs

    def step(self, x, step):
        with tf.variable_scope("LSTM_step") as scope:
            # Below is a leaky abstraction from tensorflow. Reusing of the variables needs to be set explicitly
            if step > 0:
                scope.reuse_variables()

            hidden, self.state = self.lstm_cell(x, self.state)
            # Here we have an extra affine transformation after the standard LSTM cell
            # Because we need to have the shapes of the y and the output match
            # DNC solves this, but if we're testing against baseline LSTM, we need to have this extra layer.
            # But probably it wouldn't hurt to have it in DNC as well?
            output = tf.matmul(hidden, self.weights) + self.biases

        return output, self.state
