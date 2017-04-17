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
        
        
        :param x: inputs for all time steps
        :return: outputs for all time steps
        """
        outputs, states = [], []
        for i in range(self.out_vector_size):
            output, state = self.step(x[:, :, i], i)
            outputs.append(output)
            states.append(state)

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
