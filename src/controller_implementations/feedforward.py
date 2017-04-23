from controller import *


class Feedforward(Controller):
    def __init__(self, inp_vector_size, out_vector_size, total_output_length, batch_size, n_layers, layer_sizes):
        assert n_layers == len(layer_sizes)
        assert layer_sizes[-1] == out_vector_size
        self.inp_vector_size = inp_vector_size
        self.total_output_length = total_output_length
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.out_vector_size = out_vector_size

    def __call__(self, x):
        """
        This should not be called from DNC!
        :param x: inputs for all time steps
        :return: list of outputs for every time step
        """
        outputs = []
        for i in range(self.total_output_length):
            output, _, = self.step(x[:, :, i], i)
            outputs.append(output)
        outputs = tf.transpose(outputs, [1, 2, 0])

        return outputs

    def step(self, x, step):
        """
        Returns the output vector for just one time step
        
        :param x: one vector representing input for one time step
        :param step: current time step
        :return: output of feedforward network and None, because it doesn't have a state (it is required by interface)
        """
        with tf.variable_scope("FF_step") as scope:
            # Below is a leaky abstraction from tensorflow. Reusing of the variables needs to be set explicitly
            if step > 0:
                scope.reuse_variables()
            for layer_size in self.layer_sizes[:-1]:
                x = tf.layers.dense(x, layer_size, activation=tf.nn.relu)

            x = tf.layers.dense(x, self.layer_sizes[-1])
        return x, None

    def notify(self, states):
        """
        FF controller doesn't have an internal state
        :param states: 
        :return: 
        """
        pass
