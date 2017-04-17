from controller import *
from memory import *


class DNC(Controller):
    def __init__(self, controller, mem_hp):
        """
        For now, the output of controller and dnc is of the same shape (shape of output), but it is not always true
        
        :param controller: 
        """
        self.controller = controller
        self.batch_size = controller.batch_size
        self.out_vector_size = self.controller.out_vector_size

        self.mem_hp = mem_hp

        self.memory = Memory(controller.batch_size, controller.out_vector_size, self.mem_hp)

        self.output_weights = tf.Variable(
            tf.random_normal([self.controller.out_vector_size, self.out_vector_size], stddev=0.01),
            name="output_weights")

    def __call__(self, x):
        """
        High level overview of __call__:
        for step in steps:
            output, state = self.step(data[step])
        return all_outputs
        
        In DNC the step() method calls the controllers step() method
            
            
        :param x: input data of shape [batch_size, input_size, sequence_length]
        :return: list of outputs for every time step of the network
        """
        with tf.variable_scope("DNC") as scope:
            outputs, states = [], []
            for i in range(self.controller.total_output_length):
                dnc_output, dnc_state = self.step(x[:, :, i], i)

                outputs.append(dnc_output)
                states.append(dnc_state)

        controller_outputs = tf.transpose(outputs, [1, 2, 0])

        # Checking if the controller is not feedforward. Is there a better way?
        if states[0][0] is not None:
            controller_states = tf.transpose([state[0] for state in states], [2, 3, 1, 0])
            controller_states = tf.reshape(controller_states,
                                           [self.batch_size, 2 * self.controller.memory_size,
                                            self.controller.total_output_length, 1])
            tf.summary.image("LSTM cell state and 'hidden' state", controller_states,
                             max_outputs=self.batch_size)

            weights_expanded = tf.expand_dims(tf.expand_dims(self.controller.weights, axis=0), axis=3)
            tf.summary.image("LSTM output weights", tf.transpose(weights_expanded), max_outputs=self.batch_size)

        memory_outputs = tf.transpose([mem[1] for mem in states], [1, 2, 0])
        memory_states = tf.reshape(tf.transpose([mem[2] for mem in states], [1, 2, 0]),
                                   [self.batch_size, self.memory.memory_size, self.controller.total_output_length, 1])

        tf.summary.image("Memory states", memory_states, max_outputs=self.batch_size)
        tf.summary.image("Memory outputs", tf.expand_dims(memory_outputs, axis=3), max_outputs=self.batch_size)

        tf.summary.image("Input", tf.expand_dims(x, axis=3), max_outputs=self.batch_size)
        tf.summary.image("Output", tf.expand_dims(controller_outputs, axis=3), max_outputs=self.batch_size)

        # inner_weights = [var for var in tf.global_variables() if var.name.startswith("rnn/basic_lstm_cell/weights")][0]
        # inner_weights_expanded = tf.expand_dims(tf.expand_dims(inner_weights, axis=0), axis=3)
        # tf.summary.image("LSTM inner weights", tf.transpose(inner_weights_expanded), max_outputs=self.batch_size)

        return controller_outputs

    def step(self, x, step):
        """
        Returns the output vector for just one time step
        High level overview of step function for DNC:
        
        c_output, c_state = controller.step()
        dnc_output, dnc_state = process(c_output, c_state)
        return dnc_output, dnc_state
        
        :param x: input data for just one time step of shape [batch_size, input_size, 1]
        :param step: current time step
        :return: 
        """
        with tf.variable_scope("DNC_step"):
            controller_output, controller_cell = self.controller.step(x, step)

            # making sure the dimensions for everything align, using simple matmul for that
            # multiplying x @ W instead of W @ x (like in the paper) because it needs to be done in batches
            output_vector = controller_output @ self.output_weights  # shape [batch_size, out_vector_size]

            memory_output, memory_state = self.memory(controller_output)
            output = output_vector + memory_output

        return output, (controller_cell, memory_output, memory_state)
