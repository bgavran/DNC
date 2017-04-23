from controller import *
from controller_implementations.dnc.memory import *


class DNC(Controller):
    def __init__(self, controller, mem_hp, output_activation=None):
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
        self.output_activation = output_activation

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
        with tf.variable_scope("DNC"):
            dnc_outputs, dnc_states = [], []
            for i in range(self.controller.total_output_length):
                dnc_output, dnc_state = self.step(x[:, :, i], i)

                dnc_outputs.append(dnc_output)
                dnc_states.append(dnc_state)

        dnc_outputs = tf.transpose(dnc_outputs, [1, 2, 0])
        self.notify(dnc_states)

        return dnc_outputs

    def notify(self, states):
        # Notifying the controller, if its not feedforward. Is there a better way to check?
        if states[0][0] is not None:
            self.controller.notify([state[0] for state in states])

        memory_variables = [state[2] for state in states]
        memory_variables = tf.transpose(memory_variables, [1, 2, 0])
        tf.summary.image("Memory states", tf.expand_dims(memory_variables, axis=3), max_outputs=Controller.max_outputs)

        memory_outputs = tf.transpose([mem[1] for mem in states], [1, 2, 0])
        tf.summary.image("Memory outputs", tf.expand_dims(memory_outputs, axis=3), max_outputs=Controller.max_outputs)

    def step(self, x, step):
        """
        Returns the output vector for just one time step
        High level overview of step function for DNC:
        
        c_output, c_state = controller.step()
        dnc_output, dnc_state = process(c_output, c_state)
        return dnc_output, dnc_state
        
        :param x: input data for just one time step of shape [batch_size, input_size]
        :param step: current time step
        :return: 
        """
        with tf.variable_scope("DNC_step"):
            # concatenating and flattening previous time step' read vectors with x to obtain the input vector
            read_vectors_flat = tf.reshape(self.memory.read_vectors,
                                           [self.batch_size, self.memory.num_read_heads * self.memory.word_size])
            controller_input = tf.concat([x, read_vectors_flat], axis=1)

            controller_output, controller_cell = self.controller.step(controller_input, step)
            tf.Print(step, [step], message="STEP")
            memory_output, memory_state = self.memory(controller_output)

            # making sure the dimensions for everything align, using simple matmul for that
            output_vector = tf.einsum("bc,co->bo", controller_output, self.output_weights)

            output = output_vector + memory_output
            if self.output_activation is not None:
                output = self.output_activation(output)

        state = (controller_cell, memory_output, memory_state)
        return output, state
