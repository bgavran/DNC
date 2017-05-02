from controller import *
from controller_implementations.dnc.memory import *


class DNC(Controller):
    def __init__(self, controller, out_vector_size, mem_hp):
        """
        For now, the output of controller and dnc is of the same shape (shape of output), but it is not always true
        
        :param controller: 
        """
        self.controller = controller
        self.batch_size = controller.batch_size
        self.out_vector_size = out_vector_size
        self.controller_output_size = self.controller.out_vector_size

        self.mem_hp = mem_hp

        self.memory = Memory(controller.batch_size, self.controller_output_size, self.out_vector_size, self.mem_hp)

        self.output_weights = tf.Variable(
            tf.random_normal([self.controller.out_vector_size, self.out_vector_size], stddev=0.1),
            name="output_weights_controller")

    def __call__(self, x, sequence_length):
        """

        :param x: input data of shape [batch_size, input_size, sequence_length]
        :return: 
        """
        with tf.variable_scope("DNC") as scope:
            condition = lambda step, *_: step < sequence_length

            initial_state = [self.controller.state, self.memory.init_memory()]

            all_summaries = [tf.TensorArray(tf.float32, sequence_length) for _ in
                             range(len(initial_state[1]) + 10)]
            all_outputs = tf.TensorArray(tf.float32, sequence_length)

            step, x, output, state, all_outputs, all_summaries = tf.while_loop(condition,
                                                                               self.while_loop_step,
                                                                               loop_vars=[0, x, tf.zeros(
                                                                                   (self.batch_size,
                                                                                    self.out_vector_size)),
                                                                                          initial_state, all_outputs,
                                                                                          all_summaries],
                                                                               swap_memory=True)
        all_outputs = tf.transpose(all_outputs.stack(), [1, 2, 0])
        all_summaries = [summary.stack() for summary in all_summaries]
        return all_outputs, all_summaries

    def while_loop_step(self, step, x, output, state, all_outputs, all_summaries):
        x_step = x[:, :, step]
        # concatenating and flattening previous time step' read vectors with x to obtain the input vector
        read_vectors_flat = tf.reshape(state[1][6],
                                       [self.batch_size, self.memory.num_read_heads * self.memory.word_size])
        controller_input = tf.concat([x_step, read_vectors_flat], axis=1)
        controller_state, memory_state = state

        controller_output, controller_state = self.controller.step(controller_input, step)
        memory_output, memory_state, extra_images = self.memory.step(controller_output, memory_state)

        # making sure the dimensions for everything align, using simple matmul for that
        output_vector = tf.einsum("bc,co->bo", controller_output, self.output_weights)
        output = output_vector + memory_output

        state = [controller_state, memory_state]
        new_summaries = [all_summaries[i].write(step, state) for i, state in enumerate(memory_state)]
        new_summaries.extend(
            [all_summaries[len(memory_state) + i].write(step, sum_img) for i, sum_img in enumerate(extra_images)])
        new_summaries.extend([all_summaries[-1].write(step, memory_output)])
        all_outputs = all_outputs.write(step, output)
        return [step + 1, x, output, state, all_outputs, new_summaries]

    def step(self, x, step):
        """
        Returns the output vector for just one time step
        High level overview of step function for DNC:

        c_output, c_state = controller.step()
        dnc_output, dnc_state = process(c_output, c_state)
        return dnc_output, dnc_state
        """
        # concatenating and flattening previous time step' read vectors with x to obtain the input vector
        read_vectors_flat = tf.reshape(self.memory.read_vectors,
                                       [self.batch_size, self.memory.num_read_heads * self.memory.word_size])
        controller_input = tf.concat([x, read_vectors_flat], axis=1)

        controller_output, controller_cell = self.controller.step(controller_input, step)
        memory_output, memory_state = self.memory.step(controller_output)

        # making sure the dimensions for everything align, using simple matmul for that
        output_vector = tf.einsum("bc,co->bo", controller_output, self.output_weights)

        output = output_vector + memory_output

        state = (controller_cell, memory_output, memory_state)
        return output, state

    def notify(self, summaries):
        def summary_convert_4d(summary, title, transpose=None):
            if transpose is None:
                transpose = [1, 2, 0, 3]
            tf.summary.image(title, tf.transpose(summary, transpose), max_outputs=Controller.max_outputs)

        def summary_convert(summary, title):
            tf.summary.image(title, tf.expand_dims(tf.transpose(summary, [1, 2, 0]), axis=3),
                             max_outputs=Controller.max_outputs)

        summary_convert_4d(summaries[0], "Read_weightings")
        summary_convert_4d(summaries[6], "R_read_vectors", transpose=[1, 3, 0, 2])
        summary_convert_4d(summaries[7], "R_read_modes", transpose=[1, 3, 0, 2])
        summary_convert_4d(summaries[14], "Forward_weighting")
        summary_convert_4d(summaries[15], "Backward_weighting")

        summary_convert(summaries[1], "Write_weighting")
        summary_convert(summaries[2], "Usage_vector")
        summary_convert(summaries[3], "Precedence_weighting")
        summary_convert(summaries[8], "Write_gate")
        summary_convert(summaries[9], "Allocation_gate")
        summary_convert(summaries[10], "Write_strength")
        summary_convert(summaries[12], "Erase_vector")
        summary_convert(summaries[16], "Memory_output")
