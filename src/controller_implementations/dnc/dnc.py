from controller import *
from controller_implementations.dnc.memory import *


class DNC(Controller):
    def __init__(self, controller, batch_size, out_vector_size, mem_hp, initializer=tf.random_normal,
                 initial_stddev=0.1):
        """

        :param controller: FF or LSTM controller
        :param batch_size:
        :param out_vector_size: length of the output vector
        :param mem_hp: memory hyperparameters. Not used at all here, just passed on to memory
        :param initializer:
        :param initial_stddev:
        """
        self.controller = controller
        self.batch_size = batch_size
        self.out_vector_size = out_vector_size
        self.controller_output_size = self.controller.out_vector_size

        self.mem_hp = mem_hp

        self.memory = Memory(self.batch_size, self.controller_output_size, self.out_vector_size, self.mem_hp,
                             initializer=initializer, initial_stddev=initial_stddev)

        self.output_weights = tf.Variable(
            initializer([self.controller.out_vector_size, self.out_vector_size], stddev=initial_stddev),
            name="output_weights_controller")

    def __call__(self, x, sequence_length):
        """
        Performs the DNC calculation for all time steps of the input data (x).
        Uses tf.while_loop as a main way of doing that. Tf.while_loop requires explicit passing of the parameters which 
        change from step to step so it messes up the beauty of the code a little bit.

        :param x: input data of shape [batch_size, input_size, sequence_length]
        :param sequence_length:
        :return: output of the same shape as x and summaries which are to be passed to notify() method
        """
        # TODO is it possible to extend tensorflow's RNN and then use tf.nn.dynamic_rnn here?

        # dynamic shapes, batch size and sequence length
        # ideally, both would be dynamic, but tf throws an error because tf.unstack() (method used in DNC memory)
        # doesn't really work with dynamic shapes

        batch_size = self.batch_size
        # batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        with tf.variable_scope("DNC"):
            condition = lambda step, *_: step < seq_len

            initial_state = [self.controller.initial_state(batch_size), self.memory.init_memory(batch_size)]
            output_initial = tf.zeros((batch_size, self.out_vector_size))

            all_outputs = tf.TensorArray(tf.float32, seq_len)
            # TODO remove the magic number
            all_summaries = [tf.TensorArray(tf.float32, seq_len)
                             for _ in range(len(initial_state[1]) + 13)]

            step, x, output, state, all_outputs, all_summaries = tf.while_loop(condition,
                                                                               self.while_loop_step,
                                                                               loop_vars=[0,
                                                                                          x,
                                                                                          output_initial,
                                                                                          initial_state,
                                                                                          all_outputs,
                                                                                          all_summaries],
                                                                               swap_memory=True)
        all_outputs = tf.transpose(all_outputs.stack(), [1, 0, 2])
        all_summaries = [summary.stack() for summary in all_summaries]
        return all_outputs, all_summaries

    def while_loop_step(self, step, x, output, state, all_outputs, all_summaries):
        """
        Replacement for the .step() method which is only usable in the tf.while_loop.
        Processes one time step of DNC and keeps track of all needed variables (including ones for visualization in tb)
        
        :param step: integer representing current time step
        :param output: tensor of shape [batch_size, out_vector_size]. Represents output for *just one* time step
        :param state: list containing [controller_state, memory_state]
        :param all_outputs: tensorarray containing all the outputs. Used for computing gradients
        :param all_summaries: all summaries that will be forwarded to the notify() method
        :return: the updated arguments of this method
        """
        batch_size = tf.shape(x)[0]
        x_step = x[:, step, :]
        # concatenating and flattening previous time step' read vectors with x to obtain the input vector
        read_vectors_flat = tf.reshape(state[1][6],
                                       [batch_size, self.memory.num_read_heads * self.memory.word_size])
        controller_input = tf.concat([x_step, read_vectors_flat], axis=1)
        controller_state, memory_state = state

        controller_output, controller_state = self.controller.step(controller_input, controller_state, step)
        memory_output, memory_state, extra_images = self.memory.step(controller_output, memory_state)

        # making sure the dimensions for everything align, using simple matmul for that
        output_vector = controller_output @ self.output_weights
        output = output_vector + memory_output

        state = [controller_state, memory_state]
        all_outputs = all_outputs.write(step, output)
        new_summaries = [all_summaries[i].write(step, state) for i, state in enumerate(memory_state)]
        new_summaries.extend(
            [all_summaries[len(memory_state) + i].write(step, summ_img) for i, summ_img in enumerate(extra_images)])
        new_summaries.extend([all_summaries[-2].write(step, memory_output)])
        new_summaries.extend([all_summaries[-1].write(step, controller_state)])
        return [step + 1, x, output, state, all_outputs, new_summaries]

    def step(self, x, state, step):
        """
        Not really used here since DNC needs to use Tensorflow's tf.while_loop which is not really compatible with
        anything.

        :param x:
        :param state:
        :param step:
        :return:
        """
        raise NotImplementedError()

    def notify(self, summaries):
        """
        Processes all the tensors for display in tensorboard
        
        :param summaries: 
        :return: 
        """
        if self.controller is not None:
            self.controller.notify(summaries[-1])

        def summary_convert(summary, title):
            tf.summary.image(title, tf.expand_dims(tf.transpose(summary, [1, 2, 0]), axis=3),
                             max_outputs=Controller.max_outputs)

        # TODO please tell me there's a smarter way to this whole method
        n = self.mem_hp.mem_size
        r = self.mem_hp.num_read_heads
        summaries[0] = tf.reshape(tf.transpose(summaries[0], [0, 1, 3, 2]), [-1, self.batch_size, n * r])
        summaries[6] = tf.reshape(summaries[6],
                                  [-1, self.batch_size, self.mem_hp.num_read_heads * self.mem_hp.word_size])
        summaries[7] = tf.reshape(summaries[7], [-1, self.batch_size, self.mem_hp.num_read_heads * 3])
        summaries[14] = tf.reshape(tf.transpose(summaries[14], [0, 1, 3, 2]), [-1, self.batch_size, n * r])
        summaries[15] = tf.reshape(tf.transpose(summaries[15], [0, 1, 3, 2]), [-1, self.batch_size, n * r])
        summaries[17] = tf.reshape(summaries[17],
                                   [-1, self.batch_size, self.mem_hp.num_read_heads * self.mem_hp.word_size])

        summary_names = ["Read_weightings",
                         "Write_weighting",
                         "Usage_vector",
                         "Precedence_weighting",
                         "Memory_matrix",
                         "Link_matrix",
                         "R_read_vectors",
                         "R_read_modes",
                         "Write_gate",
                         "Allocation_gate",
                         "Write_strength",
                         "R_read_strengths",
                         "Erase_vector",
                         "Write_vector",
                         "Forward_weighting",
                         "Backward_weighting",
                         "R_free_gates",
                         "R_read_keys",
                         "Memory_output"
                         ]
        for i, summ_name in enumerate(summary_names):
            # don't want to visualise memory and link matrices since they're too big and impractical to analyze
            # would do it if perhaps there was a way to have a 2d matrix for each time step in tensorboard... hmm image?
            if i not in [4, 5, 19]:
                summary_convert(summaries[i], summ_name)
