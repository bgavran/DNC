import tensorflow as tf


class Memory:
    epsilon = 1e-6

    def __init__(self, batch_size, inp_vector_size, out_vector_size, mem_size):
        self.batch_size = batch_size
        self.inp_vector_size = inp_vector_size
        self.out_vector_size = out_vector_size
        self.memory_size = mem_size
        self.interface_weights = tf.Variable(tf.random_normal([self.inp_vector_size, self.memory_size], stddev=0.1),
                                             name="memory_weights")
        self.output_weights = tf.Variable(tf.random_normal([self.memory_size, self.out_vector_size], stddev=0.1),
                                          name="output_weights")
        mem_interf_weights = tf.expand_dims(tf.expand_dims(self.interface_weights, axis=0), axis=3)
        tf.summary.image("Memory interface weights", tf.transpose(mem_interf_weights), max_outputs=self.batch_size)

        self.m = tf.fill([self.batch_size, self.memory_size], Memory.epsilon)  # initial memory matrix

    def __call__(self, controller_output):
        self.m += tf.nn.tanh(tf.matmul(controller_output, self.interface_weights))
        output = tf.matmul(self.m, self.output_weights)
        return tf.nn.tanh(output), self.m
