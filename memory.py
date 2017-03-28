import tensorflow as tf


class Memory:
    epsilon = 1e-6

    def __init__(self, batch_size, input_size, output_size, mem_size):
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.interface_weights = tf.Variable(tf.random_normal([self.input_size, self.mem_size], stddev=0.1),
                                             name="memory_weights")
        self.output_weights = tf.Variable(tf.random_normal([self.mem_size, self.output_size], stddev=0.1),
                                          name="output_weights")

        self.m = tf.fill([self.batch_size, self.mem_size], Memory.epsilon),  # initial memory matrix

    def __call__(self, controller_output):
        self.m += tf.matmul(controller_output, self.interface_weights)
        return tf.matmul(self.m, self.output_weights)
