import tensorflow as tf


class Memory:
    epsilon = 1e-6

    def __init__(self, batch_size, interface_vector_size, out_vector_size, mem_size):
        self.batch_size = batch_size
        self.interface_vector_size = interface_vector_size
        self.out_vector_size = out_vector_size
        self.memory_size = mem_size

        # made up thing, not even close to the real memory implementation
        self.interf_to_mem_weights = tf.Variable(
            tf.random_normal([self.interface_vector_size, self.memory_size], stddev=0.1),
            name="memory_weights")
        self.mem_to_output_weights = tf.Variable(
            tf.random_normal([self.memory_size, self.out_vector_size], stddev=0.1),
            name="output_weights")

        mem_interf_weights = tf.expand_dims(tf.expand_dims(self.interf_to_mem_weights, axis=0), axis=3)
        tf.summary.image("Memory interface weights", tf.transpose(mem_interf_weights), max_outputs=self.batch_size)

        self.m = tf.fill([self.batch_size, self.memory_size], Memory.epsilon)  # initial memory matrix

    def __call__(self, interface_vector):
        self.m += interface_vector @ self.interf_to_mem_weights
        output = self.m @ self.mem_to_output_weights
        return tf.nn.tanh(output), self.m
