import tensorflow as tf


class Memory:
    epsilon = 1e-6

    def __init__(self, batch_size, out_vector_size, mem_hp):
        self.batch_size = batch_size
        self.out_vector_size = out_vector_size

        self.memory_size = mem_hp.mem_size
        self.word_size = mem_hp.word_size
        self.num_read_heads = mem_hp.num_read_heads

        self.interface_vector_size = (self.word_size * self.memory_size) + \
                                     5 * self.num_read_heads + 3 * self.word_size + 3
        self.interface_weights = tf.Variable(
            tf.random_normal([self.out_vector_size, self.interface_vector_size], stddev=0.01),
            name="interface_weights")

        # made up thing, not even close to the real memory implementation
        self.mem_to_output_weights = tf.Variable(
            tf.random_normal([self.memory_size, self.out_vector_size], stddev=0.1),
            name="output_weights")

        self.m = tf.fill([self.batch_size, self.memory_size, self.word_size], Memory.epsilon)  # initial memory matrix

    def __call__(self, controller_output):
        """
        
        :param controller_output: 
        :return: 
        """
        interface_vector = controller_output @ self.interface_weights  # shape [batch_size, interf_vector_size]

        r, w = self.num_read_heads, self.word_size
        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        indexes = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes[:-1]))]
        names = ["r_read_keys", "r_read_strengths", "write_key", "write_strength", "erase_vector", "write_vector",
                 "r_free_gates", "allocation_gate", "write_gate", "r_read_modes"]
        functions = [tf.identity, Memory.oneplus, tf.identity, Memory.oneplus, tf.nn.sigmoid, tf.identity,
                     tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.softmax]

        assert len(names) == len(sizes) == len(functions) == len(indexes) + 1
        # Computing the interface dictionary from ordered lists of sizes, vector names and functions, as in the paper
        interf = {name: fn(interface_vector[:, i[0]:i[1]]) for name, i, fn in zip(names, indexes, functions)}

        interf["write_key"] = tf.expand_dims(interf["write_key"], dim=1)
        # interf["write_strength"] = tf.expand_dims(interf["write_strength"], dim=1)

        write_content_weighting = Memory.content_based_addressing(self.m, interf["write_key"], interf["write_strength"])

        # Fix this, add allocation weighting
        ag = interf["allocation_gate"]
        write_weighting = interf["write_gate"] * (ag * 0
                                                  + tf.einsum("bi,bnr->bn", (1 - ag), write_content_weighting))
        self.m = self.m * (1 - tf.einsum("bw,bn->bnw", interf["erase_vector"], write_weighting)) + \
                 tf.einsum("bw,bn->bnw", interf["write_vector"], write_weighting)

        output = self.m @ self.mem_to_output_weights
        return tf.nn.tanh(output), self.m

    @staticmethod
    def content_based_addressing(memory, keys, strength):
        """
        
        :param memory: array of shape [batch_size, memory_size, word_size], i.e. [b, n, w]
        :param keys: array of shape [batch_size, n_keys, word_size], i.e. [b, r, w]
        :param strength: array of shape [batch_size, n_keys], i.e. [b, r]
        :return: tensor of shape [batch_size, memory_size, n_keys]
        """
        keys = tf.nn.l2_normalize(keys, dim=2)
        memory = tf.nn.l2_normalize(memory, dim=2)
        # Einstein summation convention. Functionality has to be tested
        similarity = tf.einsum("brw,bnw,bn->bnr", keys, memory, strength)
        return tf.nn.softmax(similarity, dim=1)

    @staticmethod
    def oneplus(x):
        return 1 + tf.nn.softplus(x)
