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
        self.output_weights = tf.Variable(
            tf.random_normal([self.num_read_heads, self.word_size, self.out_vector_size], stddev=0.01),
            name="output_weights")

        self.m = tf.fill([self.batch_size, self.memory_size, self.word_size], Memory.epsilon)  # initial memory matrix
        self.read_vectors = tf.fill([self.batch_size, self.num_read_heads, self.word_size], Memory.epsilon)
        # self.write_weighting = tf.fill([self.batch_size, self.memory_size], Memory.epsilon)
        # self.read_weightings = tf.fill([self.batch_size, self.memory_size], Memory.epsilon)

        r, w = self.num_read_heads, self.word_size
        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        indexes = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes[:-1]))]
        names = ["r_read_keys", "r_read_strengths", "write_key", "write_strength", "erase_vector", "write_vector",
                 "r_free_gates", "allocation_gate", "write_gate", "r_read_modes"]
        functions = [tf.identity, Memory.oneplus, tf.identity, Memory.oneplus, tf.nn.sigmoid, tf.identity,
                     tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.softmax]
        assert len(names) == len(sizes) == len(functions) == len(indexes) + 1
        # Computing the interface dictionary from ordered lists of sizes, vector names and functions, as in the paper
        self.split_interface = lambda iv: {name: fn(iv[:, i[0]:i[1]]) for name, i, fn in
                                           zip(names, indexes, functions)}

    def __call__(self, controller_output):
        """
        
        :param controller_output: 
        :return: 
        """
        interface_vector = controller_output @ self.interface_weights  # shape [batch_size, interf_vector_size]

        interf = self.split_interface(interface_vector)

        interf["write_key"] = tf.expand_dims(interf["write_key"], dim=1)
        interf["r_read_keys"] = tf.reshape(interf["r_read_keys"], [-1, self.num_read_heads, self.word_size])

        # calculating wcw with memory from previous time step, not updated, shape [batch_size, memory_size, 1]
        # it represents a normalized probability distribution over the memory locations
        write_content_weighting = Memory.content_based_addressing(self.m, interf["write_key"], interf["write_strength"])

        ag = interf["allocation_gate"]

        # write weighting, shape [batch_size, memory_size], represents a normalized probability distribution over the
        # memory locations
        # TODO Add allocation weighting
        self.write_weighting = interf["write_gate"] * (ag * 0
                                                       + tf.einsum("bi,bnr->bn", (1 - ag), write_content_weighting))
        self.m = self.m * (1 - tf.einsum("bw,bn->bnw", interf["erase_vector"], self.write_weighting)) + \
                 tf.einsum("bw,bn->bnw", interf["write_vector"], self.write_weighting)

        # calculating rcw with updated memory, shape [batch_size, memory_size, num_read_heads]
        # it represents a normalized probability distribution over the memory locations, for each read head
        read_content_weighting = Memory.content_based_addressing(self.m, interf["r_read_keys"],
                                                                 interf["r_read_strengths"])

        # TODO Update read weighting with forward and backward weightings
        read_weighting = read_content_weighting
        self.read_vectors = tf.einsum("bnw,bnr->brw", self.m, read_weighting)

        memory_output = tf.einsum("rwo,brw->bo", self.output_weights, self.read_vectors)
        memory_state_tuple = (interf["write_gate"],  # 1
                              interf["allocation_gate"],  # 1
                              interf["write_strength"],
                              tf.squeeze(interf["write_key"]),  # word_size
                              tf.squeeze(write_content_weighting),  # memory_size
                              tf.reshape(interf["r_read_keys"],
                                         [self.batch_size, self.num_read_heads * self.word_size]),  # r * word_size
                              tf.reshape(self.read_vectors,
                                         [self.batch_size, self.num_read_heads * self.word_size]),  # r * word_size
                              tf.reshape(self.m,
                                         [self.batch_size, self.memory_size * self.word_size])  # n * word_size
                              )

        # Adding a visual delimeter of tf.ones in between every memory state element. Hacking with  list's + overloading
        memory_state = tf.concat(sum([[i, tf.ones([self.batch_size, 1])] for i in memory_state_tuple], []), axis=1)

        return memory_output, memory_state

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
        similarity = tf.einsum("brw,bnw,br->bnr", keys, memory, strength)
        return tf.nn.softmax(similarity, dim=1)

    @staticmethod
    def oneplus(x):
        return 1 + tf.nn.softplus(x)
