import tensorflow as tf


class Memory:
    """
    Used by the DNC. It's not related to the Controller in any way, just used by it.
    
    Controller  |  Memory
    -------------------------
    step()      |  __call__()
    __call__()  |  None
    
    Its step() method equivalent is the __call__() method, which implements all the required operations for one time 
    step.
    Memory doesn't have the Controller's __call__() equivalent, because it depends on DNC.
    
    
    """
    epsilon = 1e-6

    def __init__(self, batch_size, out_vector_size, mem_hp):
        self.batch_size = batch_size
        self.out_vector_size = out_vector_size

        self.memory_size = mem_hp.mem_size
        self.word_size = mem_hp.word_size
        self.num_read_heads = mem_hp.num_read_heads

        self.interface_vector_size = (self.word_size * self.num_read_heads) + \
                                     5 * self.num_read_heads + 3 * self.word_size + 3

        self.interface_weights = tf.Variable(
            tf.random_normal([self.out_vector_size, self.interface_vector_size], stddev=0.01),
            name="interface_weights")
        self.output_weights = tf.Variable(
            tf.random_normal([self.num_read_heads, self.word_size, self.out_vector_size], stddev=0.01),
            name="output_weights")

        # TODO Reconsider other initialization schemes? Why should we initialize to 1e-6?
        self.m = tf.fill([self.batch_size, self.memory_size, self.word_size], Memory.epsilon)  # initial memory matrix
        self.read_vectors = tf.fill([self.batch_size, self.num_read_heads, self.word_size], Memory.epsilon)
        self.write_weighting = tf.fill([self.batch_size, self.memory_size], Memory.epsilon)
        self.read_weightings = tf.fill([self.batch_size, self.memory_size, self.num_read_heads], Memory.epsilon)
        self.precedence_weighting = tf.zeros([self.batch_size, self.memory_size])
        self.link_matrix = tf.zeros([self.batch_size, self.memory_size, self.memory_size])
        self.usage_vector = tf.zeros([self.batch_size, self.memory_size])

        # Code below is a more-or-less pythonic way to process the individual interface parameters
        r, w = self.num_read_heads, self.word_size
        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        names = ["r_read_keys", "r_read_strengths", "write_key", "write_strength", "erase_vector", "write_vector",
                 "r_free_gates", "allocation_gate", "write_gate", "r_read_modes"]
        functions = [tf.identity, Memory.oneplus, tf.identity, Memory.oneplus, tf.nn.sigmoid, tf.identity,
                     tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.softmax]

        indexes = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
        assert len(names) == len(sizes) == len(functions) == len(indexes)
        # Computing the interface dictionary from ordered lists of sizes, vector names and functions, as in the paper
        # self.split_interface is called later with the actual computed interface vector
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

        # calculating write content weighting with memory from previous time step, shape [batch_size, memory_size, 1]
        # it represents a normalized probability distribution over the memory locations
        write_content_weighting = Memory.content_based_addressing(self.m, interf["write_key"], interf["write_strength"])

        ag = interf["allocation_gate"]

        # memory retention is of shape [batch_size, memory_size]
        tf.Print(interf["r_free_gates"], [interf["r_free_gates"]], message="Free gates")
        memory_retention = tf.reduce_prod(1 - tf.einsum("br,bnr->bnr", interf["r_free_gates"], self.read_weightings), 2)
        tf.Print(memory_retention, [memory_retention], message="Memory retention")
        # calculating the usage vector, which has some sort of short-term memory
        self.usage_vector = (self.usage_vector + self.write_weighting - self.usage_vector * self.write_weighting) * \
                            memory_retention
        tf.Print(self.usage_vector, [self.usage_vector], message="Usage vector")
        self.allocation_weighting = self.calculate_allocation_weighting()

        # write weighting, shape [batch_size, memory_size], represents a normalized probability distribution over the
        # memory locations. Its sum is <= 1
        self.write_weighting = interf["write_gate"] * (ag * self.allocation_weighting
                                                       + tf.einsum("bi,bnr->bn", (1 - ag), write_content_weighting))
        self.m = self.m * (1 - tf.einsum("bw,bn->bnw", interf["erase_vector"], self.write_weighting)) + \
                 tf.einsum("bw,bn->bnw", interf["write_vector"], self.write_weighting)

        # calculating rcw with updated memory, shape [batch_size, memory_size, num_read_heads]
        # it represents a normalized probability distribution over the memory locations, for each read head
        read_content_weighting = Memory.content_based_addressing(self.m, interf["r_read_keys"],
                                                                 interf["r_read_strengths"])
        self.link_matrix = self.update_link_matrix(self.link_matrix, self.precedence_weighting, self.write_weighting)
        self.precedence_weighting = self.precedence_weighting * (1 - tf.reduce_sum(self.write_weighting)) + \
                                    self.write_weighting

        # TODO test forwardw and backwardw validity?
        forwardw = tf.einsum("bmn,bmr->bmr", self.link_matrix, self.read_weightings)
        backwardw = tf.einsum("bnm,bmr->bnr", self.link_matrix, self.read_weightings)

        self.read_weightings = tf.einsum("brs,bmrs->bmr",
                                         tf.reshape(interf["r_read_modes"], [-1, self.num_read_heads, 3]),
                                         tf.stack([backwardw, read_content_weighting, forwardw], axis=3))
        tf.Print(read_content_weighting, [read_content_weighting], message="Read content weighting")
        tf.Print(self.read_weightings, [self.read_weightings], message="Read weightings")

        self.read_vectors = tf.einsum("bnw,bnr->brw", self.m, self.read_weightings)

        memory_output = tf.einsum("rwo,brw->bo", self.output_weights, self.read_vectors)
        memory_state_tuple = (interf["r_read_modes"],  # 3
                              interf["write_gate"],  # 1
                              interf["allocation_gate"],  # 1
                              tf.squeeze(interf["write_key"]),  # word_size
                              tf.squeeze(write_content_weighting),  # memory_size
                              tf.reshape(interf["r_read_keys"],
                                         [self.batch_size, self.num_read_heads * self.word_size]),  # r * word_size
                              tf.reshape(self.read_vectors,
                                         [self.batch_size, self.num_read_heads * self.word_size])  # r * word_size
                              # tf.reshape(self.m,
                              #            [self.batch_size, self.memory_size * self.word_size])  # n * word_size
                              )

        # Adding a visual delimeter of tf.ones in between every memory state element. Hacking with list's + overloading
        memory_state = tf.concat(sum([[i, tf.ones([self.batch_size, 1])] for i in memory_state_tuple], []), axis=1)

        return memory_output, memory_state

    def calculate_allocation_weighting(self):
        """
        Calculating allocation weighting, extracted to a method because it's complicated to perform in Memory.__call__()
        
        :return: allocation tensor of shape [batch_size, memory_size]
        """
        # numerical stability
        self.usage_vector = Memory.epsilon + (1 - Memory.epsilon) * self.usage_vector

        # We're sorting the "1 - self.usage_vector" because top_k returns highest values and we need the lowest
        lowest_usage, inverse_indices = tf.nn.top_k(1 - self.usage_vector, k=self.memory_size)
        sorted_usage = 1 - lowest_usage

        # lowest_usage is equal to the paper's (1 - usage[sorted[j]])
        # tf.Print(sorted_usage, [sorted_usage], message="Sorted usage")
        # tf.Print(lowest_usage, [lowest_usage], message="Lowest usage")
        allocation_scrambled = lowest_usage * tf.cumprod(sorted_usage, axis=1, exclusive=True)

        # allocation is not in the correct order. alloation[i] contains the sorted[i] value
        # reversing the already inversed indices for each batch
        indices = tf.stack([tf.invert_permutation(ind) for ind in tf.unstack(inverse_indices)])
        allocation = tf.stack([tf.gather(mem, ind)
                               for mem, ind in
                               zip(tf.unstack(allocation_scrambled), tf.unstack(indices))])
        return allocation

    def update_link_matrix(self, link_matrix_old, precedence_weighting_old, write_weighting):
        """
        Updating the link matrix takes some effort (in order to vectorize the implementation)
        Instead of the original index-by-index operation, it's all done at once.
        
        
        :param link_matrix_old: from previous time step, shape [batch_size, memory_size, memory_size]
        :param precedence_weighting_old: from previous time step, shape [batch_size, memory_size]
        :param write_weighting: from current time step, shape [batch_size, memory_size]
        :return: updated link matrix
        """
        expanded = tf.expand_dims(write_weighting, axis=2)

        # vectorizing the paper's original implementation
        w = tf.tile(expanded, [1, 1, self.memory_size])  # shape [batch_size, memory_size, memory_size]
        # shape of w_transpose is the same: [batch_size, memory_size, memory_size]
        w_transp = tf.tile(tf.transpose(expanded, [0, 2, 1]), [1, self.memory_size, 1])

        # in einsum, m and n are the same dimension because tensorflow doesn't support duplicated subscripts. Why?
        return (1 - w - w_transp) * link_matrix_old + tf.einsum("bm,bn->bmn", write_weighting, precedence_weighting_old)

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
