import tensorflow as tf


class Memory:
    """
    Used by the DNC. It doesn't inherit Controller, but its functionality has some overlap.
    
    Controller  |  Memory
    -------------------------
    step()      |  step()
    __call__()  |  None
    
    Memory doesn't have the Controller's __call__() equivalent, because memory depends on DNC.
    
    """
    epsilon = 1e-6
    max_outputs = 2

    def __init__(self, batch_size, controller_output_size, out_vector_size, mem_hp, initializer=tf.random_normal,
                 initial_stddev=0.1):
        """
        
        :param batch_size: 
        :param controller_output_size: size of the controller output vector. Used for calculating the shape of the 
               interface weight matrix
        :param out_vector_size: size of the output of memory, after being passed through the output weight matrix
        :param mem_hp: hyperparameters for memory, object with attributes word_size, mem_size, num_read_heads 
        :param initial_stddev: 
        """
        self.batch_size = batch_size
        self.controller_output_size = controller_output_size
        self.out_vector_size = out_vector_size

        self.memory_size = mem_hp.mem_size
        self.word_size = mem_hp.word_size
        self.num_read_heads = mem_hp.num_read_heads

        self.interface_vector_size = (self.word_size * self.num_read_heads) + \
                                     5 * self.num_read_heads + 3 * self.word_size + 3

        self.interface_weights = tf.Variable(
            initializer([self.controller_output_size, self.interface_vector_size], stddev=initial_stddev),
            name="interface_weights")
        self.output_weights = tf.Variable(
            initializer([self.num_read_heads, self.word_size, self.out_vector_size], stddev=initial_stddev),
            name="output_weights_memory")

        # Code below is a more-or-less pythonic way to process the individual interface parameters
        r, w = self.num_read_heads, self.word_size
        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        names = ["r_read_keys", "r_read_strengths", "write_key", "write_strength", "erase_vector", "write_vector",
                 "r_free_gates", "allocation_gate", "write_gate", "r_read_modes"]
        functions = [tf.identity, Memory.oneplus, tf.identity, Memory.oneplus, tf.nn.sigmoid, tf.identity,
                     tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, self.reshape_and_softmax]

        indexes = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
        assert len(names) == len(sizes) == len(functions) == len(indexes)
        # Computing the interface dictionary from ordered lists of sizes, vector names and functions, as in the paper
        # self.split_interface is called later with the actual computed interface vector
        self.split_interface = lambda iv: {name: fn(iv[:, i[0]:i[1]]) for name, i, fn in
                                           zip(names, indexes, functions)}

    def init_memory(self, batch_size):
        """
        Returns the memory state for step 0. Used in DNC for the argument to tf.while_loop
        
        :return: 
        """
        read_weightings = tf.fill([batch_size, self.memory_size, self.num_read_heads], Memory.epsilon)
        write_weighting = tf.fill([batch_size, self.memory_size], Memory.epsilon, name="Write_weighting")
        precedence_weighting = tf.zeros([batch_size, self.memory_size], name="Precedence_weighting")
        m = tf.fill([batch_size, self.memory_size, self.word_size], Memory.epsilon)  # initial memory matrix
        usage_vector = tf.zeros([batch_size, self.memory_size], name="Usage_vector")
        link_matrix = tf.zeros([batch_size, self.memory_size, self.memory_size])
        read_vectors = tf.fill([batch_size, self.num_read_heads, self.word_size], Memory.epsilon)

        return [read_weightings, write_weighting, usage_vector, precedence_weighting, m, link_matrix, read_vectors]

    def step(self, controller_output, memory_state):
        """
        One step of memory. Performs all the read and write operations and returns some useful tensors for visualization
        
        :param controller_output: used to create the interface vector which parametrizes memory operations
        :param memory_state: list of a bunch of useful memory tensors from previous step which are necessary for 
            the calculation of next step
        :return: output of the memory, all the updated memory states and a bunch of of other tensors for visualization
        """

        read_weightings, write_weighting, usage_vector, precedence_weighting, m, link_matrix, read_vectors = \
            memory_state

        interface_vector = controller_output @ self.interface_weights  # shape [batch_size, interf_vector_size]

        interf = self.split_interface(interface_vector)

        interf["write_key"] = tf.expand_dims(interf["write_key"], dim=1)
        interf["r_read_keys"] = tf.reshape(interf["r_read_keys"], [-1, self.num_read_heads, self.word_size])

        # memory retention is of shape [batch_size, memory_size]
        memory_retention = tf.reduce_prod(1 - tf.einsum("br,bnr->bnr", interf["r_free_gates"], read_weightings), 2)

        # calculating the usage vector, which has some sort of short-term memory
        usage_vector = (usage_vector + write_weighting - usage_vector * write_weighting) * memory_retention
        allocation_weighting = self.calculate_allocation_weighting(usage_vector)

        # calculating write content weighting with memory from previous time step, shape [batch_size, memory_size, 1]
        # it represents a normalized probability distribution over the memory locations
        write_content_weighting = Memory.content_based_addressing(m, interf["write_key"], interf["write_strength"])

        # write weighting, shape [batch_size, memory_size], represents a normalized probability distribution over the
        # memory locations. Its sum is <= 1
        write_weighting = interf["write_gate"] * (interf["allocation_gate"] * allocation_weighting
                                                  + tf.einsum("bnr,bi->bn",
                                                              write_content_weighting,
                                                              (1 - interf["allocation_gate"])))

        m = m * (1 - tf.einsum("bn,bw->bnw", write_weighting, interf["erase_vector"])) + \
            tf.einsum("bn,bw->bnw", write_weighting, interf["write_vector"])

        link_matrix = self.update_link_matrix(link_matrix, precedence_weighting, write_weighting)
        precedence_weighting = tf.einsum("bn,b->bn",
                                         precedence_weighting,
                                         (1 - tf.reduce_sum(write_weighting, axis=1))) + write_weighting

        forwardw = tf.einsum("bmn,bnr->bmr", link_matrix, read_weightings)
        backwardw = tf.einsum("bnm,bnr->bmr", link_matrix, read_weightings)

        # calculating rcw with updated memory, shape [batch_size, memory_size, num_read_heads]
        # it represents a normalized probability distribution over the memory locations, for each read head
        read_content_weighting = Memory.content_based_addressing(m, interf["r_read_keys"],
                                                                 interf["r_read_strengths"])

        read_weightings = Memory.calculate_read_weightings(interf["r_read_modes"],
                                                           backwardw,
                                                           read_content_weighting,
                                                           forwardw)

        read_vectors = tf.einsum("bnw,bnr->brw", m, read_weightings)

        memory_output = tf.einsum("brw,rwo->bo", read_vectors, self.output_weights)

        extra_visualization_info = [interf["r_read_modes"], interf["write_gate"], interf["allocation_gate"],
                                    interf["write_strength"], interf["r_read_strengths"], interf["erase_vector"],
                                    interf["write_vector"], forwardw, backwardw, interf["r_free_gates"],
                                    interf["r_read_keys"]]

        memory_state = [read_weightings,
                        write_weighting,
                        usage_vector,
                        precedence_weighting,
                        m,
                        link_matrix,
                        read_vectors]
        return memory_output, memory_state, extra_visualization_info

    @staticmethod
    def calculate_read_weightings(r_read_modes, backwardw, read_content_weighting, forwardw):
        return tf.einsum("brs,bnrs->bnr", r_read_modes, tf.stack([backwardw, read_content_weighting, forwardw], axis=3))

    def calculate_allocation_weighting(self, usage_vector):
        """

        :param: usage vector: tensor of shape [batch_size, memory_size]
        :return: allocation tensor of shape [batch_size, memory_size]
        """
        usage_vector = Memory.epsilon + (1 - Memory.epsilon) * usage_vector

        # We're sorting the "-self.usage_vector" because top_k returns highest values and we need the lowest
        highest_usage, inverse_indices = tf.nn.top_k(-usage_vector, k=self.memory_size)
        lowest_usage = -highest_usage

        allocation_scrambled = (1 - lowest_usage) * tf.cumprod(lowest_usage, axis=1, exclusive=True)

        # allocation is not in the correct order. alloation[i] contains the sorted[i] value
        # reversing the already inversed indices for each batch
        indices = tf.stack([tf.invert_permutation(batch_indices) for batch_indices in tf.unstack(inverse_indices)])
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
        lm = (1 - w - w_transp) * link_matrix_old + tf.einsum("bn,bm->bmn", precedence_weighting_old, write_weighting)
        lm *= (1 - tf.eye(self.memory_size, batch_shape=[self.batch_size]))  # making sure self links are off
        return tf.identity(lm, name="Link_matrix")

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
        similarity = tf.einsum("bnw,brw,br->bnr", memory, keys, strength)
        content_weighting = tf.nn.softmax(similarity, dim=1, name="Content_weighting")

        return content_weighting

    @staticmethod
    def oneplus(x):
        return 1 + tf.nn.softplus(x)

    def reshape_and_softmax(self, r_read_modes):
        r_read_modes = tf.reshape(r_read_modes, [self.batch_size, self.num_read_heads, 3])
        return tf.nn.softmax(r_read_modes, dim=2)
