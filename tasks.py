import numpy as np


class CopyTask:
    @staticmethod
    def generate_values(batch_size, vector_size, sequence_length):
        """

        :param batch_size:
        :param vector_size:
        :param sequence_length:
        :return: np array of size (batch_size x vector_size x 2*sequence_length + 2
        """
        shape = (batch_size, vector_size, 2 * sequence_length + 2)
        inp_sequence = np.zeros(shape, dtype=np.float32)
        out_sequence = np.zeros(shape, dtype=np.float32)

        ones = np.random.binomial(1, 0.5, (batch_size, vector_size - 1, sequence_length))

        inp_sequence[:, :-1, :sequence_length] = ones
        out_sequence[:, :-1, sequence_length + 1:-1] = ones

        inp_sequence[:, -1, sequence_length] = 1  # adding the marker for the network, so it knows when to start copying

        return inp_sequence, out_sequence
