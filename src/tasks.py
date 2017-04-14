import numpy as np


class bAbITask:
    pass


class CopyTask:
    @staticmethod
    def generate_values(batch_size, vector_size, min_s, max_s, total_length):
        """

        :param batch_size:
        :param vector_size:
        :param sequence_length:
        :param total_length: total size of the sequence, in case we want to pad it, must be >= 2*max_s + 2
        :return: np array of size (batch_size x vector_size x total_length)
        """
        assert total_length >= 2 * max_s + 2
        shape = (batch_size, vector_size, total_length)
        inp_sequence = np.zeros(shape, dtype=np.float32)
        out_sequence = np.zeros(shape, dtype=np.float32)

        for i in range(batch_size):
            sequence_length = np.random.randint(min_s, max_s)
            ones = np.random.binomial(1, 0.5, (1, vector_size - 1, sequence_length))

            inp_sequence[i, :-1, :sequence_length] = ones
            out_sequence[i, :-1, sequence_length + 1:2 * sequence_length + 1] = ones

            inp_sequence[i, -1, sequence_length] = 1  # adding the marker, so the network knows when to start copying

        return np.array([inp_sequence, out_sequence])


if __name__ == "__main__":
    b = 3
    v = 5
    total = 10
    # min_s = 1
    # max_s = int((total - 2) / 2)
    val = CopyTask.generate_values(b, v, min_s, max_s, total)
    print(val)
