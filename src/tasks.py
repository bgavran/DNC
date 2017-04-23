import numpy as np


class Task:
    """
    
    """

    def generate_data(self):
        """
        
        :return: 
        """
        # TODO define the interaction between the abstract task and the controller!
        raise NotImplementedError()


class bAbITask:
    pass


class CopyTask:
    epsilon = 1e-2

    def __init__(self, inp_vector_size, out_vector_size, total_output_length, batch_size, min_seq, theoretical_max_seq):
        self.inp_vector_size = inp_vector_size
        self.out_vector_size = out_vector_size
        self.total_output_length = total_output_length
        self.batch_size = batch_size
        self.min_seq = min_seq
        self.theoretical_max_seq = theoretical_max_seq

        self.max_seq = self.min_seq

        self.x_shape = [self.batch_size, self.inp_vector_size, self.total_output_length]
        self.y_shape = [self.batch_size, self.inp_vector_size, self.total_output_length]

        # Used for curriculum training
        self.state = 0
        self.consecutive_thresh = 10

    def update_training_state(self, cost):
        if cost <= CopyTask.epsilon:
            self.state += 1
        else:
            self.state = 0

    def check_next_level(self):
        if self.state < self.consecutive_thresh:
            return False
        else:
            return True

    def next_level(self):
        self.state = 0
        if self.max_seq < self.theoretical_max_seq:
            self.max_seq += 1
        print("Increased max_seq to", self.max_seq)

    def update_state(self, cost):
        self.update_training_state(cost)
        if self.check_next_level():
            self.next_level()

    def generate_data(self, cost):
        # Update curriculum training state
        self.update_state(cost)

        # self.max_seq = self.theoretical_max_seq

        return CopyTask.generate_values(self.batch_size, self.inp_vector_size, self.min_seq, self.max_seq,
                                        self.total_output_length)

    @staticmethod
    def generate_values(batch_size, vector_size, min_s, max_s, total_length):
        """

        :param batch_size:
        :param vector_size:
        :param min_s:
        :param max_s:
        :param total_length: total size of the sequence, in case we want to pad it, must be >= 2*max_s + 2
        :return: np array of shape [batch_size, vector_size, total_length]
        """
        assert total_length >= 2 * max_s + 2
        shape = (batch_size, vector_size, total_length)
        inp_sequence = np.zeros(shape, dtype=np.float32)
        out_sequence = np.zeros(shape, dtype=np.float32)

        for i in range(batch_size):
            sequence_length = np.random.randint(min_s, max_s + 1)
            ones = np.random.binomial(1, 0.5, (1, vector_size - 1, sequence_length))

            inp_sequence[i, :-1, :sequence_length] = ones
            out_sequence[i, :-1, sequence_length + 1:2 * sequence_length + 1] = ones

            inp_sequence[i, -1, sequence_length] = 1  # adding the marker, so the network knows when to start copying

        return np.array([inp_sequence, out_sequence])


if __name__ == "__main__":
    b = 3
    v = 5
    total = 10
    min_s = 1
    max_s = int((total - 2) / 2)
    val = CopyTask.generate_values(b, v, min_s, max_s, total)
    print(val)
