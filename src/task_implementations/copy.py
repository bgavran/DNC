import numpy as np
from tasks import Task


class CopyTask(Task):
    epsilon = 1e-2

    def __init__(self, vector_size, batch_size, min_seq, train_max_seq, n_copies):
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.min_seq = min_seq
        self.train_max_seq = train_max_seq
        self.n_copies = n_copies

        self.max_seq_curriculum = self.min_seq + 1

        self.x_shape = [self.batch_size, self.vector_size, None]
        self.y_shape = [self.batch_size, self.vector_size, None]

        # Used for curriculum training
        self.state = 0
        self.consecutive_thresh = 10

    def update_training_state(self, cost):
        if cost <= CopyTask.epsilon:
            self.state += 1
        else:
            self.state = 0

    def check_lesson_learned(self):
        if self.state < self.consecutive_thresh:
            return False
        else:
            return True

    def next_lesson(self):
        self.state = 0
        # if self.max_seq_curriculum < self.train_max_seq:
        #     self.max_seq_curriculum += 1
        #     print("Increased max_seq to", self.max_seq_curriculum)
        if self.n_copies < 5:
            self.n_copies += 1
            print("Increased n_copies to", self.n_copies)
        else:
            print("Done with the training!!!")

    def update_state(self, cost):
        self.update_training_state(cost)
        if self.check_lesson_learned():
            self.next_lesson()

    def generate_data(self, cost):
        # Update curriculum training state
        self.update_state(cost)

        self.max_seq_curriculum = self.train_max_seq

        return CopyTask.generate_n_copies(self.batch_size, self.vector_size, self.min_seq, self.max_seq_curriculum,
                                          self.n_copies)

    @staticmethod
    def generate_n_copies(batch_size, inp_vector_size, min_seq, max_seq, n_copies):
        copies_list = [
            CopyTask.generate_copy_pair(batch_size, inp_vector_size, min_seq, max_seq)
            for _ in range(n_copies)]
        output = np.concatenate([i[0] for i in copies_list], axis=3)
        total_length = np.sum([i[1] for i in copies_list])
        return output, total_length

    @staticmethod
    def generate_copy_pair(batch_size, vector_size, min_s, max_s):
        """

        :param batch_size:
        :param vector_size:
        :param min_s:
        :param max_s:
        :return: np array of shape [batch_size, vector_size, total_length]
        """
        sequence_length = np.random.randint(min_s, max_s + 1)
        total_length = 2 * sequence_length + 2

        shape = (batch_size, vector_size, total_length)
        inp_sequence = np.zeros(shape, dtype=np.float32)
        out_sequence = np.zeros(shape, dtype=np.float32)

        for i in range(batch_size):
            ones = np.random.binomial(1, 0.5, (1, vector_size - 1, sequence_length))

            inp_sequence[i, :-1, :sequence_length] = ones
            out_sequence[i, :-1, sequence_length + 1:2 * sequence_length + 1] = ones

            inp_sequence[i, -1, sequence_length] = 1  # adding the marker, so the network knows when to start copying

        return np.array([inp_sequence, out_sequence]), total_length


if __name__ == "__main__":
    b = 2
    v = 3
    total = 10
    min_s = 1
    max_s = int((total - 2) / 2)
    val = CopyTask.generate_copy_pair(b, v, min_s, max_s, total)
    print(val)
