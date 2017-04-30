import numpy as np
import tensorflow as tf
from controller_implementations.dnc.dnc import DNC, Memory
from controller_implementations.feedforward import Feedforward


class DNCTest(tf.test.TestCase):
    def test_allocation_weighting(self):
        b, n = 5, 10
        u = np.random.rand(b, n)
        s = np.argsort(u, axis=1)

        correct_alloc = np.zeros((b, n)).astype(np.float32)
        for i in range(b):
            cp = np.concatenate([[1], np.cumprod(u[i][s[i]])[:-1]])
            correct_alloc[i][s[i]] = (1 - u[i][s[i]]) * cp

        with self.test_session():
            tf.global_variables_initializer().run()
            Memory.memory_size = n
            calculated_alloc = Memory.calculate_allocation_weighting(Memory, u).eval()
            self.assertAllClose(correct_alloc, calculated_alloc)

    def test_content_addressing(self):
        batch_size = 1
        memory_size = 5
        word_size = 6
        memory_array = np.array([[[0, 1, 0, 0, 1, 0],
                                  [0, 1, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [1, 0, 1, 1, 0, 0]]])
        memory = tf.constant(memory_array, dtype=tf.float32)  # initial memory matrix
        self.assertEqual([batch_size, memory_size, word_size], memory.shape)
        strength = tf.ones([batch_size, 1], dtype=tf.float32)
        keys_array = np.array([[[1, 0, 1, 1, 0, 0]]])
        keys = tf.constant(keys_array, dtype=tf.float32)
        self.assertEqual([batch_size, 1, word_size], keys.shape)

        with self.test_session():
            tf.global_variables_initializer().run()

            real_weighting = np.zeros((batch_size, memory_size))
            k = keys[0, 0, :].eval()
            for i in range(memory_size):
                m = memory[0, i, :].eval()
                real_weighting[0, i] = DNCTest.cosine_similarity(k, m)
            real_weighting = np.exp(real_weighting) / np.sum(np.exp(real_weighting))
            DNC_weighting = Memory.content_based_addressing(memory, keys, strength).eval()[:, :, 0]
            self.assertAllClose(real_weighting, DNC_weighting)

    @staticmethod
    def cosine_similarity(a, b):
        assert a.shape == b.shape and len(a.shape) == 1
        norm = np.linalg.norm
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-6)

    def test_write_weighting_less_equal(self):
        class Hp:
            batch_size = 2
            inp_vector_size = 2
            out_vector_size = inp_vector_size
            lstm_memory_size = 50
            total_output_length = 20
            min_seq = 1
            max_seq = int((total_output_length - 2) / 2)
            steps = 1000000

            class Mem:
                word_size = 8
                mem_size = 16
                num_read_heads = 4

        ff = Feedforward(Hp.inp_vector_size, Hp.out_vector_size, Hp.batch_size, [128, 128, Hp.out_vector_size])
        dnc = DNC(ff, Hp.Mem)
        x = tf.ones([Hp.batch_size, Hp.inp_vector_size])
        output, _ = dnc.step(x, 0)
        with self.test_session():
            tf.global_variables_initializer().run()
            output.eval()
            for item in np.nditer(dnc.memory.write_weighting.eval()):
                self.assertLessEqual(item, 1)

    if __name__ == '__main__':
        tf.test.main()
