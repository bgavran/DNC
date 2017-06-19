import numpy as np
import tensorflow as tf
from controller_implementations.dnc.dnc import DNC, Memory
from controller_implementations.feedforward import Feedforward


class DNCTest(tf.test.TestCase):
    def test_read_weighting(self):
        b, n, r = 2, 5, 7
        backwardw = np.random.rand(b, n, r)
        forwardw = np.random.rand(b, n, r)
        read_content_weighting = np.random.rand(b, n, r)
        r_read_modes = np.random.rand(b * r * 3)

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        r_read_modes_correct = np.reshape(r_read_modes, [b, r, 3]).copy()
        for i in range(b):
            for j in range(r):
                r_read_modes_correct[i, j, :] = softmax(r_read_modes_correct[i, j, :])
                forwardw[i, :, j] = softmax(forwardw[i, :, j])
                backwardw[i, :, j] = softmax(backwardw[i, :, j])
                read_content_weighting[i, :, j] = softmax(read_content_weighting[i, :, j])

        np.testing.assert_allclose(np.sum(r_read_modes_correct, axis=2), np.ones((b, r)))

        backward_sum = np.expand_dims(r_read_modes_correct[:, :, 0], axis=1) * backwardw
        content_sum = np.expand_dims(r_read_modes_correct[:, :, 1], axis=1) * read_content_weighting
        forward_sum = np.expand_dims(r_read_modes_correct[:, :, 2], axis=1) * forwardw
        wcw_correct = backward_sum + content_sum + forward_sum
        with self.test_session():
            tf.global_variables_initializer().run()
            r_read_modes_tf = tf.constant(r_read_modes)
            backwardw = tf.constant(backwardw)
            read_content_weighting = tf.constant(read_content_weighting)
            forwardw = tf.constant(forwardw)

            Memory.batch_size = b
            Memory.num_read_heads = r
            r_read_modes_tf = Memory.reshape_and_softmax(Memory, r_read_modes_tf)
            self.assertAllClose(np.sum(r_read_modes_tf.eval(), axis=2), np.ones((b, r)))
            self.assertAllClose(r_read_modes_tf.eval(), r_read_modes_correct)
            wcw_calculated = Memory.calculate_read_weightings(r_read_modes_tf,
                                                              backwardw,
                                                              read_content_weighting,
                                                              forwardw)
            wcw_calculated = wcw_calculated.eval()
            self.assertAllClose(wcw_correct, wcw_calculated)
            self.assertAllClose(np.sum(wcw_calculated, axis=1), np.ones((b, r)))

    def test_link_matrix(self):
        b, n = 2, 5
        write_weighting = np.random.rand(b, n)
        precedence_weighting = np.random.rand(b, n)  # precedence weighting from previous time step
        link_matrix_old = np.random.rand(b, n, n) * (
            1 - np.tile(np.eye(5), [b, 1, 1]))  # random link matrix with diagonals zero
        link_matrix_correct = np.zeros((b, n, n))
        for k in range(b):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        link_matrix_correct[k, i, j] = (1 - write_weighting[k, i] - write_weighting[k, j]) * \
                                                       link_matrix_old[k, i, j] + \
                                                       write_weighting[k, i] * precedence_weighting[k, j]

        with self.test_session():
            tf.global_variables_initializer().run()
            Memory.batch_size = b
            Memory.memory_size = n
            new_link_matrix = Memory.update_link_matrix(Memory,
                                                        tf.constant(link_matrix_old, dtype=tf.float32),
                                                        tf.constant(precedence_weighting, dtype=tf.float32),
                                                        tf.constant(write_weighting, dtype=tf.float32))
            self.assertAllClose(link_matrix_correct, new_link_matrix.eval())

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

    if __name__ == '__main__':
        tf.test.main()
