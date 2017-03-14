import tensorflow as tf
from controller import *


class LSTM(Controller):
    def __init__(self, batch_size, input_size, output_size, memory_size, max_seq_length):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.max_seq_length = max_seq_length
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.memory_size)

        # Extra layer of neural network
        # The output is a vector of dimension memory_size, but we want it to be output_size, so we multiply it by some W
        self.weights = tf.Variable(tf.random_normal([self.memory_size, self.output_size], stddev=0.1))
        self.biases = tf.Variable(tf.zeros([self.output_size]))

    def __call__(self, inputs, sequence_length=None):
        outputs, state = tf.contrib.rnn.static_rnn(self.lstm_cell, inputs, sequence_length=sequence_length,
                                                   dtype=tf.float32)
        outputs = [tf.matmul(o, self.weights) + self.biases for o in outputs]
        outputs = tf.transpose(outputs, [1, 2, 0])
        return outputs, state
