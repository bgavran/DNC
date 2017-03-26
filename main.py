from lstm import *


class Hp:
    batch_size = 5
    input_size = 3
    output_size = input_size
    memory_size = 8
    total_seq_length = 20
    min_s = 1
    max_s = int((total_seq_length - 2) / 2)

    steps = 100000
    logdir = "train_summary"


controller = LSTM(Hp.batch_size, Hp.input_size, Hp.output_size, Hp.memory_size, Hp.total_seq_length)

x = tf.placeholder(tf.float32, [None, Hp.input_size, Hp.total_seq_length], name="X")
y = tf.placeholder(tf.float32, [None, Hp.input_size, Hp.total_seq_length], name="Y")

optimizer = tf.train.AdamOptimizer()

controller.run_session(x, y, optimizer, Hp)
