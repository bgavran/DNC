from dnc import *
from lstm import *
from feedforward import *
from utils import *


class Hp:
    batch_size = 4
    inp_vector_size = 3
    out_vector_size = inp_vector_size
    lstm_memory_size = 1
    total_output_length = 20
    min_seq = 1
    max_seq = int((total_output_length - 2) / 2)

    steps = 100000
    path = DataPath("train_summary").path


FF_controller = Feedforward(Hp.inp_vector_size, Hp.out_vector_size, Hp.total_output_length, Hp.batch_size, 3,
                            [128, 128, Hp.out_vector_size])
LSTM_controller = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.out_vector_size, Hp.lstm_memory_size,
                       Hp.total_output_length)
dnc = DNC(LSTM_controller)

x = tf.placeholder(tf.float32, [None, Hp.inp_vector_size, Hp.total_output_length], name="X")
y = tf.placeholder(tf.float32, [None, Hp.inp_vector_size, Hp.total_output_length], name="Y")

task = CopyTask
optimizer = tf.train.AdamOptimizer()
dnc.run_session(x, y, task, Hp, optimizer=optimizer)
