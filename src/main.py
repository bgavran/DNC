from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy import *
from utils import project_path


class Hp:
    """
    Hyperparameters
    """
    batch_size = 1

    n_blocks = 6
    inp_vector_size = n_blocks + 1
    out_vector_size = inp_vector_size
    min_seq = 1
    train_max_seq = 10
    n_copies = 1

    steps = 1000000

    lstm_memory_size = 50

    class Mem:
        word_size = 10
        mem_size = 20
        num_read_heads = 1


task = CopyTask(Hp.inp_vector_size, Hp.batch_size, Hp.min_seq,
                Hp.train_max_seq, Hp.n_copies)

ff = Feedforward(Hp.inp_vector_size, Hp.out_vector_size, Hp.batch_size, [128, 256, Hp.out_vector_size])
# lstm = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.out_vector_size, Hp.lstm_memory_size)
dnc = DNC(ff, Hp.Mem)

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
# optimizer = tf.train.AdamOptimizer()
dnc.run_session(task, Hp, project_path, optimizer=optimizer)
