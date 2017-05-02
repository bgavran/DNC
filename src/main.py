from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy import *
from task_implementations.bAbI.bAbI import *
from utils import project_path

babitask = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"), 1)


class Hp:
    """
    Hyperparameters
    """
    batch_size = 1

    n_blocks = 10
    inp_vector_size = babitask.vector_size
    # inp_vector_size = n_blocks + 1
    out_vector_size = inp_vector_size

    min_seq = 1
    train_max_seq = 10
    n_copies = 1

    steps = 1000000

    lstm_memory_size = 256
    n_layers = 1

    class Mem:
        word_size = 32
        mem_size = 256
        num_read_heads = 4


# copytask = CopyTask(Hp.inp_vector_size, Hp.batch_size, Hp.min_seq,
#                     Hp.train_max_seq, Hp.n_copies)

# ff = Feedforward(Hp.inp_vector_size, Hp.batch_size, [128, 256])
lstm = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.lstm_memory_size, Hp.n_layers)
dnc = DNC(lstm, Hp.out_vector_size, Hp.Mem)

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
dnc.run_session(babitask, Hp, project_path, optimizer=optimizer)
