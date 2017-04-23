from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *
from utils import *
from tasks import *


class Hp:
    """
    Hyperparameters
    """
    batch_size = 10
    inp_vector_size = 5
    out_vector_size = inp_vector_size
    lstm_memory_size = 50
    total_output_length = 20
    min_seq = 1
    theoretical_max_seq = int((total_output_length - 2) / 2)

    steps = 1000000

    class Mem:
        word_size = 8
        mem_size = 16
        num_read_heads = 1

    project_path = ProjectPath("log")
    train_path = project_path.train_path
    test_path = project_path.test_path


task = CopyTask(Hp.inp_vector_size, Hp.out_vector_size, Hp.total_output_length, Hp.batch_size, Hp.min_seq,
                Hp.theoretical_max_seq)

ff = Feedforward(Hp.inp_vector_size, Hp.out_vector_size, Hp.total_output_length, Hp.batch_size, 3,
                 [128, 128, Hp.out_vector_size])
# lstm = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.out_vector_size, Hp.lstm_memory_size,
#             Hp.total_output_length)
dnc = DNC(ff, Hp.Mem, output_activation=tf.nn.sigmoid)

dnc.run_session(task, Hp)
