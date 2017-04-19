from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *
from utils import *
from tasks import *


class Hp:
    batch_size = 4
    inp_vector_size = 3
    out_vector_size = inp_vector_size
    lstm_memory_size = 50
    total_output_length = 20
    min_seq = 1
    max_seq = int((total_output_length - 2) / 2)

    class Mem:
        word_size = 8
        mem_size = 16
        num_read_heads = 4

    steps = 1000000
    path = ProjectPath("log").log_path


ff = Feedforward(Hp.inp_vector_size, Hp.out_vector_size, Hp.total_output_length, Hp.batch_size, 3,
                 [128, 128, Hp.out_vector_size])
lstm = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.out_vector_size, Hp.lstm_memory_size,
            Hp.total_output_length)

dnc = DNC(lstm, Hp.Mem)

task = CopyTask(Hp.inp_vector_size, Hp.out_vector_size, Hp.total_output_length, Hp.batch_size, Hp.min_seq, Hp.max_seq)
dnc.run_session(task, Hp)
