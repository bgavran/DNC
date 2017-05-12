from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy import *
from task_implementations.bAbI.bAbI import *
from utils import project_path


class Hp:
    """
    Hyperparameters
    """
    batch_size = 1

    n_blocks = 8
    # inp_vector_size = n_blocks + 1
    inp_vector_size = 159
    out_vector_size = inp_vector_size

    min_seq = 5
    train_max_seq = 8
    n_copies = 1

    steps = 1000000

    lstm_memory_size = 256
    n_layers = 1

    class Mem:
        word_size = 64
        mem_size = 256
        num_read_heads = 4


# task = CopyTask(Hp.inp_vector_size, Hp.batch_size, Hp.min_seq, Hp.train_max_seq, Hp.n_copies)
task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"), Hp.batch_size)
print("Loaded task")

# controller = Feedforward(Hp.inp_vector_size, Hp.batch_size, [128, 256])
controller = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.lstm_memory_size, Hp.n_layers)
dnc = DNC(controller, Hp.out_vector_size, Hp.Mem, initial_stddev=0.1)

print("Loaded controller")
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
# restore_path = os.path.join(project_path.log_dir, "May_09__19:05", "train", "model.chpt")
dnc.run_session(task, Hp, project_path, optimizer=optimizer)  # , restore_path=restore_path)
