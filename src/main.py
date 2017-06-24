from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy_task import *
from task_implementations.bAbI.bAbI import *
from utils import project_path, init_wrapper

weight_initializer = init_wrapper(tf.random_normal)

n_blocks = 6
vector_size = n_blocks + 1
min_seq = 5
train_max_seq = 6
n_copies = 1
out_vector_size = vector_size

task = CopyTask(vector_size, min_seq, train_max_seq, n_copies)
# task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"))

print("Loaded task")


class Hp:
    """
    Hyperparameters
    """
    batch_size = 4
    steps = 1000000

    lstm_memory_size = 128
    n_layers = 1

    stddev = 0.1

    class Mem:
        word_size = 8
        mem_size = 10
        num_read_heads = 1


controller = Feedforward(task.vector_size, Hp.batch_size, [256, 512])

# controller = LSTM(task.vector_size, Hp.lstm_memory_size, Hp.n_layers, initializer=weight_initializer,
#                   initial_stddev=Hp.stddev)

# out_vector_size=task.vector_size is a needed argument to LSTM if you want to test just the LSTM outside of DNC

dnc = DNC(controller, Hp.batch_size, task.vector_size, Hp.Mem, initializer=weight_initializer, initial_stddev=Hp.stddev)

print("Loaded controller")

# restore_path = os.path.join(project_path.log_dir, "June_09__19:32", "train", "model.chpt")
dnc.run_session(task, Hp, project_path)  # , restore_path=restore_path)
