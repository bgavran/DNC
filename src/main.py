from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy_task import *
from task_implementations.bAbI.bAbI import *
from utils import project_path, init_wrapper

n_blocks = 6
vector_size = n_blocks + 1
min_seq = 5
train_max_seq = 6
n_copies = 1
out_vector_size = vector_size

# task = CopyTask(vector_size, min_seq, train_max_seq, n_copies)
task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"))

print("Loaded task")


class Hp:
    """
    Hyperparameters
    """
    batch_size = 4
    steps = 1000000

    lstm_memory_size = 512
    n_layers = 1

    class Mem:
        word_size = 64
        mem_size = 256
        num_read_heads = 4


initializer = init_wrapper(tf.random_normal)

# controller = Feedforward(task.vector_size, Hp.batch_size, [256, 512])

controller = LSTM(task.vector_size, Hp.lstm_memory_size, Hp.n_layers, initializer=initializer,
                  out_vector_size=task.vector_size)
# dnc = DNC(Hp.batch_size, controller, task.vector_size, Hp.Mem, initializer=initializer, initial_stddev=0.1)

print("Loaded controller")

# restore_path = os.path.join(project_path.log_dir, "June_09__19:32", "train", "model.chpt")
controller.run_session(task, Hp, project_path)  # , restore_path=restore_path)
