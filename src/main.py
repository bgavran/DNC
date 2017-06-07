from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy_task import *
from task_implementations.bAbI.bAbI import *
from utils import project_path

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
    batch_size = 16
    steps = 1000000

    lstm_memory_size = 256
    n_layers = 1

    class Mem:
        word_size = 64
        mem_size = 256
        num_read_heads = 4


def init_wrapper(init_fn):
    def inner(shape, stddev):
        if stddev is None:
            return init_fn(shape)
        else:
            return init_fn(shape, stddev=stddev)

    return inner


init_fn = tf.random_normal
initializer = init_wrapper(init_fn)

# controller = Feedforward(task.vector_size, Hp.batch_size, [256, 512])

controller = LSTM(task.vector_size, Hp.lstm_memory_size, Hp.n_layers, task.vector_size, initializer=initializer)
# dnc = DNC(Hp.batch_size, controller, task.vector_size, Hp.Mem, initializer=initializer, initial_stddev=0.1)

print("Loaded controller")
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)

# restore_path = os.path.join(project_path.log_dir, "June_10__10:48", "train", "model.chpt")
controller.run_session(task, Hp, project_path, optimizer=optimizer)  # , restore_path=restore_path)
