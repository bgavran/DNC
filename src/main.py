from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy_task import *
from task_implementations.bAbI.bAbI import *
from utils import project_path


class Hp:
    """
    Hyperparameters
    """
    batch_size = 1

    n_blocks = 6
    # inp_vector_size = n_blocks + 1
    inp_vector_size = 159
    out_vector_size = inp_vector_size

    min_seq = 5
    train_max_seq = 6
    n_copies = 2

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


initializer = init_wrapper(tf.orthogonal_initializer())

# task = CopyTask(Hp.inp_vector_size, Hp.batch_size, Hp.min_seq, Hp.train_max_seq, Hp.n_copies)
task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"), Hp.batch_size)
print("Loaded task")

# controller = Feedforward(Hp.inp_vector_size, Hp.batch_size, [128, 256])

controller = LSTM(Hp.batch_size, Hp.inp_vector_size, Hp.lstm_memory_size, Hp.n_layers)  # , Hp.out_vector_size)
dnc = DNC(controller, Hp.out_vector_size, Hp.Mem, initializer=initializer, initial_stddev=None)

print("Loaded controller")
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
# restore_path = os.path.join(project_path.log_dir, "May_13__20:03", "train", "model.chpt")
dnc.run_session(task, Hp, project_path, optimizer=optimizer)  # , restore_path=restore_path)
