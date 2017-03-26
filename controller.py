import tensorflow as tf
from utils import *
from tasks import *


class Controller:
    def __init__(self):
        raise NotImplementedError()

    def run_session(self, x, y, optimizer, hp):
        optimizer = optimizer.minimize(self(x, y))

        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(os.path.join(DataPath.base, hp.logdir), sess.graph)
            tf.global_variables_initializer().run()

            for step in range(hp.steps):
                data_batch = CopyTask.generate_values(hp.batch_size, hp.input_size, hp.min_s, hp.max_s,
                                                      hp.total_seq_length)
                sess.run(optimizer, feed_dict={x: data_batch[0], y: data_batch[1]})
                if step % 1000 == 0:
                    print("Training...", step)
                    summary = sess.run(merged, feed_dict={x: data_batch[0], y: data_batch[1]})
                    writer.add_summary(summary, step)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
