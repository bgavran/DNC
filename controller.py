import tensorflow as tf
from tasks import *


class Controller:
    """
    Controller, as in, a neural network that processes the input to produce output.
    FF network is a controller, a LSTM network is a controller, but DNC is also a controller (which uses another
    controller inside it)
    """

    def run_session(self, x, y, optimizer, hp):
        outputs = self(x)
        cost = tf.reduce_mean(tf.square(y - outputs))
        optimizer = optimizer.minimize(cost)

        tf.summary.scalar('Cost', cost)

        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(hp.path, sess.graph)
            tf.global_variables_initializer().run()

            for step in range(hp.steps):
                data_batch = CopyTask.generate_values(hp.batch_size, hp.input_size, hp.min_s, hp.max_s,
                                                      hp.total_seq_length)
                sess.run(optimizer, feed_dict={x: data_batch[0], y: data_batch[1]})
                if step % 1000 == 0:
                    print("Training...", step)
                    summary = sess.run(merged, feed_dict={x: data_batch[0], y: data_batch[1]})
                    writer.add_summary(summary, step)

    def __call__(self, x):
        """
        Returns value of output after iteration for all time steps
        
        :param x: inputs for all time steps of shape [batch_size,input_size,sequence_length] ? I hope its correct?
        :return: 
        """
        raise NotImplementedError()

    def step(self, x):
        """
        Returns the output vector for just one time step!
        :param x: one vector representing the input (not all of them)
        :param state: 
        :return: 
        """
        pass
