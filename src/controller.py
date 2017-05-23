import tensorflow as tf
from task_implementations.copy_task import *
from tensorflow.python.client import timeline


class Controller:
    """
    Controller: a neural network that optionally uses memory or other controllers to produce output.
    FF network is a controller, a LSTM network is a controller, but DNC is also a controller (which uses another
    controller inside it)
    The only problem is that this is theory, and the way tf.while_loop is implemented prevents easy nesting of 
    controllers :/
    
    """
    max_outputs = 1
    clip_value = 10

    def run_session(self, task, hp, project_path, restore_path=None, optimizer=tf.train.AdamOptimizer()):
        x = tf.placeholder(tf.float32, task.x_shape, name="X")
        y = tf.placeholder(tf.float32, task.y_shape, name="Y")

        sequence_length = tf.placeholder(tf.int32, [], name="Sequence_length")
        mask = tf.placeholder(tf.float32, task.mask, name="Output_mask")

        outputs, summaries = self(x, sequence_length)
        assert outputs.shape[:2] == y.shape[:2]

        # summary_outputs = tf.nn.sigmoid(outputs)
        summary_outputs = tf.nn.softmax(outputs, dim=1)
        # summary_outputs_masked = tf.einsum("bht,bt->bht", summary_outputs, mask)

        tf.summary.image("0_Input", tf.expand_dims(x, axis=3), max_outputs=Controller.max_outputs)
        tf.summary.image("0_Network_output", tf.expand_dims(summary_outputs, axis=3),
                         max_outputs=Controller.max_outputs)
        tf.summary.image("0_Y", tf.expand_dims(y * mask, axis=3), max_outputs=Controller.max_outputs)
        # tf.summary.image("0OutputMasked", tf.expand_dims(summary_outputs_masked, axis=3),
        #                  max_outputs=Controller.max_outputs)

        cost = task.cost(outputs, y, mask)
        tf.summary.scalar("Cost", cost)

        # optimizer, clipping gradients and summarizing gradient histograms
        gradients = optimizer.compute_gradients(cost)
        from tensorflow.python.framework import ops
        for i, (gradient, variable) in enumerate(gradients):
            clipped_gradient = tf.clip_by_value(gradient, -Controller.clip_value, Controller.clip_value)
            gradients[i] = clipped_gradient, variable
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + "/gradients",
                                 gradient.values if isinstance(gradient, ops.IndexedSlices) else gradient)
        optimizer = optimizer.apply_gradients(gradients)
        # optimizer = optimizer.minimize(cost)

        self.notify(summaries)

        merged = tf.summary.merge_all()

        from numpy import prod, sum
        n_vars = sum([prod(var.shape) for var in tf.trainable_variables()])
        print("This model has", n_vars, "parameters!")

        saver = tf.train.Saver()

        with tf.Session() as sess:
            if restore_path is not None:
                saver.restore(sess, restore_path)
                print("Restored model", restore_path, "!!!!!!!!!!!!!")
            else:
                tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(project_path.train_path, sess.graph)
            test_writer = tf.summary.FileWriter(project_path.test_path, sess.graph)

            # from tensorflow.python import debug as tf_debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            from time import time
            t = time()

            cost_value = 9999
            print("Starting...")
            for step in range(hp.steps):
                # Generates new curriculum training data based on current cost
                data_batch, seqlen, m = task.generate_data(cost_value)
                _, cost_value = sess.run([optimizer, cost],
                                         feed_dict={x: data_batch[0], y: data_batch[1], sequence_length: seqlen,
                                                    mask: m})

                if step % 100 == 0:
                    summary = sess.run(merged, feed_dict={x: data_batch[0], y: data_batch[1],
                                                          sequence_length: seqlen,
                                                          mask: m})
                    train_writer.add_summary(summary, step)
                    test_data_batch, seqlen, m = task.generate_data(cost=cost, train=False)
                    summary, pred, cost_value = sess.run([merged, outputs, cost],
                                                         feed_dict={x: test_data_batch[0], y: test_data_batch[1],
                                                                    sequence_length: seqlen, mask: m})
                    test_writer.add_summary(summary, step)
                    task.display_output(pred, test_data_batch, m)
                    # input()

                    print("Summary generated. Step", step,
                          " Test cost == %.9f Time == %.2fs" % (cost_value, time() - t))
                    t = time()

                    if step % 1000 == 0:  # and step > 0:
                        task.test(sess, outputs, [x, y, sequence_length, mask])
                        saver.save(sess, project_path.model_path)
                        print("Model saved!")

    def notify(self, summaries):
        """
        Method which implements all the tf summary operations. If the instance uses another controller inside it, 
        like DNC, then it's responsible for calling the controller's notify method
        
        :param summaries: 
        :return: 
        """
        raise NotImplementedError()

    def __call__(self, x, sequence_length):
        """
        Returns all outputs after iteration for all time steps
        
        High level overview of __call__:
        for step in steps:
            output, state = self.step(data[step])
        self.notify(all_states)
        
        
        :param x: inputs for all time steps of shape [batch_size, input_size, sequence_length]
        :return: list of two tensors [all_outputs, all_summaries]
        """
        raise NotImplementedError()

    def step(self, x, state, step):
        """
        Returns the output vector for just one time step.
        But I'm not sure anymore how much does all of this work since because of the way tf.while_loop is implemented...
        
        :param x: one vector representing input for one time step
        :param state: state of the controller
        :param step: current time step
        :return: output of the controller and its current state
        """
        raise NotImplementedError()
