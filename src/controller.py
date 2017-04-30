import tensorflow as tf
from task_implementations.copy import *


class Controller:
    """
    Controller: a neural network that optionally uses memory or other controllers to produce output.
    FF network is a controller, a LSTM network is a controller, but DNC is also a controller (which uses another
    controller inside it)
    This implementation of controller (FF, LSTM) is trying to establish a clear interface between sizes of inputs and 
    outputs for a controller. This, in theory, allows for easy nesting of controllers.
    
    What this implies is a lower parameter count for the output and interface weight matrices, since they operate only 
    on the outputs of the controller and not outputs of every layer of the controller. This is where this implementation
    differs from DeepMind's DNC implementation.
    
    
    """
    max_outputs = 2
    clip_value = 10

    def run_session(self, task, hp, project_path, optimizer=tf.train.AdamOptimizer()):
        x = tf.placeholder(tf.float32, task.x_shape, name="X")
        y = tf.placeholder(tf.float32, task.y_shape, name="Y")

        sequence_length = tf.placeholder(tf.int32, [], name="Sequence_length")

        outputs, summaries = self(x, sequence_length)
        assert outputs.shape[:2] == y.shape[:2]

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=y))

        # optimizer, clipping gradients and summarizing gradient histograms
        gradients = optimizer.compute_gradients(cost)
        from tensorflow.python.framework import ops
        for i, (gradient, variable) in enumerate(gradients):
            clipped_gradient = tf.clip_by_value(gradient, -Controller.clip_value, Controller.clip_value)
            gradients[i] = clipped_gradient, variable
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + "/gradients", grad_values)
        optimizer = optimizer.apply_gradients(gradients)
        # optimizer = optimizer.minimize(cost)

        tf.summary.image("0Input", tf.expand_dims(x, axis=3), max_outputs=Controller.max_outputs)
        tf.summary.image("0Output", tf.nn.sigmoid(tf.expand_dims(outputs, axis=3)), max_outputs=Controller.max_outputs)
        tf.summary.scalar("Cost", cost)

        self.notify(summaries)

        merged = tf.summary.merge_all()

        from numpy import prod, sum
        n_vars = sum([prod(var.shape) for var in tf.trainable_variables()])
        print("This model has", n_vars, "parameters!")

        with tf.Session() as sess:

            train_writer = tf.summary.FileWriter(project_path.train_path, sess.graph)
            test_writer = tf.summary.FileWriter(project_path.test_path, sess.graph)
            tf.global_variables_initializer().run()

            from time import time
            t = time()

            cost_value = 9999
            for step in range(hp.steps):
                # Generates new curriculum training data based on current cost
                data_batch, seqlen = task.generate_data(cost_value)

                _, cost_value = sess.run([optimizer, cost],
                                         feed_dict={x: data_batch[0], y: data_batch[1], sequence_length: seqlen})

                if step % 200 == 0:
                    summary = sess.run(merged, feed_dict={x: data_batch[0], y: data_batch[1], sequence_length: seqlen})
                    train_writer.add_summary(summary, step)

                    if step % 1000 == 0 and cost_value < 0.05:
                        test_data_batch, seqlen = CopyTask.generate_n_copies(hp.batch_size, hp.out_vector_size,
                                                                             hp.train_max_seq,
                                                                             20, 1)
                        summary = sess.run(merged, feed_dict={x: test_data_batch[0], y: test_data_batch[1],
                                                              sequence_length: seqlen})
                        test_writer.add_summary(summary, step)

                    print("Summary generated. Step", step,
                          " Train cost == %.9f Time == %.2fs" % (cost_value, time() - t))
                    t = time()

    def notify(self, states):
        """
        Method which implements all the tf summary operations. If the instance uses another controller inside it, 
        like DNC, then it's responsible for calling the controller's notify method
        
        :param states: list of states for all time steps. Each state is specific to the actual controller
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
        return all_outputs
        
        
        :param x: inputs for all time steps of shape [batch_size, input_size, sequence_length]
        :return: list of outputs for every time step of the network
        """
        raise NotImplementedError()

    def step(self, x, step):
        """
        Returns the output vector for just one time step
        
        :param x: one vector representing input for one time step
        :param step: current time step
        :return: output of the controller and its current state
        """
        raise NotImplementedError()
