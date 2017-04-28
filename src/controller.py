import tensorflow as tf
from tasks import CopyTask


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

    def run_session(self, task, hp, optimizer=tf.train.AdamOptimizer()):
        from time import time
        t = time()

        x = tf.placeholder(tf.float32, task.x_shape, name="X")
        y = tf.placeholder(tf.float32, task.y_shape, name="Y")

        seq_length = tf.placeholder(tf.float32, [], name="Sequence_length")

        outputs = self(x)
        print(time() - t)
        assert outputs.shape == y.shape
        # if output_activation is not None:
        #     outputs = output_activation(outputs)
        # cost = tf.reduce_mean(tf.nn.l2_loss(y - outputs))

        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=y))
        cost /= seq_length * (hp.out_vector_size - 1)

        # optimizer, clipping gradients and summarizing gradient histograms
        gradients = optimizer.compute_gradients(cost)
        from tensorflow.python.framework import ops
        for i, (gradient, variable) in enumerate(gradients):
            clipped_gradient = tf.clip_by_value(gradient, -Controller.clip_value, Controller.clip_value)
            gradients[i] = clipped_gradient, variable
            # if isinstance(gradient, ops.IndexedSlices):
            #     grad_values = gradient.values
            # else:
            #     grad_values = gradient
            # tf.summary.histogram(variable.name, variable)
            # tf.summary.histogram(variable.name + "/gradients", grad_values)
        # optimizer = optimizer.apply_gradients(gradients)
        optimizer = optimizer.minimize(cost)

        tf.summary.image("0Input", tf.expand_dims(x, axis=3), max_outputs=Controller.max_outputs)
        tf.summary.image("0Output", tf.nn.sigmoid(tf.expand_dims(outputs, axis=3)), max_outputs=Controller.max_outputs)
        tf.summary.scalar("Cost", cost)
        check = tf.add_check_numerics_ops()

        merged = tf.summary.merge_all()
        from numpy import prod, sum
        n_vars = sum([prod(var.shape) for var in tf.trainable_variables()])
        print("This model has ", n_vars, "parameters!")
        with tf.Session() as sess:
            print(time() - t)

            # from tensorflow.python import debug as tf_debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            train_writer = tf.summary.FileWriter(hp.train_path, sess.graph)
            test_writer = tf.summary.FileWriter(hp.test_path, sess.graph)
            tf.global_variables_initializer().run()
            print(time() - t)

            cost_value = 9999
            # data_batch = task.generate_data(cost_value)
            # sess.run([check], feed_dict={x: data_batch[0], y: data_batch[1]})
            for step in range(hp.steps):
                # print(step)
                # Generates new curriculum training data based on current cost
                data_batch, al = task.generate_data(cost_value)

                _, cost_value = sess.run([optimizer, cost],
                                         feed_dict={x: data_batch[0], y: data_batch[1], seq_length: al})

                if step % 100 == 0:
                    summary = sess.run(merged, feed_dict={x: data_batch[0], y: data_batch[1], seq_length: al})
                    train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        sess.run([check], feed_dict={x: data_batch[0], y: data_batch[1], seq_length: al})
                        test_data_batch, al = CopyTask.generate_values(hp.batch_size, hp.out_vector_size,
                                                                       hp.train_max_seq,
                                                                       hp.theoretical_max_seq, hp.total_output_length)
                        summary = sess.run(merged,
                                           feed_dict={x: test_data_batch[0], y: test_data_batch[1], seq_length: al})
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

    def __call__(self, x):
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
