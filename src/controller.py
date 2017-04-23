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

    def run_session(self, task, hp, optimizer=tf.train.AdamOptimizer()):
        x = tf.placeholder(tf.float32, task.x_shape, name="X")
        y = tf.placeholder(tf.float32, task.y_shape, name="Y")

        outputs = self(x)
        assert outputs.shape == y.shape
        cost = tf.nn.l2_loss(y - outputs)
        # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=y))
        optimizer = optimizer.minimize(cost)

        tf.summary.image("Input", tf.expand_dims(x, axis=3), max_outputs=Controller.max_outputs)
        tf.summary.image("Output", tf.expand_dims(outputs, axis=3), max_outputs=Controller.max_outputs)
        tf.summary.scalar("Cost", cost)
        check = tf.add_check_numerics_ops()

        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(hp.train_path, sess.graph)
            test_writer = tf.summary.FileWriter(hp.test_path, sess.graph)
            tf.global_variables_initializer().run()

            from time import time
            t = time()
            cost_value = 9999
            data_batch = task.generate_data(cost_value)
            sess.run([check], feed_dict={x: data_batch[0], y: data_batch[1]})
            for step in range(hp.steps):
                # Generates new curriculum training data based on current cost
                data_batch = task.generate_data(cost_value)

                sess.run([check], feed_dict={x: data_batch[0], y: data_batch[1]})
                _, cost_value = sess.run([optimizer, cost], feed_dict={x: data_batch[0], y: data_batch[1]})

                if step % 200 == 0:
                    summary = sess.run(merged, feed_dict={x: data_batch[0], y: data_batch[1]})
                    train_writer.add_summary(summary, step)

                    test_data_batch = CopyTask.generate_values(hp.batch_size, hp.out_vector_size, hp.min_seq,
                                                               hp.theoretical_max_seq, hp.total_output_length)
                    summary = sess.run(merged, feed_dict={x: test_data_batch[0], y: test_data_batch[1]})
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
