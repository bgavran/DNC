import tensorflow as tf

from utils import *
from lstm import *
from tasks import *
from tensorflow.python.ops import array_ops

batch_size = 5
input_size = 2
output_size = input_size
memory_size = 8
seq_length = 4
total_sequences_length = 2 * seq_length + 2

controller = LSTM(batch_size, input_size, output_size, memory_size, total_sequences_length)

x = tf.placeholder(tf.float32, [None, input_size, total_sequences_length], name="X")
y = tf.placeholder(tf.float32, [None, input_size, total_sequences_length], name="Y")

x_list = tf.split(x, total_sequences_length, axis=2)
x_list = [tf.squeeze(i, 2) for i in x_list]

outputs, _ = controller(x_list)
tf.summary.image("Input", tf.expand_dims(x, axis=3), max_outputs=batch_size)
tf.summary.image("Output", tf.expand_dims(outputs, axis=3), max_outputs=batch_size)
weights_expanded = tf.expand_dims(tf.expand_dims(controller.weights, axis=0), axis=3)
# tf.summary.image("LSTM weights", tf.get_variable("output_weights"), max_outputs=batch_size)
tf.summary.image("LSTM output weights", tf.transpose(weights_expanded), max_outputs=batch_size)

inner_weights = [v for v in tf.global_variables() if v.name.startswith("rnn/basic_lstm_cell/weights")][0]
inner_weights_expanded = tf.expand_dims(tf.expand_dims(inner_weights, axis=0), axis=3)
# i, j, f, o = array_ops.split(value=inner_weights, num_or_size_splits=4, axis=1)
# input_weights = tf.expand_dims(tf.expand_dims(i, axis=0), axis=3)
tf.summary.image("LSTM inner weights", tf.transpose(inner_weights_expanded), max_outputs=batch_size)

cost = tf.reduce_mean(tf.square(y - outputs))
optimizer = tf.train.AdamOptimizer().minimize(cost)

tf.summary.scalar('Cost', cost)
merged = tf.summary.merge_all()
training_steps = 100000
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join(DataPath.base, "train_summary"), sess.graph)
    tf.global_variables_initializer().run()

    for step in range(training_steps):
        batch_x, batch_y = CopyTask.generate_values(batch_size, input_size, seq_length)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 1000 == 0:
            print("Training...", step)
            summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary, step)
