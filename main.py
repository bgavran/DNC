import tensorflow as tf

from utils import *
from lstm import *
from tasks import *

batch_size = 5
input_size = 3
output_size = 3
memory_size = 32
seq_length = 4
total_sequences_length = 2 * seq_length + 2

controller = LSTM(batch_size, input_size, output_size, memory_size, total_sequences_length)

x = tf.placeholder(tf.float32, [None, input_size, total_sequences_length], name="X")
y = tf.placeholder(tf.float32, [None, input_size, total_sequences_length], name="Y")

x_list = tf.split(x, total_sequences_length, axis=2)
x_list = [tf.squeeze(i, 2) for i in x_list]
outputs, _ = controller(x_list)
cost = tf.reduce_mean(tf.square(y - outputs))
optimizer = tf.train.AdamOptimizer().minimize(cost)

tf.summary.scalar('cost', cost)
merged = tf.summary.merge_all()
training_steps = 1000
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join(DataPath.base, "train_summary"), sess.graph)
    tf.global_variables_initializer().run()

    for step in range(training_steps):
        batch_x, batch_y = CopyTask.generate_values(batch_size, input_size, seq_length)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 100 == 0:
            print("Training...", step)
            summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary, step)
