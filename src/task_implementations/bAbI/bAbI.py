import os
import re
import pickle
import numpy as np
import tensorflow as tf
from utils import project_path
from tasks import Task


class bAbITask(Task):
    base = os.path.join(project_path.base, "src", "task_implementations", "bAbI")
    output_symbol = "-"
    newstory_delimiter = " NEWSTORY "
    processed_append = "-processed.p"

    def __init__(self, tasks_dir, batch_size):
        # works only with all examples in the batch being the same size! so that's why the batch is 1
        # also a bunch of other code is configured that it only works with batch_size == 1
        assert batch_size == 1
        self.tasks_dir = os.path.join(bAbITask.base, tasks_dir)
        self.processed_dir = self.tasks_dir + bAbITask.processed_append
        self.batch_size = batch_size
        self.files_path = []

        for f in os.listdir(self.tasks_dir):
            f_path = os.path.join(self.tasks_dir, f)
            if os.path.isfile(f_path):
                self.files_path.append(f_path)

        if not os.path.isfile(self.processed_dir):
            pickle.dump(self.preprocess_files(), open(self.processed_dir, "wb"))
            print("Pickled!", self.processed_dir)
        self.word_to_ind, all_input_stories, all_output_stories = pickle.load(open(self.processed_dir, "rb"))
        self.ind_to_word = {ind: word for word, ind in self.word_to_ind.items()}

        self.train_list = [k for k, v in all_input_stories.items() if k[-9:] == "train.txt"]
        self.test_list = [k for k, v in all_input_stories.items() if k[-8:] == "test.txt"]

        def flatten(forest):
            return [leaf for tree in forest for leaf in tree]

        self.vector_size = len(self.word_to_ind)

        self.x_train = np.array(flatten([v for k, v in all_input_stories.items() if k in self.train_list]))
        self.y_train = np.array(flatten([v for k, v in all_output_stories.items() if k in self.train_list]))

        self.x_test_stories = {k: v for k, v in all_input_stories.items() if k in self.test_list}
        self.y_test_stories = {k: v for k, v in all_output_stories.items() if k in self.test_list}
        self.x_test = np.array(flatten(list(self.x_test_stories.values())))
        self.y_test = np.array(flatten(list(self.y_test_stories.values())))

        # self.x_test_onehot = [[np.eye(self.vector_size)[ind] for ind in story] for story in flatten(self.x_test)]
        # self.y_test_onehot = [[np.eye(self.vector_size)[ind] for ind in story] for story in flatten(self.y_test)]

        assert len(self.x_train) == len(self.y_train)

        self.x_shape = [self.batch_size, self.vector_size, None]
        self.y_shape = [self.batch_size, self.vector_size, None]
        self.mask = [self.batch_size, None]

    def display_output(self, prediction, data_batch, mask):
        text = self.indices_to_words([np.argmax(i) for i in data_batch[0][0].T])

        correct_indices = bAbITask.tensor_to_indices(data_batch[1], mask)
        out_indices = bAbITask.tensor_to_indices(prediction, mask)

        correct_words = self.indices_to_words(correct_indices)
        out_words = self.indices_to_words(out_indices)

        print(text)
        print("Output:", out_words)
        print("Correct:", correct_words)
        print("-------------------------------------------------------------------\n")

    def indices_to_words(self, indices):
        """
        
        :param indices: list of indices to be transformed into a string
        :return: 
        """
        return " ".join([self.ind_to_word[ind] for ind in indices])

    @staticmethod
    def tensor_to_indices(data_tensor, mask):
        """
        
        :param data_tensor: input/output tensor of shape [batch_size, vector_size, n_steps], in which the first 
        dimension should actually be 1 since the code doesn't work otherwise
        :param mask: tensor the same shape of data_tensor, used to mask unimportant output steps
        :return: list of one_hot indices for the max values of data_tensor, after masking
        """
        locations = np.unique(np.nonzero(data_tensor * mask)[2])
        indices = np.argmax(data_tensor[0, :, locations], axis=1)
        return indices

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def to_onehot(x, n_elements):
        return np.transpose([np.eye(n_elements)[indices] for indices in x], [0, 2, 1])

    def generate_data(self, cost=None, train=True):
        if train:
            ind = np.random.randint(0, len(self.x_train), self.batch_size)
            x = bAbITask.to_onehot(self.x_train[ind], len(self.word_to_ind))
            y = bAbITask.to_onehot(self.y_train[ind], len(self.word_to_ind))
        else:
            ind = np.random.randint(0, len(self.x_test), self.batch_size)
            x = bAbITask.to_onehot(self.x_test[ind], len(self.word_to_ind))
            y = bAbITask.to_onehot(self.y_test[ind], len(self.word_to_ind))

        return [x, y], x.shape[2], x[:, 0, :]

    def test(self, sess, outputs_tf, pl):
        from time import time
        t = time()
        print("Testing...")
        num_passed_tasks = 0
        num_tasks = len(self.x_test_stories)
        task_errors = []
        for i, (inp, output) in enumerate(zip(self.x_test_stories.items(), self.y_test_stories.items())):
            total_correct = 0
            for inp_story, out_story in zip(inp[1], output[1]):
                x = bAbITask.to_onehot(np.expand_dims(inp_story, axis=0), self.vector_size)
                y = bAbITask.to_onehot(np.expand_dims(out_story, axis=0), self.vector_size)
                seqlen = x.shape[2]
                m = x[:, 0, :]
                outputs = sess.run(outputs_tf, feed_dict={pl[0]: x, pl[1]: y, pl[2]: seqlen, pl[3]: m})
                outputs_list = bAbITask.tensor_to_indices(outputs, m)
                correct_list = bAbITask.tensor_to_indices(y, m)
                total_correct += np.array_equal(outputs_list, correct_list)
            task_name = inp[0].split("/")[-1]
            num_stories = len(inp[1])
            task_error = 1 - total_correct / num_stories
            print(i, task_name, " Total_correct:", total_correct, "    task error:", task_error)
            num_passed_tasks += task_error > 0.05
            task_errors.append(task_error)
        mean_error = np.mean(task_errors)
        print("TOTAL PASSED TASKS:", num_passed_tasks, " MEAN ERROR", mean_error)
        print("Time == ", time() - t)

    def cost(self, x, y, mask=None):
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y, dim=1)

        return tf.reduce_mean(softmax_cross_entropy * mask)

    def preprocess_files(self):
        word_to_ind, all_input_stories, all_output_stories = {bAbITask.output_symbol: 0}, dict(), dict()
        for file_path in self.files_path:
            print(file_path)
            file = open(file_path).read().lower()
            file = re.sub("\n1 ", bAbITask.newstory_delimiter, file)  # adding a delimeter between two stories
            file = re.sub("\d+|\n|\t", " ", file)  # removing all numbers, newlines and tabs
            file = re.sub("([?.])", r" \1", file)  # adding a space before all punctuations
            stories = file.split(bAbITask.newstory_delimiter)

            input_stories = []
            output_stories = []
            for i, story in enumerate(stories):
                input_tokens = story.split()
                output_tokens = story.split()

                for i, token in enumerate(input_tokens):
                    if token == "?":
                        output_tokens[i + 1] = output_tokens[i + 1].split(",")
                        input_tokens[i + 1] = [bAbITask.output_symbol for _ in range(len(output_tokens[i + 1]))]

                input_tokens = bAbITask.flatten_if_list(input_tokens)
                output_tokens = bAbITask.flatten_if_list(output_tokens)

                for token in output_tokens:
                    if token not in word_to_ind:
                        word_to_ind[token] = len(word_to_ind)

                input_stories.append([word_to_ind[elem] for elem in input_tokens])
                output_stories.append([word_to_ind[elem] for elem in output_tokens])
            all_input_stories[file_path] = input_stories
            all_output_stories[file_path] = output_stories
        return word_to_ind, all_input_stories, all_output_stories

    @staticmethod
    def flatten_if_list(l):
        newl = []
        for elem in l:
            if isinstance(elem, list):
                newl.extend(elem)
            else:
                newl.append(elem)
        return newl


if __name__ == '__main__':
    tasks_dir = os.path.join("tasks_1-20_v1-2", "en-10k")
    batch_size = 5
    self = bAbITask(tasks_dir, batch_size)
    self.generate_data()
