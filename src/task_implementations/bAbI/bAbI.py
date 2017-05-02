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
        self.x_test = np.array(flatten([v for k, v in all_input_stories.items() if k in self.test_list]))
        self.y_test = np.array(flatten([v for k, v in all_output_stories.items() if k in self.test_list]))
        assert len(self.x_train) == len(self.y_train)

        self.x_shape = [self.batch_size, self.vector_size, None]
        self.y_shape = [self.batch_size, self.vector_size, None]
        self.mask = [self.batch_size, None]

    def display_output(self, prediction, data_batch, mask):
        text = " ".join([self.ind_to_word[np.argmax(i)] for i in data_batch[0][0].T])

        correct_ind = np.argmax([data_batch[1][0, :, i] for i in np.nonzero(data_batch[1][0] * mask)[1]], axis=1)
        correct_words = " ".join([self.ind_to_word[i] for i in correct_ind])

        outputs_list = np.unique(np.nonzero(prediction * mask)[2])
        output_ind = np.argmax([bAbITask.softmax(i) for i in prediction[0, :, outputs_list]], axis=1)
        output_words = " ".join([self.ind_to_word[i] for i in output_ind])

        print(text)
        print("Output:", output_words)
        print("Correct:", correct_words)
        print("-------------------------------------------------------------------\n")

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def one_hot(x, n_elements):
        return np.transpose([np.eye(n_elements)[indices] for indices in x], [0, 2, 1])

    def generate_data(self, cost=None, train=True):
        if train:
            ind = np.random.randint(0, len(self.x_train), self.batch_size)
            x = bAbITask.one_hot(self.x_train[ind], len(self.word_to_ind))
            y = bAbITask.one_hot(self.y_train[ind], len(self.word_to_ind))
        else:
            ind = np.random.randint(0, len(self.x_test), self.batch_size)
            x = bAbITask.one_hot(self.x_test[ind], len(self.word_to_ind))
            y = bAbITask.one_hot(self.y_test[ind], len(self.word_to_ind))

        return [x, y], x.shape[2], x[:, 0, :]

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
