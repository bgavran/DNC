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
    pad_symbol = "*"
    newstory_delimiter = " NEWSTORY "
    processed_append = "-processed.p"

    def __init__(self, tasks_dir):
        """
        Init tries to read from pickle file, if it doesn't exist it creates it.
        Creates separate dictionaries of train and test sets, data and labels

        :param tasks_dir: relative directory of bAbI tasks
        """
        self.tasks_dir = os.path.join(bAbITask.base, tasks_dir)
        self.processed_dir = self.tasks_dir + bAbITask.processed_append
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

        from natsort import natsorted
        self.train_list = natsorted([k for k, v in all_input_stories.items() if k[-9:] == "train.txt"])
        self.test_list = natsorted([k for k, v in all_input_stories.items() if k[-8:] == "test.txt"])

        self.vector_size = len(self.word_to_ind)
        self.n_tasks = 20

        self.x_train_stories = {k: v for k, v in all_input_stories.items() if k in self.train_list}
        self.y_train_stories = {k: v for k, v in all_output_stories.items() if k in self.train_list}

        self.x_test_stories = {k: v for k, v in all_input_stories.items() if k in self.test_list}
        self.y_test_stories = {k: v for k, v in all_output_stories.items() if k in self.test_list}

        assert len(self.x_train_stories.keys()) == len(self.y_train_stories.keys())

        # shape [batch_size, length, vector_size]
        self.x_shape = [None, None, self.vector_size]
        self.y_shape = [None, None, self.vector_size]
        self.mask = [None, None, 1]

        self.mean_test_errors = []

    def display_output(self, prediction, data_batch, mask):
        """
        For a batch of stories and the corresponding network output, it prints the first story and its output.
        It prints out the story words (with asterisks for padding if there was padding), the network output converted
        to words and the correct output converted to words.

        :param prediction:
        :param data_batch:
        :param mask:
        :return:
        """
        # taking just the first story in the batch
        prediction = prediction[:1, :, :]
        mask = mask[:1, :, :]
        data_batch = data_batch[:, :1, :, :]

        text = self.indices_to_words([np.argmax(i) for i in data_batch[0][0]])

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
        Converts the one hot tensor to indices
        
        :param data_tensor: input/output tensor of shape [batch_size, n_steps, vector_size], in which the first
        dimension should actually be 1 since the code doesn't work otherwise
        :param mask: tensor the same shape of data_tensor, used to mask unimportant output steps
        :return: list of one_hot indices for the max values of data_tensor, after masking
        """
        assert len(data_tensor.shape) == 3 and data_tensor.shape[0] == mask.shape[0]
        locations = np.unique(np.nonzero(data_tensor * mask)[1])
        indices = np.argmax(data_tensor[0, locations, :], axis=1)
        return indices

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def to_onehot(x, depth):
        """
        
        :param x: 
        :param depth:
        :return: 
        """
        return np.array([np.eye(depth)[int(indices)] for indices in x])

    def generate_data(self, batch_size=16, train=True, cost=None):
        """
        Main method for generating train/test data.
        Generates batch_size random indices, which correspond to some stories (1-20) and then takes a random story
        for each of the generated task indices.
        It converts the stories to one hot and pads them.

        :param cost: used in copy task for curriculum learning, but not here
        :param train: sampling from train or test data
        :param batch_size:
        :return: data_batch, lenghts of the each sampled story and corresponding masks
        """
        task_indices = np.random.randint(0, self.n_tasks, batch_size)
        # task_ind = 0
        if train:
            task_names = [self.train_list[ind] for ind in task_indices]
            x_task_names_stories = [self.x_train_stories[task_name] for task_name in task_names]
            y_task_names_stories = [self.y_train_stories[task_name] for task_name in task_names]
        else:
            task_names = [self.test_list[ind] for ind in task_indices]
            x_task_names_stories = [self.x_test_stories[task_name] for task_name in task_names]
            y_task_names_stories = [self.y_test_stories[task_name] for task_name in task_names]
        x = []
        y = []
        for x_task_stories, y_task_stories in zip(x_task_names_stories, y_task_names_stories):
            story_ind = np.random.randint(0, len(x_task_stories))
            x.append(bAbITask.to_onehot(x_task_stories[story_ind], self.vector_size))
            y.append(bAbITask.to_onehot(y_task_stories[story_ind], self.vector_size))

        x, y, lengths = self.pad_stories(x, y)
        return np.array([x, y]), lengths, x[:, :, :1]

    def pad_stories(self, x, y):
        """
        Pads the stories in a batch to the size of the longest one

        :param x:
        :param y:
        :return:
        """
        lengths = [len(story) for story in x]
        max_length = np.max(lengths)

        # padding the stories to the max length
        for i, story in enumerate(x):
            padding = bAbITask.to_onehot(np.ones(max_length - len(story)), self.vector_size)
            if len(padding) > 0:
                x[i] = np.vstack((x[i], padding))
                y[i] = np.vstack((y[i], padding))
        x = np.array(x)
        y = np.array(y)
        return x, y, lengths

    def test(self, sess, outputs_tf, fd, batch_size):
        """
        Evaluates the performance of the network on the whole test set.

        :param sess: tensorflow session to be used
        :param outputs_tf: object that represents callable tensorflow outputs
        :param fd: list of feed_dict variables [x, y, lenghts, mask]
        :param batch_size:
        :return:
        """
        from time import time
        t = time()
        print("Testing...")
        num_passed_tasks = 0
        num_tasks = len(self.x_test_stories)
        task_errors = []
        for ind, (inp, output) in enumerate(zip(self.x_test_stories.items(), self.y_test_stories.items())):
            # if inp[0] != self.test_list[0]:
            #     continue
            correct_questions = 0
            total_questions = 0
            num_stories = len(inp[1])
            # num_stories = 30

            """
            Processing each story with other batch - 1 stories. Stupid hack because I can't have variable batch 
            size because I'm using tf.unstack and for some reason it doesn't work.
            """
            x = [bAbITask.to_onehot(i, self.vector_size) for i in inp[1][:batch_size]]
            y = [bAbITask.to_onehot(i, self.vector_size) for i in output[1][:batch_size]]
            for index in range(num_stories):
                x = list(x)
                y = list(y)
                x[0] = bAbITask.to_onehot(inp[1][index], self.vector_size)
                y[0] = bAbITask.to_onehot(output[1][index], self.vector_size)
                x, y, lengths = self.pad_stories(x, y)
                m = x[:, :, :1]
                outputs = sess.run(outputs_tf, feed_dict={fd[0]: x, fd[1]: y, fd[2]: lengths, fd[3]: m})
                """
                Each story (one whole sequence) has several questions and each of those questions might have several
                words as an answer.
                So a list of answer indices like [23, 56, 52, 45, 68] might only correspond to three questions.
                In the code below we loop through the answers, check if the question indices in the story are adjacent
                and count the adjacent words as one question only.
                Sanity check is that total number of questions turns out to be 1000 in all tasks :)
                """

                outputs_list = bAbITask.tensor_to_indices(outputs[:1], m[:1])
                correct_list = bAbITask.tensor_to_indices(y[:1], m[:1])
                answers = y[0] * m[0, :, :1]
                locations = np.argwhere(answers > 0)[:, 0]
                i = 0
                while i < len(locations):
                    all_words_correct = True
                    j = 0
                    while i + j < len(locations) and locations[i] + j == locations[i + j]:
                        if outputs_list[i + j] != correct_list[i + j]:
                            all_words_correct = False
                        j += 1
                    total_questions += 1
                    correct_questions += all_words_correct
                    i += j

            task_name = inp[0].split("/")[-1]
            task_error = 1 - correct_questions / total_questions
            print(ind, task_name, " Total_correct:", correct_questions, " total questions:", total_questions,
                  " task error:", task_error * 100, "%")
            num_passed_tasks += task_error <= 0.05
            task_errors.append(task_error)
        mean_error = np.mean(task_errors)
        print("TOTAL PASSED TASKS:", num_passed_tasks, "TOTAL TASKS:", num_tasks, " MEAN ERROR", mean_error)
        self.mean_test_errors.append(mean_error)
        print("ALL ERRORS: ", self.mean_test_errors)
        print("Time == ", time() - t)

    def cost(self, network_output, correct_output, mask=None):
        """
        Mean of the batch cross entropy cost on the masked softmax outputs.
        Mask makes all the outputs of the network which are not marked with the "-" in inputs completely irrelevant.


        :param network_output: tensor of shape [batch_size, time_steps, vector_size]
        :param correct_output: tensor of shape [batch_size, time_steps, vector_size]
        :param mask: 
        :return: 
        """
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=network_output,
                                                                        labels=correct_output,
                                                                        dim=2)
        masked = softmax_cross_entropy * mask[:, :, 0]
        tf.summary.image("0Masked_SCE", tf.reshape(masked, [1, 1, -1, 1]))
        return tf.reduce_mean(masked)

    def preprocess_files(self):
        word_to_ind = {bAbITask.output_symbol: 0, bAbITask.pad_symbol: 1}
        all_input_stories, all_output_stories = dict(), dict()
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
