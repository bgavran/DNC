import os
import re
import pickle
import numpy as np
from utils import project_path
from tasks import Task


class bAbITask(Task):
    base = os.path.join(project_path.base, "src", "task_implementations", "bAbI")
    output_symbol = "-"
    newstory_delimiter = " NEWSTORY "
    processed_append = "-processed.p"

    def __init__(self, tasks_dir, batch_size):
        self.tasks_dir = os.path.join(bAbITask.base, tasks_dir)
        self.processed_dir = tasks_dir + bAbITask.processed_append
        self.batch_size = batch_size
        self.files_path = []

        for f in os.listdir(self.tasks_dir):
            f_path = os.path.join(self.tasks_dir, f)
            if os.path.isfile(f_path):
                self.files_path.append(f_path)

        if not os.path.isfile(self.processed_dir):
            pickle.dump(self.preprocess_files(), open(self.processed_dir, "wb"))
            print("Pickled!", self.processed_dir)
        self.onehot, all_input_stories, all_output_stories = pickle.load(open(self.processed_dir, "rb"))

        def flatten(forest):
            return [leaf for tree in forest for leaf in tree]

        self.x = np.array(flatten(list(all_input_stories.values())))
        self.y = np.array(flatten(list(all_output_stories.values())))
        assert len(self.x) == len(self.y)

    @staticmethod
    def one_hot(x, n_elements):
        return np.array([np.eye(n_elements)[indices] for indices in x])

    def generate_data(self, cost=None):
        ind = np.random.randint(0, len(self.x), self.batch_size)
        x = bAbITask.one_hot(self.x[ind], len(self.onehot))
        y = bAbITask.one_hot(self.y[ind], len(self.onehot))
        # TODO sequence length output
        return x, y

    def preprocess_files(self):
        onehot, all_input_stories, all_output_stories = {bAbITask.output_symbol: 0}, dict(), dict()
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
                output_tokens = input_tokens.copy()

                for i, token in enumerate(input_tokens):
                    if token == "?":
                        input_tokens[i + 1] = input_tokens[i + 1].split(",")
                        output_tokens[i + 1] = ["-" for _ in range(len(input_tokens[i + 1]))]

                input_tokens = bAbITask.flatten_if_list(input_tokens)
                output_tokens = bAbITask.flatten_if_list(output_tokens)

                for token in input_tokens:
                    if token not in onehot:
                        onehot[token] = len(onehot)

                input_stories.append([onehot[elem] for elem in input_tokens])
                output_stories.append([onehot[elem] for elem in output_tokens])
            all_input_stories[file_path] = input_stories
            all_output_stories[file_path] = output_stories
        return onehot, all_input_stories, all_output_stories

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
    task = bAbITask(tasks_dir, batch_size)
    task.generate_data()
