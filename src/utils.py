import os
import json


class DataPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, logdir):
        self.logdir = logdir

        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())

        self.path = os.path.join(DataPath.base, self.logdir, self.timestamp)
