import os
import json


class DataPath:
    # base = json.loads(open("config.json").read()).get("path", "")
    base = json.loads(open("config.json").read())["path"]

    def __init__(self, logdir):
        self.logdir = logdir

        from time import gmtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", gmtime())

        self.path = os.path.join(DataPath.base, self.logdir, self.timestamp)
