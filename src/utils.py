import os


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, log_dir):
        self.log_dir = log_dir

        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())

        self.log_path = os.path.join(ProjectPath.base, self.log_dir, self.timestamp)
        self.train_path = os.path.join(self.log_path, "train")
        self.test_path = os.path.join(self.log_path, "test")
        self.model_path = os.path.join(self.train_path, "model.chpt")


project_path = ProjectPath("log")


# Wrapper for initializing all weights in all networks easily. Orthogonal, xavier and normal inits have different
# interfaces and this makes it a bit simpler
def init_wrapper(init_fn):
    def inner(shape, stddev):
        if stddev is None:
            return init_fn(shape)
        else:
            return init_fn(shape, stddev=stddev)

    return inner
