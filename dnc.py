from controller import *
from memory import *


class DNC(Controller):
    def __init__(self, controller):
        super().__init__()
        memory_size = 1
        self.controller = controller
        self.memory = Memory(controller.batch_size, controller.output_size, controller.output_size, memory_size)

    def __call__(self, x):
        # batch_size, input_size, sequence_length = x.shape
        return self.controller(x)
        # DNC needs just one time step but controller returns output after all iterations
        pass

    def step(self, x):
        controller_output = self.controller.step(x)
        return controller_output + self.memory(controller_output)
