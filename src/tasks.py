class Task:
    """
    Abstract class representing Task.
    Defines methods which will be called from the Controller class at various points during training.
    
    """

    def generate_data(self, batch_size, train=True, cost=9999):
        """
        Generates the next batch of train/test data, optionally based on the current performance of the network
        (current cost).

        :param batch_size:
        :param train: flag indicating wheter it's test or train sample we want
        :param cost: current cost of the network
        :return:
        """
        raise NotImplementedError()

    def cost(self, network_output, correct_output, mask=None):
        """
        Implementation of the cost function to be optimized.
        The object that this method returns will be optimized.
        
        :param network_output:
        :param correct_output:
        :param mask: optional parameter that could be used to discard some of the network outputs
        :return: 
        """
        raise NotImplementedError()

    def test(self, sess, outputs_tf, fd, batch_size):
        """
        Evaluates the performance of the network on the whole test set.
        Currently used only by bAbI task and has more info there.

        :param sess:
        :param outputs_tf:
        :param fd:
        :param batch_size:
        :return:
        """
        raise NotImplementedError()
