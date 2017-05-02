class Task:
    """
    Abstract class representing Task
    
    """

    def generate_data(self, cost):
        """
        
        :return: 
        """
        raise NotImplementedError()

    def cost(self, x, y, mask=None):
        """
        
        :param x: 
        :param y: 
        :return: 
        """
        raise NotImplementedError()
