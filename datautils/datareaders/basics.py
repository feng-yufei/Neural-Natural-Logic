import abc

class DataReader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.label2id_dict
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}


    @abc.abstractmethod
    def get_train_examples(self, data_dir):
        """ Get train examples """
        pass

    @abc.abstractmethod
    def get_dev_examples(self, data_dir):
        """ Get dev examples """
        pass

    @abc.abstractmethod
    def get_test_examples(self, data_dir):
        """ Get test examples """
        pass


    def label2id(self, label):
        tmp = label.strip()
        return self.label2id_dict[tmp] if tmp in self.label2id_dict.keys() else -1


    def id2label(self, id):
        return self.id2label_dict[id]
