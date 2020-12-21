"""Data Reader for Basic Rel dataset"""

import os
import logging

from tqdm import tqdm

from . import DataReader

logger = logging.getLogger(__name__)

LABEL_DICT = {'eq': 1, 'ent_f': 2, 'ent_r': 3, 'neg': 4, 'alt': 5, 'cov': 6, 'ind': 0}

class BasicRelReader(DataReader):
    """
    Data Reader for Basic Rel dataset
    Retrun of get_xx_examples: dict list. each example is a dict
    """
    def __init__(self):
        self.label2id_dict = LABEL_DICT
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        self.data_keys = ['guid', 'word_a', 'word_b', 'label_id', 'label']


    def get_train_examples(self, data_dir):
        examples = []
        logger.info("Read Basic Rel Train set...")
        for label in ["eq", "ent", "neg", "alt", "cov", "ind"]:
            logger.info("  >> train_{}".format(label))
            examples += self._create_examples(os.path.join(data_dir, "train", label+"_train.txt"), label, "train")
        return examples


    def get_dev_examples(self, data_dir):
        examples = []
        logger.info("Read Basic Rel Dev set...")
        for label in ["eq", "ent", "neg", "alt", "cov", "ind"]:
            logger.info("  >> dev_{}".format(label))
            examples += self._create_examples(os.path.join(data_dir, "dev", label+"_dev.txt"), label, "dev")

        return examples


    def get_test_examples(self, data_dir):
        examples =[]
        logger.info("Read Basic Rel Test set...")
        for label in ["eq", "ent", "neg", "alt", "cov", "ind"]:
            logger.info("  >> test_{}".format(label))
            examples += self._create_examples(os.path.join(data_dir, "test", label+"_test.txt"), label, "test")
        return examples


    #
    # def label2id(self, label):
    #     tmp = label.strip()
    #     return self.label2id_dict[tmp]
    #
    #
    # def id2label(self, id):
    #     return self.id2label_dict[id]


    def _create_examples(self, file_path, file_label, ex_type):
        examples = []
        if file_label == 'ent':
            file_label = 'ent_r'
            with open(file_path, 'r') as f:
                for (i, line) in enumerate(tqdm(f)):
                    guid = "%s-%s-%s" % (ex_type, file_label, i)
                    label = file_label
                    label_id = self.label2id(file_label)
                    w_list = line.strip().split('\t')
                    word_a = w_list[1]
                    word_b = w_list[0]
                    ex = dict()
                    for k in self.data_keys:
                        ex[k] = eval(k)
                    examples.append(ex)
            file_label = 'ent_f'

        with open(file_path, 'r') as f:
            for (i, line) in enumerate(tqdm(f)):
                guid = "%s-%s-%s" % (ex_type, file_label, i)
                label = file_label
                label_id = self.label2id(file_label)
                w_list = line.strip().split('\t')
                word_a = w_list[0]
                word_b = w_list[1]

                ex = dict()
                for k in self.data_keys:
                    ex[k] = eval(k)
                examples.append(ex)
        logger.info("    {} examples".format(len(examples)))
        return examples


if __name__ == '__main__':
    from config import MINI_BR_DIR

    loader = BasicRelReader()
    dev_set = loader.get_dev_examples(MINI_BR_DIR)

