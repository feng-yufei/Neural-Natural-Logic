"""Data Reader for the SNLI 1.0 data set"""

import os
import logging
from tqdm import tqdm
from . import DataReader
from utils.data import read_tsv

logger = logging.getLogger(__name__)

class SnliReader(DataReader):
    """
    Data Reader for SNLI1.0
    Notice: There are some '-' gold label in the dataset, they are labeled as -1.
            There should be 5 labels; however in training set most example lacks labels, no label will also be labeld as -1

    Retrun of get_xx_examples: dict list. each example is a dict
    """
    def __init__(self):
        self.label2id_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        self.data_keys = ['guid', 'pair_id', 'gold_label_id', 'several_labels', 'premise', 'hypothesis',
                          'premise_bp', 'hypothesis_bp', 'premise_p', 'hypothesis_p'] #,
                          # "premise_length", "hypothesis_length"]


    def get_train_examples(self, data_dir):
        logger.info("Read SNLI Train set...")
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "snli_1.0_train.txt")), "train")


    def get_dev_examples(self, data_dir):
        logger.info("Read SNLI Dev set...")
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "snli_1.0_dev.txt")), "dev")


    def get_test_examples(self, data_dir):
        logger.info("Read SNLI Test set...")
        return self._create_examples(
            read_tsv(os.path.join(data_dir, "snli_1.0_test.txt")), "test")


    # def get_labels(self):
    #     return self.label2id_dict.keys()
    #
    #
    # def label2id(self, label):
    #     tmp = label.strip()
    #     return self.label2id_dict[tmp] if tmp in self.label2id_dict.keys() else -1
    #
    #
    # def id2label(self, id):
    #     return self.id2label_dict[id]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            if i == 0:
                continue
            gold_label_id = self.label2id(line[0])
            several_labels = [self.label2id(line[-5]), self.label2id(line[-4]), self.label2id(line[-3]),
                      self.label2id(line[-2]), self.label2id(line[-1])]

            pair_id = line[-6]
            premise = line[5]
            hypothesis = line[6]
            premise_bp = line[1]
            hypothesis_bp = line[2]
            premise_p = line[3]
            hypothesis_p = line[4]
            # premise_length = len(premise)
            # hypothesis_length = len(hypothesis)
            guid = "%s-%s" % (set_type, i)

            ex = dict()
            for k in self.data_keys:
                ex[k] = eval(k)
            examples.append(ex)
        logger.info("  {} examples".format(len(examples)))
        return examples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from config import SNLI_DIR

    loader = SnliReader()
    train_set = loader.get_train_examples(SNLI_DIR)
    dev_set = loader.get_dev_examples(SNLI_DIR)
    test_set = loader.get_test_examples(SNLI_DIR)

    # Dataset Numbers:
    # training set: 550152 examples
    # dev set: 10000 examples
    # test set: 10000 examples
    num_train = len(train_set)
    num_dev = len(dev_set)
    num_test = len(test_set)
    print("Dataset Numbers:")
    print("training set: {} examples".format(num_train))
    print("dev set: {} examples".format(num_dev))
    print("test set: {} examples".format(num_test))
    print("\n")


    # Analyze label distribution
    # Label Distribution:
    # training set: ent: 183416 (0.3333914990766188), neu: 182764 (0.3322063720571769), con: 183187 (0.332975250476232)
    # dev set: ent: 3329 (0.3329), neu: 3235 (0.3235), con: 3278 (0.3278)
    # test set: ent: 3368 (0.3368), neu: 3219 (0.3219), con: 3237 (0.3237)
    num_ent_train = len(list(filter(lambda x: x['gold_label_id'] == 0, train_set)))
    num_neu_train = len(list(filter(lambda x: x['gold_label_id'] == 1, train_set)))
    num_con_train = len(list(filter(lambda x: x['gold_label_id'] == 2, train_set)))
    num_ent_dev = len(list(filter(lambda x: x['gold_label_id'] == 0, dev_set)))
    num_neu_dev = len(list(filter(lambda x: x['gold_label_id'] == 1, dev_set)))
    num_con_dev = len(list(filter(lambda x: x['gold_label_id'] == 2, dev_set)))
    num_ent_test = len(list(filter(lambda x: x['gold_label_id'] == 0, test_set)))
    num_neu_test = len(list(filter(lambda x: x['gold_label_id'] == 1, test_set)))
    num_con_test = len(list(filter(lambda x: x['gold_label_id'] == 2, test_set)))
    print("Label Distribution:")
    print("training set: ent: {} ({}), neu: {} ({}), con: {} ({})".format(num_ent_train, num_ent_train / num_train,
                                                                          num_neu_train, num_neu_train / num_train,
                                                                          num_con_train, num_con_train / num_train))
    print("dev set: ent: {} ({}), neu: {} ({}), con: {} ({})".format(num_ent_dev, num_ent_dev / num_dev,
                                                                     num_neu_dev, num_neu_dev / num_dev,
                                                                     num_con_dev, num_con_dev / num_dev))
    print("test set: ent: {} ({}), neu: {} ({}), con: {} ({})".format(num_ent_test, num_ent_test / num_test,
                                                                      num_neu_test, num_neu_test / num_test,
                                                                      num_con_test, num_con_test / num_test))
    print("\n")

    # Analyze number of confused label
    # Useless Examples:
    # 785 (0.0014268783899722259) examples in training set has no gold label
    # 158 (0.0158)  examples in dev set has no gold label
    # 176 (0.0176)  examples in test set has no gold label
    #
    # Label Number:
    # training set 1: 510711 (0.9283089037211534), 2: 0 (0.0), 3: 37 (6.725414067385014e-05), 4: 2628 (0.004776861667321031), 5: 36776 (0.06684698047085169)
    # dev set: 1: 0 (0.0), 2: 0 (0.0), 3: 0 (0.0), 4: 14 (0.0014), 5: 9986 (0.9986)
    # test set: 1: 0 (0.0), 2: 0 (0.0), 3: 0 (0.0), 4: 10 (0.001), 5: 9990 (0.999)
    # Error label:
    # training set: no conflict: 535366 (0.9731237912431473), 1 error: 9975 (0.01813135278977446), 2 errors: 4811 (0.00874485596707819), 3 errors: 0 (0.0)
    # dev set: no conflict: 5641 (0.5641), 1 error: 2852 (0.2852), 2 errors: 1507 (0.1507), 3 errors: 0 (0.0)
    # test set: no conflict: 5550 (0.555), 1 error: 2871 (0.2871), 2 errors: 1579 (0.1579), 3 errors: 0 (0.0)
    no_label_train = len(list(filter(lambda x: x['gold_label_id'] == -1, train_set)))
    no_label_dev = len(list(filter(lambda x: x['gold_label_id'] == -1, dev_set)))
    no_label_test = len(list(filter(lambda x: x['gold_label_id'] == -1, test_set)))
    print("Useless Examples:")
    print("{} ({}) examples in training set has no gold label".format(no_label_train, no_label_train/num_train))
    print("{} ({})  examples in dev set has no gold label".format(no_label_dev, no_label_dev/num_dev))
    print("{} ({})  examples in test set has no gold label".format(no_label_test, no_label_test/num_test))
    print("\n")

    label1_train = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 1, train_set)))
    label1_dev = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 1, dev_set)))
    label1_test = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 1, test_set)))
    label2_train = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 2, train_set)))
    label2_dev = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 2, dev_set)))
    label2_test = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 2, test_set)))
    label3_train = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 3, train_set)))
    label3_dev = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 3, dev_set)))
    label3_test = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 3, test_set)))
    label4_train = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 4, train_set)))
    label4_dev = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 4, dev_set)))
    label4_test = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 4, test_set)))
    label5_train = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 5, train_set)))
    label5_dev = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 5, dev_set)))
    label5_test = len(list(filter(lambda x: len([i for i in x['several_labels'] if i != -1]) == 5, test_set)))
    print("Label Number:")
    print("training set 1: {} ({}), 2: {} ({}), 3: {} ({}), 4: {} ({}), 5: {} ({})".format(
        label1_train, label1_train / num_train, label2_train, label2_train / num_train,
        label3_train, label3_train / num_train, label4_train, label4_train / num_train,
        label5_train, label5_train / num_train))
    print("dev set: 1: {} ({}), 2: {} ({}), 3: {} ({}), 4: {} ({}), 5: {} ({})".format(
        label1_dev, label1_dev / num_dev, label2_dev, label2_dev / num_dev,
        label3_dev, label3_dev / num_dev, label4_dev, label4_dev / num_dev,
        label5_dev, label5_dev / num_dev))
    print("test set: 1: {} ({}), 2: {} ({}), 3: {} ({}), 4: {} ({}), 5: {} ({})".format(
        label1_test, label1_test / num_test, label2_test, label2_test / num_test,
        label3_test, label3_test / num_test, label4_test, label4_test / num_test,
        label5_test, label5_test / num_test))
    print("\n")


    def err_num(gold_label, labels):
        """
        return the number of error labels
        :param gold_label: gold label
        :param labels: label list
        :return:
        """
        return len([x for x in labels if (gold_label != -1 and x != -1 and x != gold_label)])


    no_conflict_train = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 0, train_set)))
    no_conflict_dev = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 0, dev_set)))
    no_conflict_test = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 0, test_set)))
    err_1_train = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 1, train_set)))
    err_1_dev = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 1, dev_set)))
    err_1_test = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 1, test_set)))
    err_2_train = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 2, train_set)))
    err_2_dev = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 2, dev_set)))
    err_2_test = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 2, test_set)))
    err_3_train = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 3, train_set)))
    err_3_dev = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 3, dev_set)))
    err_3_test = len(list(filter(lambda x: err_num(x['gold_label_id'], x['several_labels']) == 3, test_set)))
    print("Error label:")
    print("training set: no conflict: {} ({}), 1 error: {} ({}), 2 errors: {} ({}), 3 errors: {} ({})".format(
        no_conflict_train, no_conflict_train / num_train, err_1_train, err_1_train / num_train,
        err_2_train, err_2_train/ num_train, err_3_train, err_3_train / num_train))
    print("dev set: no conflict: {} ({}), 1 error: {} ({}), 2 errors: {} ({}), 3 errors: {} ({})".format(
        no_conflict_dev, no_conflict_dev / num_dev, err_1_dev, err_1_dev / num_dev,
        err_2_dev, err_2_dev / num_dev, err_3_dev, err_3_dev / num_dev))
    print("test set: no conflict: {} ({}), 1 error: {} ({}), 2 errors: {} ({}), 3 errors: {} ({})".format(
        no_conflict_test, no_conflict_test / num_test, err_1_test, err_1_test / num_test,
        err_2_test, err_2_test / num_test, err_3_test, err_3_test / num_test))
    print("\n")


    # Analyze the sentence length
    # Sentence Length(all, 95%):
    # training: 	premise: 402, 121.0 	hypothesis: 295, 67.0
    # dev: 	premise: 300, 131.0 	hypothesis: 232, 68.0
    # test: 	premise: 265, 130.0 	hypothesis: 159, 69.0
    p_len_train = [len(ex['premise']) for ex in train_set]
    p_len_dev = [len(ex['premise']) for ex in dev_set]
    p_len_test = [len(ex['premise']) for ex in test_set]

    h_len_train = [len(ex['hypothesis']) for ex in train_set]
    h_len_dev = [len(ex['hypothesis']) for ex in dev_set]
    h_len_test = [len(ex['hypothesis']) for ex in test_set]

    print("Sentence Length(all, 95%):")
    print("training: \tpremise: {}, {} \thypothesis: {}, {}".format(max(p_len_train), np.percentile(p_len_train, 95),
                                                                    max(h_len_train), np.percentile(h_len_train, 95)))
    print("dev: \tpremise: {}, {} \thypothesis: {}, {}".format(max(p_len_dev), np.percentile(p_len_dev, 95),
                                                               max(h_len_dev), np.percentile(h_len_dev, 95)))
    print("test: \tpremise: {}, {} \thypothesis: {}, {}".format(max(p_len_test), np.percentile(p_len_test, 95),
                                                                max(h_len_test), np.percentile(h_len_test, 95)))

    sns.jointplot(x=p_len_train, y=h_len_train, alpha=0.1)
    plt.show()
    sns.jointplot(x=p_len_dev, y=h_len_dev, alpha=0.1)
    plt.show()
    sns.jointplot(x=p_len_test, y=h_len_test, alpha=0.1)
    plt.show()



