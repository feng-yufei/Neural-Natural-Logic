import logging
import os
import torch
from tqdm import tqdm

from utils.embed import load_vocab, find_sentence_ids
from utils.tokenizer import StanfordTokenizer

from . import DataAllPreprocessor

logger = logging.getLogger(__name__)

class SnliPreprocessor(DataAllPreprocessor):
    def __init__(self, label2id_dict, vocab_path, do_lower_case=False, drop_unk_samples=True):
        self.do_lower_case = do_lower_case
        self.drop_unk_samples = drop_unk_samples
        # self.max_length = max_length
        # self.reader = reader
        self.label2id_dict = label2id_dict
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        # self.vocab_size = max(self.word2id.values()) + 1

        self.word2id, self.id2word = load_vocab(vocab_path)


        self.feature_keys = ['guids', 'pair_id', 'label_ids', 'several_labels',
                             'premise', 'premise_bp', 'premise_p',
                             'hypothesis', 'hypothesis_bp', 'hypothesis_p',
                             "p_tokens", "p_token_ids", "p_length",
                             "h_tokens", "h_token_ids", "h_length"]
        self.tokenizer = StanfordTokenizer(self.do_lower_case)

    def _data2feature(self, data):
        # preprocess
        if self.drop_unk_samples and data["gold_label_id"] not in self.id2label_dict.keys():
            return dict()
        # Translation tables to remove parentheses from strings.
        parentheses_table = str.maketrans({"(": None, ")": None})

        this_output = data
        this_output["premise"] = this_output["premise"].translate(parentheses_table)
        this_output["hypothesis"] = this_output["hypothesis"].translate(parentheses_table)
        this_output["several_labels"] = tuple(this_output["several_labels"])

        this_output["p_tokens"] = self.tokenizer.tokenize(this_output["premise"])
        this_output["p_length"] = len(this_output["p_tokens"])
        this_output["h_tokens"] = self.tokenizer.tokenize(this_output["hypothesis"])
        this_output["h_length"] = len(this_output["h_tokens"])

        this_output["p_token_ids"] = torch.tensor(find_sentence_ids(self.word2id, this_output["p_tokens"]), dtype=int)
        this_output["h_token_ids"] = torch.tensor(find_sentence_ids(self.word2id, this_output["h_tokens"]), dtype=int)

        this_output["guids"] = this_output["guid"]
        this_output["label_ids"] = this_output["gold_label_id"]
        this_output.pop("guid")
        this_output.pop("gold_label_id")
        return this_output

    def _fea2tensor(self, fea):
        fea["p_token_ids"] = torch.nn.utils.rnn.pad_sequence(fea["p_token_ids"],
                                                                batch_first=True,
                                                                padding_value=self.word2id["[PAD]"])
        fea["h_token_ids"] = torch.nn.utils.rnn.pad_sequence(fea["h_token_ids"],
                                                                batch_first=True,
                                                                padding_value=self.word2id["[PAD]"])
        fea["p_length"] = torch.tensor(fea["p_length"], dtype=int)
        fea["h_length"] = torch.tensor(fea["h_length"], dtype=int)
        fea["label_ids"] = torch.tensor(fea["label_ids"], dtype=torch.long)
        return fea

    def preprocess(self, reader, data_dir, cache_dir, stage, overwrite, **kwargs):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(cache_dir, "cached_{stage}_snli_preprocess_{size}_{lower}".format(
            stage=stage, size=len(self.id2word.keys()), lower=self.do_lower_case))
        self._preprocess(reader, data_dir, cached_examples_dir, stage, overwrite, **kwargs)
        return cached_examples_dir

    def preprocess_one_sample(self, premise, hypothesis):
        tmp = self.drop_unk_samples
        self.drop_unk_samples = False
        sample = {
                     'guid': '',
                     'pair_id': '',
                     'gold_label_id': -1,
                     'several_labels': [],
                     'premise': premise,
                     'premise_bp': "",
                     'premise_p': "",
                     'hypothesis': hypothesis,
                     'hypothesis_bp': "",
                     'hypothesis_p': ""
        }
        fea = self._data2feature(sample)
        fea["p_length"] = torch.tensor(fea["p_length"], dtype=int)
        fea["h_length"] = torch.tensor(fea["h_length"], dtype=int)
        self.drop_unk_samples = tmp
        return fea