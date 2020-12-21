import logging
import os
import torch
from tqdm import tqdm

from utils.tokenizer import StanfordTokenizer

from utils.embed import load_vocab, find_sentence_ids

from . import DataAllPreprocessor

logger = logging.getLogger(__name__)


class BasicRelNnlPreprocessor(DataAllPreprocessor):
    """
    Preprocessor:
    lower case, tokenizer, max_length, select samples(so doing init embedding here)
    lower case, tokenizer, max_length, build_vocab, select samples(so doing init embedding here)
    output examples shoule be list of dict:
    {...ex1..}
    {...ex2..}
    {...ex3..}
    """
    def __init__(self, label2id_dict, vocab_path, do_lower_case=False, drop_unk_samples=False):
        # self.reader = reader
        self.do_lower_case = do_lower_case
        self.label2id_dict = label2id_dict
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        # self.embed_model = embed_model
        # self.unk = self.embed_model.unk
        # self.padding = self.embed_model.padding   # no use here
        self.drop_unk_samples = drop_unk_samples
        self.word2id, self.id2word = load_vocab(vocab_path)
        self.tokenizer = StanfordTokenizer(self.do_lower_case)
        self.feature_keys = ["guids", "label_ids", "word_as", "word_bs", "word_a_tokens", "word_b_tokens",
                             "word_a_token_ids", "word_b_token_ids", "word_a_lengths", "word_b_lengths",
                             "labels"]

    def _data2feature(self, data):
        # preprocess
        if self.drop_unk_samples and data["label_id"] not in self.id2label_dict.keys():
            return dict()
        # Translation tables to remove parentheses from strings.

        this_output = data

        this_output["word_a_tokens"] = self.tokenizer.tokenize(this_output["word_a"])
        this_output["word_a_lengths"] = len(this_output["word_a_tokens"])
        this_output["word_b_tokens"] = self.tokenizer.tokenize(this_output["word_b"])
        this_output["word_b_lengths"] = len(this_output["word_b_tokens"])
        this_output["word_a_token_ids"] = torch.tensor(find_sentence_ids(self.word2id, this_output["word_a_tokens"]), dtype=int)
        this_output["word_b_token_ids"] = torch.tensor(find_sentence_ids(self.word2id, this_output["word_b_tokens"]), dtype=int)

        this_output["guids"] = this_output.pop("guid")
        this_output["label_ids"] = this_output.pop("label_id")
        this_output["labels"] = this_output.pop("label")
        this_output["word_as"] = this_output.pop("word_a")
        this_output["word_bs"] = this_output.pop("word_b")
        return this_output

    def _fea2tensor(self, fea):
        fea['word_a_token_ids'] = torch.nn.utils.rnn.pad_sequence(fea["word_a_token_ids"],
                                                                  batch_first=True,
                                                                  padding_value=self.word2id["[PAD]"])
        fea['word_b_token_ids'] = torch.nn.utils.rnn.pad_sequence(fea["word_b_token_ids"],
                                                                  batch_first=True,
                                                                  padding_value=self.word2id["[PAD]"])
        fea["word_a_lengths"] = torch.tensor(fea["word_a_lengths"], dtype=int)
        fea["word_b_lengths"] = torch.tensor(fea["word_b_lengths"], dtype=int)
        fea["label_ids"] = torch.tensor(fea["label_ids"], dtype=torch.long)
        return fea

    def preprocess(self, reader, data_dir, cache_dir, stage, overwrite, batch_size):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(
            cache_dir, "cached_{stage}_basicrel_nnl_preprocess_{size}_{lower}".format(
                stage=stage, size=len(self.id2word.keys()), lower=self.do_lower_case))
        self._preprocess(reader, data_dir, cached_examples_dir, stage, overwrite, batch_size=batch_size)
        return cached_examples_dir


