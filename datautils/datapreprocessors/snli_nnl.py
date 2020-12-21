import logging
import os
from tqdm import tqdm
import torch

from stanfordnlp.server import CoreNLPClient
from config import STANFORD_CORENLP_HOME
os.environ['CORENLP_HOME'] = STANFORD_CORENLP_HOME

from utils.embed import load_vocab, find_sentence_ids

from model.nnl.nnl import REL_LABEL_DICT

from . import DataAllPreprocessor

logger = logging.getLogger(__name__)

class SnliNnlPreprocessor(DataAllPreprocessor):
    def __init__(self, label2id_dict, vocab_path, do_lower_case=False, drop_unk_samples=True):
        self.do_lower_case = do_lower_case
        self.drop_unk_samples = drop_unk_samples
        # self.max_length = max_length
        # self.reader = reader
        self.label2id_dict = label2id_dict
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        # self.vocab_size = max(self.word2id.values()) + 1

        self.word2id, self.id2word = load_vocab(vocab_path)

        # preprocess param
        self.map_project = {0: REL_LABEL_DICT["eq"],
                            1: REL_LABEL_DICT["ent_f"],
                            2: REL_LABEL_DICT["ent_r"],
                            3: REL_LABEL_DICT["neg"],
                            4: REL_LABEL_DICT["alt"],
                            5: REL_LABEL_DICT["cov"],
                            6: REL_LABEL_DICT["ind"]}

        self.project_keys = ["project_ind", "project_eq", "project_ent_f", "project_ent_r",
                             "project_neg", "project_alt", "project_cov"]
        self.map_project_dir = {"up": 1, "down": 2, 'flat': 0}
        self.feature_keys = ['guids', 'pair_id', 'label_ids', 'several_labels',
                             'premise', 'premise_bp', 'premise_p',
                             'hypothesis', 'hypothesis_bp', 'hypothesis_p',
                             "p_tokens", "p_token_ids", "p_polarities", "p_polarity_dirs", "p_operators",
                             "p_project_matrix", "p_length",
                             "h_tokens", "h_token_ids", "h_polarities", "h_polarity_dirs", "h_operators",
                             "h_project_matrix", "h_length"]


    def _data2feature(self, data, client):
        # preprocess
        if self.drop_unk_samples and data["gold_label_id"] not in self.id2label_dict.keys():
            return dict()
        # Translation tables to remove parentheses from strings.
        parentheses_table = str.maketrans({"(": None, ")": None})

        this_output = data
        this_output["premise"] = this_output["premise"].translate(parentheses_table)
        this_output["hypothesis"] = this_output["hypothesis"].translate(parentheses_table)

        this_output["several_labels"] = tuple(this_output["several_labels"])

        this_output["p_tokens"] = []
        this_output["p_token_ids"] = []
        this_output["p_polarities"] = []
        this_output["p_polarity_dirs"] = []
        this_output["p_operators"] = []
        this_output["p_project_matrix"] = []
        this_output["p_length"] = 0
        this_output["h_tokens"] = []
        this_output["h_token_ids"] = []
        this_output["h_polarities"] = []
        this_output["h_polarity_dirs"] = []
        this_output["h_operators"] = []
        this_output["h_project_matrix"] = []
        this_output["h_length"] = 0

        for s in client.annotate(data["premise"]).sentence:
            if self.do_lower_case:
                this_output["p_tokens"].extend([t.word.lower() for t in s.token])
            else:
                this_output["p_tokens"].extend([t.word for t in s.token])
            this_output["p_token_ids"].extend(find_sentence_ids(self.word2id, this_output["p_tokens"]))
            this_output["p_polarity_dirs"].extend([self.map_project_dir[t.polarity_dir] for t in s.token])
            for t in s.token:
                op = dict()
                polarity = dict()
                project_matrix = list()
                op["name"] = t.operator.name
                op["quant_span_s"] = t.operator.quantifierSpanBegin
                op["quant_span_e"] = t.operator.quantifierSpanEnd
                op["sub_span_s"] = t.operator.subjectSpanBegin
                op["sub_span_e"] = t.operator.subjectSpanEnd
                op["obj_span_s"] = t.operator.objectSpanBegin
                op["obj_span_e"] = t.operator.objectSpanEnd

                polarity["project_eq"] = self.map_project[t.polarity.projectEquivalence]
                polarity["project_ent_f"] = self.map_project[t.polarity.projectForwardEntailment]
                polarity["project_ent_r"] = self.map_project[t.polarity.projectReverseEntailment]
                polarity["project_neg"] = self.map_project[t.polarity.projectNegation]
                polarity["project_alt"] = self.map_project[t.polarity.projectAlternation]
                polarity["project_cov"] = self.map_project[t.polarity.projectCover]
                polarity["project_ind"] = self.map_project[t.polarity.projectIndependence]

                if op["name"] == '':
                    op = {}
                else:
                    op["quant_span_s"] += this_output["p_length"]
                    op["quant_span_e"] += this_output["p_length"]
                    op["sub_span_s"] += this_output["p_length"]
                    op["sub_span_e"] += this_output["p_length"]
                    op["obj_span_s"] += this_output["p_length"]
                    op["obj_span_e"] += this_output["p_length"]

                for key in self.project_keys:
                    project_matrix.append(polarity[key])

                this_output["p_operators"].append(op)
                this_output['p_polarities'].append(polarity)
                this_output["p_project_matrix"].append(project_matrix)

            this_output["p_length"] += len(s.token)
        this_output["p_token_ids"] = torch.tensor(this_output["p_token_ids"], dtype=int)
        this_output["p_polarity_dirs"] = torch.tensor((this_output["p_polarity_dirs"]), dtype=int)
        this_output["p_project_matrix"] = torch.tensor(this_output["p_project_matrix"], dtype=int)


        for s in client.annotate(data["hypothesis"]).sentence:
            if self.do_lower_case:
                this_output["h_tokens"].extend([t.word.lower() for t in s.token])
            else:
                this_output["h_tokens"].extend([t.word for t in s.token])
            this_output["h_token_ids"].extend(find_sentence_ids(self.word2id, this_output["h_tokens"]))
            this_output["h_polarity_dirs"].extend([self.map_project_dir[t.polarity_dir] for t in s.token])
            for t in s.token:
                op = dict()
                polarity = dict()
                project_matrix = list()
                op["name"] = t.operator.name
                op["quant_span_s"] = t.operator.quantifierSpanBegin
                op["quant_span_e"] = t.operator.quantifierSpanEnd
                op["sub_span_s"] = t.operator.subjectSpanBegin
                op["sub_span_e"] = t.operator.subjectSpanEnd
                op["obj_span_s"] = t.operator.objectSpanBegin
                op["obj_span_e"] = t.operator.objectSpanEnd

                polarity["project_eq"] = self.map_project[t.polarity.projectEquivalence]
                polarity["project_ent_f"] = self.map_project[t.polarity.projectForwardEntailment]
                polarity["project_ent_r"] = self.map_project[t.polarity.projectReverseEntailment]
                polarity["project_neg"] = self.map_project[t.polarity.projectNegation]
                polarity["project_alt"] = self.map_project[t.polarity.projectAlternation]
                polarity["project_cov"] = self.map_project[t.polarity.projectCover]
                polarity["project_ind"] = self.map_project[t.polarity.projectIndependence]

                if op["name"] == '':
                    op = {}
                else:
                    op["quant_span_s"] += this_output["h_length"]
                    op["quant_span_e"] += this_output["h_length"]
                    op["sub_span_s"] += this_output["h_length"]
                    op["sub_span_e"] += this_output["h_length"]
                    op["obj_span_s"] += this_output["h_length"]
                    op["obj_span_e"] += this_output["h_length"]

                for key in self.project_keys:
                    project_matrix.append(polarity[key])

                this_output["h_operators"].append(op)
                this_output['h_polarities'].append(polarity)
                this_output["h_project_matrix"].append(project_matrix)

            this_output["h_length"] += len(s.token)

        this_output["h_token_ids"] = torch.tensor(this_output["h_token_ids"], dtype=int)
        this_output["h_polarity_dirs"] = torch.tensor((this_output["h_polarity_dirs"]), dtype=int)
        this_output["h_project_matrix"] = torch.tensor(this_output["h_project_matrix"], dtype=int)

        this_output["guids"] = this_output.pop("guid")
        this_output["label_ids"] = this_output.pop("gold_label_id")
        return this_output

    def _fea2tensor(self, fea):
        fea["p_token_ids"] = torch.nn.utils.rnn.pad_sequence(fea["p_token_ids"],
                                                                batch_first=True,
                                                                padding_value=self.word2id["[PAD]"])
        fea["h_token_ids"] = torch.nn.utils.rnn.pad_sequence(fea["h_token_ids"],
                                                                batch_first=True,
                                                                padding_value=self.word2id["[PAD]"])
        fea["p_project_matrix"] = torch.nn.utils.rnn.pad_sequence(fea["p_project_matrix"],
                                                                  batch_first=True,
                                                                  padding_value=self.word2id["[PAD]"])
        fea["h_project_matrix"] = torch.nn.utils.rnn.pad_sequence(fea["h_project_matrix"],
                                                                  batch_first=True,
                                                                  padding_value=self.word2id["[PAD]"])
        fea["p_length"] = torch.tensor(fea["p_length"], dtype=int)
        fea["h_length"] = torch.tensor(fea["h_length"], dtype=int)
        fea["label_ids"] = torch.tensor(fea["label_ids"], dtype=torch.long)
        return fea

    def _preprocess_and_save(self, data, cached_examples_dir):
        index = dict()
        index['guids'] = []
        index['feafile_name'] = []
        index['offset'] = []
        index['label_ids'] = []
        feafile_name = "fea"

        output = dict()

        for key in self.feature_keys:
            output[key] = list()
        with CoreNLPClient(annotators=['natlog'], timeout=60000, memory='16G') as client:
            for ex in tqdm(data):
                # # preprocess
                # if self.drop_unk_samples and ex["gold_label_id"] not in self.id2label_dict.keys():
                #     continue
                this_output = self._data2feature(ex, client)

                if len(this_output) == 0:
                    continue
                for key in self.feature_keys:
                    output[key].append(this_output[key])

        output = self._fea2tensor(output)

        index["guids"] = output["guids"]
        index['feafile_name'] = [feafile_name] * len(index["guids"])
        index['offset'] = list(range(len(index["guids"])))
        index["label_ids"] = output["label_ids"].numpy()

        torch.save(output, os.path.join(cached_examples_dir, feafile_name))
        self._save_index(cached_examples_dir, index)
        return

    def preprocess(self, reader, data_dir, cache_dir, stage, overwrite, **kwargs):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(cache_dir, "cached_{stage}_snli_nnl_preprocess_{size}_{lower}".format(
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
        with CoreNLPClient(annotators=['natlog'], timeout=60000, memory='16G') as client:
            fea = self._data2feature(sample, client)
        fea["p_length"] = torch.tensor(fea["p_length"], dtype=int)
        fea["h_length"] = torch.tensor(fea["h_length"], dtype=int)
        self.drop_unk_samples = tmp
        return fea


