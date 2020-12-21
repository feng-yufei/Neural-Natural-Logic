import torch
import os
from . import LoadAllDataset

class SnliDataset(LoadAllDataset):
    def __init__(self, cached_examples_dir, premise_max_length=None, hypothesis_max_lengths=None):

        fea_list = ['guids', 'label_ids', "p_token_ids", "p_project_matrix", "p_length",
                    "h_token_ids", "h_project_matrix", "h_length"]
        # self.word2id = word2id_dict
        # self.vocab_size = max(self.word2id.values()) + 1
        self.premise_max_length = premise_max_length
        self.hypothesis_max_lengths = hypothesis_max_lengths


        self.cached_examples_dir = cached_examples_dir
        self.label2id_dict = None
        self.id2label_dict = None

        feafile_name = set()

        with open(os.path.join(cached_examples_dir, "index"), "r") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    self.label2id_dict = eval(line)
                    self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
                    continue
                idx = line.split()
                feafile_name.add(idx[1])

        self.data = dict()
        for i in feafile_name:
            this_fea = torch.load(os.path.join(self.cached_examples_dir, i))
            for key in this_fea.keys():
                if key not in fea_list:
                    continue
                if key not in self.data.keys():
                    self.data[key] = []
                self.data[key].extend(this_fea[key])


    def __getitem__(self, idx):
        this_item = dict()
        for key in self.data.keys():
            this_item[key] = self.data[key][idx]
        if self.premise_max_length is not None:
            this_item["p_token_ids"] = this_item["p_token_ids"][0: self.premise_max_length]
            if "p_project_matrix" in this_item.keys():
                this_item["p_project_matrix"] = this_item["p_project_matrix"][0: self.premise_max_length]
        if self.hypothesis_max_lengths is not None:
            this_item["h_token_ids"] = this_item["h_token_ids"][0: self.hypothesis_max_lengths]
            if "h_project_matrix" in this_item.keys():
                this_item["h_project_matrix"] = this_item["h_project_matrix"][0: self.hypothesis_max_lengths]

        return this_item
