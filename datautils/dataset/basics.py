import os
import torch
from torch.utils.data import Dataset as Dataset
from utils import get_index_by_value


class LoadPerItemDataset(Dataset):
    # Has problem... Cannot load big dataset. T^T
    def __init__(self, cached_examples_dir):

        self.cached_examples_dir = cached_examples_dir
        self.label2id_dict = None
        self.id2label_dict = None

        self.index = dict()
        self.index["guids"] = []
        self.index["feafile_name"] = []
        self.index["offset"] = []
        self.index["label_ids"] = []

        with open(os.path.join(cached_examples_dir, "index"), "r") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    self.label2id_dict = eval(line)
                    self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
                    continue
                idx = line.split()
                self.index["guids"].append(idx[0])
                self.index["feafile_name"].append(idx[1])
                self.index["offset"].append(int(idx[2]))
                self.index["label_ids"].append(int(idx[-1]))

    def __len__(self):
        return len(self.index["guids"])

    def __getitem__(self, idx):
        this_batch = torch.load(os.path.join(self.cached_examples_dir, self.index["feafile_name"][idx]))
        this_item = dict()
        for key in this_batch.keys():
            this_item[key] = this_batch[key][self.index["offset"][idx]]
        del this_batch
        return this_item

    def get_index_by_label(self, label_id):
        return get_index_by_value(self.index["label_ids"], label_id)

    def label2id(self, label):
        return self.label2id_dict[label.strip()]

    def id2label(self, id):
        return self.id2label_dict[id]

    def get_classed_index(self):
        output = []
        for key in self.id2label_dict.keys():
            samples = self.get_index_by_label(key)
            if len(samples) > 0:
                output.append(samples)
        return output


class LoadAllDataset(Dataset):
    def __init__(self, cached_examples_dir):

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
                if key not in self.data.keys():
                    self.data[key] = []
                self.data[key].extend(this_fea[key])

    def __len__(self):
        return len(self.data["guids"])

    def __getitem__(self, idx):
        this_item = dict()
        for key in self.data.keys():
            this_item[key] = self.data[key][idx]
        return this_item

    def get_index_by_label(self, label_id):
        return get_index_by_value(self.data["label_ids"], label_id)

    def label2id(self, label):
        return self.label2id_dict[label.strip()]

    def id2label(self, id):
        return self.id2label_dict[id]

    def get_classed_index(self):
        output = []
        for key in self.id2label_dict.keys():
            samples = self.get_index_by_label(key)
            if len(samples) > 0:
                output.append(samples)
        return output
