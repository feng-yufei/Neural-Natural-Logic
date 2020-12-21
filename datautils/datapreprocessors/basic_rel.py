import logging
import os
import torch
import shutil

from tqdm import trange
import numpy as np

from . import DataBatchPreprocessor

logger = logging.getLogger(__name__)


class BasicRelPreprocessor(DataBatchPreprocessor):
    """
    Preprocessor:
    lower case, tokenizer, max_length, select samples(so doing init embedding here)
    lower case, tokenizer, max_length, build_vocab, select samples(so doing init embedding here)
    output examples shoule be list of dict:
    {...ex1..}
    {...ex2..}
    {...ex3..}
    """
    def __init__(self, label2id_dict, embed_model, drop_unk_samples=False):
        # self.reader = reader
        self.label2id_dict = label2id_dict
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        self.embed_model = embed_model
        self.unk = self.embed_model.unk
        # self.padding = self.embed_model.padding   # no use here
        self.drop_unk_samples = drop_unk_samples

    def _to_fealist(self, data):
        d = {
            'guids': [],
            'word_as': [],
            'word_bs': [],
            'labels': [],
            'label_ids': []
        }
        for idx, ex in enumerate(data):
            d["guids"].append(ex["guid"])
            d["word_as"].append(ex["word_a"])
            d["word_bs"].append(ex["word_b"])
            d["labels"].append(ex["label"])
            d["label_ids"].append(ex["label_id"])
        return d

    def _data2feature(self, data, batch_size):
        feature = self._to_fealist(data)
        fea_a, mask_a = self.embed_model.get_output_embeddings(feature['word_as'], batch_size)
        fea_b, mask_b = self.embed_model.get_output_embeddings(feature['word_bs'], batch_size)
        # fea_a = mask_a[:, :, None]*fea_a
        # fea_b = mask_b[:, :, None]*fea_b

        if self.drop_unk_samples:
            # Just check the first dim of all tokens
            a_test = fea_a[:, 0, :]
            b_test = fea_b[:, 0, :]

            condition_a = a_test != self.unk
            row_cond_a = condition_a.all(1)
            condition_b = b_test != self.unk
            row_cond_b = condition_b.all(1)
            row_cond = row_cond_a * row_cond_b
            fea_a = fea_a[row_cond, :, :]
            fea_b = fea_b[row_cond, :, :]
            mask_a = mask_a[row_cond, :]
            mask_b = mask_b[row_cond, :]
            for k in feature.keys():
                feature[k] = [feature[k][i] for i in range(len(row_cond)) if row_cond[i]]

        # Compute average
        feature['word_a_embeds'] = np.sum(fea_a, axis=1, keepdims=True) / (np.sum(mask_a, axis=1, keepdims=True)[:, :, None])
        feature['word_b_embeds'] = np.sum(fea_b, axis=1, keepdims=True) / (np.sum(mask_b, axis=1, keepdims=True)[:, :, None])
        return feature

    def _fea2tensor(self, fea):
        fea['word_a_embeds'] = torch.tensor(fea, dtype=torch.float32)
        fea['word_b_embeds'] = torch.tensor(fea, dtype=torch.float32)
        fea["label_ids"] = torch.tensor(fea["label_ids"], dtype=torch.long)
        return fea

    def preprocess(self, reader, data_dir, cache_dir, stage, overwrite, batch_size):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(
            cache_dir, "cached_{stage}_basicrel_preprocess_{embed_model}_{dim}_{max_length}".format(
                stage=stage, embed_model=self.embed_model.model_name, dim=self.embed_model.dim,
                max_length=self.embed_model.max_length)
        )
        self._preprocess(reader, data_dir, cached_examples_dir, stage, overwrite, batch_size=batch_size)
        return cached_examples_dir


