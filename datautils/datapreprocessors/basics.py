import logging
import shutil
import torch
from tqdm import trange, tqdm
import os
import abc

logger = logging.getLogger(__name__)


class DataPreprocessor(metaclass=abc.ABCMeta):
    """ Preprocessor: Transform Data into Features """
    def __init__(self, label2id_dict, drop_unk_samples=True):
        # self.do_lower_case = do_lower_case
        self.drop_unk_samples = drop_unk_samples
        # self.reader = reader
        self.label2id_dict = label2id_dict
        self.id2label_dict = {id: label for label, id in self.label2id_dict.items()}
        # raise NotImplementedError

    @abc.abstractmethod
    def _preprocess_and_save(self, data, cached_examples_dir, **kwargs):

        index = dict()
        index['guids'] = []
        index['feafile_name'] = []
        index['offset'] = []
        index['label_ids'] = []

        self._save_index(cached_examples_dir, index)
        return

    def preprocess(self, reader, data_dir, cache_dir, stage, overwrite, **kwargs):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(cache_dir, "cached_{stage}_preprocess".format(stage=stage))
        self._preprocess(reader, data_dir, cached_examples_dir, stage, overwrite, **kwargs)
        return cached_examples_dir

    def _save_index(self, cached_examples_dir, index):
        logger.info("  >> Saving features into cached dir %s", cached_examples_dir)
        with open(os.path.join(cached_examples_dir, "index"), 'w') as f:
            f.write(str(self.label2id_dict) + "\n")
            for i in range(len(index["guids"])):
                f.write(index["guids"][i] + "\t" + index['feafile_name'][i] + "\t" +
                        str(index['offset'][i]) + "\t" + str(index["label_ids"][i]) + "\n")
        return

    def _preprocess(self, reader, data_dir, cached_examples_dir, stage, overwrite, **kwargs):
        if overwrite and os.path.exists(cached_examples_dir):
            shutil.rmtree(cached_examples_dir)

        if os.path.exists(cached_examples_dir) and not overwrite:
            logger.info("Loading examples from cached dir %s", cached_examples_dir)
        else:
            os.mkdir(cached_examples_dir)
            logger.info("  >> Preprocess {} data...".format(stage))
            if stage == "train":
                data = reader.get_train_examples(data_dir)
            elif stage == "dev":
                data = reader.get_dev_examples(data_dir)
            elif stage == "test":
                data = reader.get_test_examples(data_dir)

            self._preprocess_and_save(data, cached_examples_dir, **kwargs)
            return


class DataBatchPreprocessor(DataPreprocessor):
    """ Preprocessor: Transform Data into Features """
    def __init__(self, reader, drop_unk_samples=True):
        # self.do_lower_case = do_lower_case
        self.drop_unk_samples = drop_unk_samples
        self.reader = reader
        self.label2id_dict = self.reader.label2id_dict
        self.id2label_dict = self.reader.id2label_dict
        # raise NotImplementedError

    def preprocess(self, data_dir, cache_dir, stage, overwrite, **kwargs):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(cache_dir, "cached_{stage}_preprocess".format(stage=stage))
        self._preprocess(data_dir, cached_examples_dir, stage, overwrite, **kwargs)
        return cached_examples_dir

    def _preprocess_and_save(self, data, cached_examples_dir, file_size=100000, **kwargs):    # 100000
        index = dict()
        index['guids'] = []
        index['feafile_name'] = []
        index['offset'] = []
        index['label_ids'] = []

        for i in trange(0, len(data), file_size):
            this_feature = self._data2feature(data[i: i+file_size], **kwargs)
            feafile_name = "fea_{start}_{end}".format(start=i, end=i+len(data[i: i+file_size]))
            index["guids"].extend(this_feature["guids"])
            index['feafile_name'].extend([feafile_name] * len(this_feature["guids"]))
            index['offset'].extend(list(range(len(this_feature["guids"]))))
            index["label_ids"].extend(this_feature["label_ids"].numpy())

            this_feature = self._fea2tensor(this_feature)
            torch.save(this_feature, os.path.join(cached_examples_dir, feafile_name))

        self._save_index(cached_examples_dir, index)
        return

    def _data2feature(self, data, **kwargs):
        """ Transform one batch data to feature dict. Should contain `guids` and `label_ids` """
        raise NotImplementedError

    def _fea2tensor(self, fea):
        """ Transform feature into tensor """
        fea["label_ids"] = torch.tensor(fea["label_ids"], dtype=torch.long)
        return fea


class DataAllPreprocessor(DataPreprocessor):
    """ Preprocessor: Transform Data into Features """
    def __init__(self, reader, feature_keys, drop_unk_samples=True):
        # self.do_lower_case = do_lower_case
        self.drop_unk_samples = drop_unk_samples
        self.reader = reader
        self.label2id_dict = self.reader.label2id_dict
        self.id2label_dict = self.reader.id2label_dict
        self.feature_keys = feature_keys

    def preprocess(self, data_dir, cache_dir, stage, overwrite, **kwargs):
        assert stage in ["train", 'dev', 'test']
        cached_examples_dir = os.path.join(cache_dir, "cached_{stage}_preprocess".format(stage=stage))
        self._preprocess(data_dir, cached_examples_dir, stage, overwrite, **kwargs)
        return cached_examples_dir

    def _preprocess_and_save(self, data, cached_examples_dir, **kwargs):    # 100000
        index = dict()
        index['guids'] = []
        index['feafile_name'] = []
        index['offset'] = []
        index['label_ids'] = []
        feafile_name = "fea"

        output = dict()
        for key in self.feature_keys:
            output[key] = list()

        for ex in tqdm(data):
            this_output = self._data2feature(ex)
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

    def _data2feature(self, data, **kwargs):
        """ Transform one data to feature dict. Should contain `guids` and `label_ids` """
        raise NotImplementedError

    def _fea2tensor(self, fea):
        """ Transform feature into tensor """
        raise NotImplementedError
