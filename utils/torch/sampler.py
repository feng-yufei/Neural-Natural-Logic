
import random

from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, data_source, classed_labels, max_n=1, max_ratio=0.9, sample_num=None):
        self.data_source = data_source
        self.classed_labels = classed_labels
        self.class_num = len(self.classed_labels)
        assert max_n < self.class_num
        self.max_num = sorted([len(i) for i in classed_labels], reverse=True)[max_n-1]
        self.num_samples_per_class = int(max_ratio * self.max_num)
        if sample_num is not None:
            self.num_samples_per_class = sample_num


    def __iter__(self):
        sample_list = []
        for c in self.classed_labels:
            if len(c) < self.num_samples_per_class:
                times = self.num_samples_per_class // len(c)
                rest = self.num_samples_per_class % len(c)
                sample_list.extend(list(range(len(c))) * times + random.sample(c, rest))
            else:
                sample_list.extend(random.sample(c, self.num_samples_per_class))
        random.shuffle(sample_list)
        return iter(sample_list)


    def __len__(self):
        return self.num_samples_per_class * self.class_num

