import numpy as np
import pickle


def pad(input, max):
    return input + [[0, 0, 0, 0, 0, 0, 0]] * (max-len(input))

def array(x, dtype=np.int32):
    return np.array(x, dtype=dtype)


def load_pkl(file):
    # load pickle file
    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data


class DataLoader:

    def __init__(self, data_file,  max_word=128):

        self.max_word = max_word
        self.dataset = load_pkl(data_file)#[1001:]
        assert self.max_word == len(self.dataset[0]['sent_1_idxs'])
        assert self.max_word == len(self.dataset[0]['sent_2_idxs'])

        self.sample_index = np.arange(0, len(self.dataset))
        print('Training Samples: {} Loaded'.format(len(self.dataset)))

        self.pos = 0

        self.project_keys = ["project_ind", "project_eq", "project_ent_f", "project_ent_r",
                             "project_neg", "project_alt", "project_cov"]

    def __len__(self):
        return len(self.dataset)

    def iter_reset(self, shuffle=True):
        self.pos = 0
        if shuffle:
            np.random.shuffle(self.sample_index)

    def project(self, sentence_features):
        proj = []
        for features in sentence_features:
            feature_ = {'project_eq': features[0], 'project_ent_f': features[1], 'project_ent_r': features[2],
                        'project_neg': features[3], 'project_alt': features[4], 'project_cov': features[5],
                        'project_ind': features[6]}
            proj.append([feature_[t] for t in self.project_keys])

        return proj




    def sampled_batch(self, batch_size, phase='train'):

        index = self.sample_index
        n = len(self.dataset)
        # batch iterator, shuffle if train
        self.iter_reset(shuffle=True if phase == 'train' else False)

        while self.pos < n:

            Index = []

            X1_batch = []
            X2_batch = []

            M1_batch = []
            M2_batch = []
            Y_batch = []

            P1_batch = []
            P2_batch = []

            for i in range(batch_size):
                Index.append(index[self.pos])
                sample = self.dataset[index[self.pos]]
                X1_batch.append(sample['sent_1_idxs'])
                X2_batch.append(sample['sent_2_idxs'])
                Y_batch.append(sample['y'])
                M1_batch.append(sample['sent_1_mask'])
                M2_batch.append(sample['sent_2_mask'])
                P1_batch.append(pad(self.project(sample['sent_1_projection']), max=self.max_word))
                P2_batch.append(pad(self.project(sample['sent_2_projection']), max=self.max_word))
                self.pos += 1
                if self.pos >= n:
                    break

            yield Index, array(X1_batch), array(X2_batch), array(M1_batch), array(M2_batch), array(Y_batch),\
                  array(P1_batch), array(P2_batch)

    def get_data(self, index=0):
        return self.dataset[index]

    def get_sentences(self, index):
        sample = self.dataset[index]
        return sample['sent_1_tokens'], sample['sent_2_tokens']

if __name__ == "__main__":
    iterator = DataLoader(data_file='./data/snli/train_records.pkl')
    x = 0
    for x1, x2, m1, m2, y, p1, p2 in iterator.sampled_batch(1, 'dev'):
        l1 = np.sum(m1)
        l2 = np.sum(m2)
        print(x1[0, 0:l1])
        print(x2[0, 0:l2])
        print(m1[0, 0:l1])
        print(m2[0, 0:l2])
        print(y)
        print(p1[0, 0:l1])
        print(p2[0, 0:l2])
        x+= 1
        if x == 13:
            exit()