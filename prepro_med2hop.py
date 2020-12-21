import pickle
import numpy as np
# import spacy
import os
from tqdm import tqdm
from multiprocessing import Pool
from stanfordnlp.server import CoreNLPClient
STANFORD_CORENLP_HOME = 'stanford-corenlp-full-2018-10-05'
os.environ['CORENLP_HOME'] = STANFORD_CORENLP_HOME

client = CoreNLPClient(annotators=['natlog'], timeout=60000, memory='16G')
label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
# nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", 'ner'])
projection_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}  # switch independent to 0, originally 6
aggregation_map = {'ind:ent_f':'ind', 'ind:ent_r':'ind', 'ent_f:alt':'alt',
                  'ent_f:ind':'ind', 'alt:ind':'ind', 'alt:ent_f':'ind', 'ent_r:ind':'ind',
                  'ent_r:alt':'alt', 'ind:alt': 'ind', 'alt:ent_r':'alt', 'ent_f:ent_r':'ind',
                  'ent_f:ent_f':'ent_f', 'ind:ind':'ind', 'ent_r:ent_f':'ind', 'ent_r:ent_r':'ent_r'}

def parse_switch_points(switch_points, switch_states):

    def separate(sent, s0, s1):
        left_brac = []
        right_brac = []
        for i, w in enumerate(sent):
            if w == s0:
                left_brac.append(i)
            if w == s1:
                right_brac.append(i)
        sep_list = []
        for l, r in zip(left_brac, right_brac):
            sep_list.append(sent[(l+1):r])
        return sep_list

    switch_point_tups = separate(switch_points, s0='(', s1=')')
    switch_point_pairs = []
    for tup in switch_point_tups:
        sp = separate(tup, s0='[', s1=']')
        span1 = [int(w.replace(' ', '')) for w in sp[0].split(',')]
        span2 = [int(w.replace(' ', '')) for w in sp[1].split(',')]
        switch_point_pairs.append((span1, span2))
    switch_states_list = switch_states.replace('\'', '').replace('*', '')\
        .replace('[', '').replace(']', '').replace(' ', '').split(',')
    #print(switch_states_list)
    agg_key = '{}:{}'.format(switch_states_list[0], switch_states_list[1])
    switch_states_list[1] = aggregation_map[agg_key] if agg_key in aggregation_map.keys() else 'ind'
    return switch_point_pairs, switch_states_list

def read_from_file(file):
    samples = []
    with open(file, 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            splits = line.split('\t')
            sentence_1 = splits[1].replace('*', '').replace('.', '')
            sentence_2 = splits[2].replace('*', '').replace('.', '')
            label = splits[11].replace('\n', '')
            switch_points = splits[10]
            switch_states_nop = splits[9].split('->')[-1]
            switch_states = splits[9].split('->')[-1]
            consistent = splits[8]
            if '[]' in switch_points or 'None' in switch_states:
                continue
            parsed_switch_points, parsed_switch_states = parse_switch_points(switch_points, switch_states)
            #print(switch_points, parsed_switch_points)
            #print(switch_states, parsed_switch_states)
            if consistent != 'True':
                continue
            samples.append((sentence_1.lower(), sentence_2.lower(), label, parsed_switch_points, parsed_switch_states, switch_states_nop))
    #print(len(samples))
    return samples


def build_vocab(samples):
    print('build vocab from {} samples (train + dev)'.format(len(samples)))
    word2idx = {'<pad>': 0, '<unk>': 1}
    for samples in tqdm(samples, total=len(samples), ascii=True):
        for tk in samples['sent_1_tokens'] + samples['sent_2_tokens']:
            if tk not in word2idx.keys():
                word2idx[tk] = len(word2idx)
    return word2idx


def build_word_embedding(word2idx, emb_file, vec_size=300):
    print('loading word embedding from {}'.format(emb_file))
    word_embedding = 0.1*np.random.randn(len(word2idx), vec_size).astype(np.float32)
    has_embeding = 0
    with open(emb_file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in word2idx.keys():
                word_embedding[word2idx[word]] = vector
                has_embeding += 1
    print("{} / {} tokens have corresponding embedding vector".format(
        has_embeding, len(word2idx)))
    return word_embedding


def tokenize(samples):

    def polarity(token):
        polarity = [token.polarity.projectEquivalence, token.polarity.projectForwardEntailment,
                    token.polarity.projectReverseEntailment, token.polarity.projectNegation,
                    token.polarity.projectAlternation, token.polarity.projectCover,
                    token.polarity.projectIndependence]
        return [projection_map[p] for p in polarity]

    print('tokenize data of {} samples'.format(len(samples)))
    tokenized_samples = []
    for s1, s2, lb, switch_points, switch_states, switch_states_nop in tqdm(samples, total=len(samples), ascii=True):
        instance = {}
        s1 = client.annotate(s1).sentence
        s2 = client.annotate(s2).sentence
        sent_1_tokens = []
        for sp in s1:
            sent_1_tokens += [tk.word.lower() for tk in sp.token]
        sent_2_tokens = []
        for sp in s2:
            sent_2_tokens += [tk.word.lower() for tk in sp.token]
        instance['sent_1_tokens'] = sent_1_tokens.copy()
        instance['sent_2_tokens'] = sent_2_tokens.copy()
        instance['switch_points'] = switch_points
        instance['switch_states'] = switch_states
        instance['switch_states_nop'] = switch_states_nop


        instance['sent_1_projection'] = []
        for sp in s1:
            instance['sent_1_projection'] += [polarity(t) for t in sp.token]
        instance['sent_2_projection'] = []
        for sp in s2:
            instance['sent_2_projection'] += [polarity(t) for t in sp.token]
        instance['y'] = label_dict[lb]
        assert len(instance['sent_1_projection']) == len(instance['sent_1_tokens'])
        assert len(instance['sent_2_projection']) == len(instance['sent_2_tokens'])

        tokenized_samples.append(instance)

    return tokenized_samples


def convert(tokenized_samples, word2idx, pad=128):
    print('convert data of {} samples'.format(len(tokenized_samples)))
    for ind, sample in enumerate(tqdm(tokenized_samples, total=len(tokenized_samples), ascii=True)):
        sent_1_idx = [0] * pad
        sent_1_mask = [0] * pad
        sent_2_idx = [0] * pad
        sent_2_mask = [0] * pad
        for i, tk in enumerate(sample['sent_1_tokens']):
            sent_1_idx[i] = word2idx[tk] if tk in word2idx.keys() else word2idx['<unk>']
            sent_1_mask[i] = 1
        for i, tk in enumerate(sample['sent_2_tokens']):
            sent_2_idx[i] = word2idx[tk] if tk in word2idx.keys() else word2idx['<unk>']
            sent_2_mask[i] = 1
        tokenized_samples[ind]['sent_1_idxs'] = sent_1_idx
        tokenized_samples[ind]['sent_2_idxs'] = sent_2_idx
        tokenized_samples[ind]['sent_1_mask'] = sent_1_mask
        tokenized_samples[ind]['sent_2_mask'] = sent_2_mask

    return tokenized_samples


def main(file, emb, save, w2idx=None):

    # use multiprocessing on training data
    train_data = read_from_file(file[0])
    data_batch = []
    n_proc = 8
    n_train = len(train_data)
    batch_size = int(n_train / n_proc) + 1
    for i in range(n_proc):
        # print((i*batch_size),min(i*batch_size + batch_size, n_train) )
        data_batch.append(train_data[(i * batch_size): min(i * batch_size + batch_size, n_train)])

    p = Pool(n_proc)
    train_samples_list = p.map(tokenize, data_batch)
    train_samples = []
    for i in range(n_proc):
        train_samples += train_samples_list[i]
    assert len(train_samples) == n_train

    test_samples = train_samples

    with open(w2idx, 'rb') as f:
        word2idx = pickle.load(f)

    test_samples = convert(test_samples, word2idx)

    with open(save[0], 'wb') as f:
        print('saving {} test samples ... '.format(len(test_samples)))
        pickle.dump(test_samples, f)

    #with open(save[1], 'wb') as f:
    #    print('saving word2idx table ...')
    #   pickle.dump(word2idx, f)



if __name__ == '__main__':
    file = ['./preprocess/med_2hop.tsv']
    save = ['./data/med2hop_bysnli/test_records.pkl',
            './data/med2hop_bysnli/word_emb.pkl',
            './data/med2hop_bysnli/word2idx.pkl']
    embed = './preprocess/glove/glove.840B.300d.txt'
    word2idx = './data/snli/word2idx.pkl'
    main(file, embed, save, w2idx=word2idx)

