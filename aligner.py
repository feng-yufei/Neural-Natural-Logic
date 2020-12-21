import os
from loader import DataLoader
import numpy as np
from time import time, strftime
import torch
from tqdm import tqdm
from model.esim_hardaligner import ESIM_Aligner as ESIM
from torch.optim import Adam
import pickle
import random


# code saving and logging
dataset = 'med2hop_bysnli'
with open('./data/{}/word2idx.pkl'.format(dataset), 'rb') as f:
    word2idx = pickle.load(f)
idx2word = {v: k for k, v in word2idx.items()}


def run_epoch(model, data_iterator, phase='train', batch_size=16):

    model.eval()
    align_list = []

    for idx, x1_batch, x2_batch, m1_batch, m2_batch, y_batch, _, _ in tqdm(data_iterator.sampled_batch(batch_size=batch_size, phase='dev'),
                                                   total=int(len(data_iterator) / batch_size), ascii=True):

        x1_batch = torch.tensor(x1_batch, dtype=torch.int64).cuda()
        x2_batch = torch.tensor(x2_batch, dtype=torch.int64).cuda()
        m1_batch = torch.tensor(m1_batch, dtype=torch.float32).cuda()
        m2_batch = torch.tensor(m2_batch, dtype=torch.float32).cuda()
        y_batch = torch.tensor(y_batch, dtype=torch.int64).cuda()

        # forward
        (batch_loss, batch_pred, _), attention_mat = model(x1_batch, m1_batch, x2_batch, m2_batch, y_batch)
        top1_align = torch.argmax(attention_mat, dim=-1)

        for i in range(len(idx)):
            x1_align = np.zeros_like(x1_batch[0].cpu())
            x2_align = np.zeros_like(x2_batch[0].cpu())
            for w in range(attention_mat.shape[1]):
                tar = top1_align[i, w].item()
                if x2_batch[i, w].item() == x1_batch[i, tar].item():
                    x2_align[w] = 1
                    x1_align[tar] = 1
            align_list.append((x1_align, x2_align))

    with open('./data/{}/{}_align.pkl'.format(dataset, phase), 'wb') as dmp:
        pickle.dump(align_list, dmp)
        print('aligned words saved in ', './data/{}/{}_align.pkl'.format(dataset, phase))






if __name__ == "__main__":

    # default for all
    max_word = 128
    batch_size = 32
    learning_rate = 0.0000
    label_size = 3
    n_epochs = 64
    init_checkpoint = './results/esim_saved_model_snli-20200323-184427/esim_model.pt'

    torch.manual_seed(1211)
    np.random.seed(1211)
    random.seed(1211)

    # dataset
    train_iterator = DataLoader('./data/{}/train_records.pkl'.format(dataset))
    test_iterator = DataLoader('./data/{}/test_records.pkl'.format(dataset))
    if os.path.exists('./data/{}/dev_records.pkl'.format(dataset)):
        dev_iterator = DataLoader('./data/{}/dev_records.pkl'.format(dataset))
    else:
        dev_iterator = test_iterator

    # models and checkpoint loading
    f = open('./data/{}/word_emb.pkl'.format(dataset), 'rb')
    embedding = torch.tensor(pickle.load(f)).cuda()

    esim_model = ESIM(embedding, hidden_size=300, padding_idx=0, dropout=0.5, num_classes=3).cuda()
    print(str(esim_model))
    if init_checkpoint != '':
        print('Loading pretrained model : {}'.format(init_checkpoint))
        esim_model.load_state_dict(torch.load(init_checkpoint))

    print('Start Printing Alignment ... ')

    with torch.no_grad():
        run_epoch(esim_model, train_iterator, phase='train', batch_size=batch_size)
        run_epoch(esim_model, dev_iterator, phase='dev', batch_size=batch_size)
        run_epoch(esim_model, test_iterator, phase='test', batch_size=batch_size)

