import os
from logger import Logger
from loader_aligned import DataLoader
import numpy as np
from time import time, strftime
import torch
from tqdm import tqdm
from model.nnl.nnl_aligned_modular import NNL_Aligned as NNL
from torch.optim import Adam
import pickle
import random
import shutil


# code saving and logging
dataset = 'snli'
save_path = './results/{}-{}'.format('nnl_saved_model_{}'.format(dataset), strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_path):
    os.mkdir(save_path)
logger = Logger(os.path.join(save_path, 'log.txt'), log_type='w+')

source = ['train_aligned.py', 'loader_aligned.py', 'model/nnl/nnl_aligned_modular.py', 'model/nnl/relnn_aligned_modular.py']
for src in source:
    shutil.copyfile('./{}'.format(src), os.path.join(save_path, src.split('/')[-1]))


def run_epoch(model, data_iterator, optimizer, scheduler=None, phase='train', batch_size=16):

    if phase == 'train':
        model.train()
    else:
        model.eval()

    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()
    count = [0, 0, 0]
    for idx, x1_batch, x2_batch, m1_batch, m2_batch, y_batch, p1_batch, p2_batch, align_1, align_2\
            in tqdm(data_iterator.sampled_batch(batch_size=batch_size, phase=phase),
                                                total=int(len(data_iterator) / batch_size), ascii=True):

        x1_batch = torch.tensor(x1_batch, dtype=torch.int64).cuda()
        x2_batch = torch.tensor(x2_batch, dtype=torch.int64).cuda()
        m1_batch = torch.tensor(m1_batch, dtype=torch.float32).cuda()
        m2_batch = torch.tensor(m2_batch, dtype=torch.float32).cuda()
        y_batch = torch.tensor(y_batch, dtype=torch.int64).cuda()
        p1_batch = torch.tensor(p1_batch, dtype=torch.int64).cuda()
        p2_batch = torch.tensor(p2_batch, dtype=torch.int64).cuda()
        align_1 = torch.tensor(align_1, dtype=torch.float32).cuda()
        align_2 = torch.tensor(align_2, dtype=torch.float32).cuda()

        # forward
        batch_loss, _, batch_pred = model(x1_batch, m1_batch, p1_batch, align_1, x2_batch, m2_batch, p2_batch, align_2, y_batch)
        batch_loss = batch_loss.mean()

        # update model params
        if phase == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss.item() * n_sample

        t_correct += torch.sum(torch.argmax(batch_pred, dim=1) == y_batch).item()

    logger.cache_in("{} Loss: {:.4f},  Accuarcy: {:.2f}%, {:.2f} Seconds Used:".
          format(phase, t_loss / n_all, 100.0 * t_correct / n_all, time() - t0))

    if phase == 'dev':
        scheduler.step(1.0 * t_correct / n_all)
    #print(count)

    return 1.0 * t_correct / n_all


if __name__ == "__main__":

    # default for all
    max_word = 128
    batch_size = 128
    learning_rate = 0.0004
    label_size = 3
    n_epochs = 32
    init_checkpoint = ''


    torch.manual_seed(1211)
    np.random.seed(1211)
    random.seed(1211)

    train_iterator = DataLoader('./data/{}/train_records.pkl'.format(dataset), './data/{}/train_align.pkl'.format(dataset))
    test_iterator = DataLoader('./data/{}/test_records.pkl'.format(dataset), './data/{}/test_align.pkl'.format(dataset))
    if os.path.exists('./data/{}/dev_records.pkl'.format(dataset)):
        dev_iterator = DataLoader('./data/{}/dev_records.pkl'.format(dataset), './data/{}/dev_align.pkl'.format(dataset))
    else:
        dev_iterator = test_iterator

    f = open('./data/{}/word_emb.pkl'.format(dataset), 'rb')
    embedding = torch.tensor(pickle.load(f)).cuda()

    nnl_model = NNL(embedding, hidden_size=300, padding_idx=0, dropout=0.5, num_classes=3).cuda()
    logger.cache_in(str(nnl_model), to_print=False)
    if init_checkpoint != '':
        logger.cache_in('Loading pretrained model : {}'.format(init_checkpoint))
        nnl_model.load_state_dict(torch.load(init_checkpoint))

    # traininig sample size and warming up
    optimizer = Adam(filter(lambda x: x.requires_grad, nnl_model.parameters()), lr=learning_rate, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)

    best_dev = 0
    logger.cache_in('Start Training ... ')
    for i in range(0, n_epochs):
        logger.cache_in('Epoch {}...'.format(i))
        run_epoch(nnl_model, train_iterator, optimizer, phase='train', batch_size=batch_size)
        with torch.no_grad():
            dev_acc = run_epoch(nnl_model, dev_iterator, optimizer, scheduler=scheduler, phase='dev', batch_size=batch_size)
            run_epoch(nnl_model, test_iterator, optimizer, phase='test', batch_size=batch_size)

        # saving best dev model
        if dev_acc > best_dev:
            torch.save(nnl_model.state_dict(), os.path.join(save_path, 'nnl_model.pt'))
            logger.cache_in('Model saved at {}'.format(os.path.join(save_path, 'nnl_model.pt')))
            best_dev = dev_acc
        logger.cache_in('')
        logger.cache_write()

