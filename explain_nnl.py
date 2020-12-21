import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loader_aligned import DataLoader
#from model.nnl.nnl_direct import NNL
from model.nnl.nnl_aligned_modular_final2 import NNL_Aligned as NNL
from model.nnl.nnl_aligned_modular import REL_LABEL_DICT
import pickle
from tqdm import tqdm
from utils.torch.esim import *

state_map = {'ind': 0, 'eq': 1, 'ent_f': 2, 'ent_r': 3, 'neg': 4, 'alt': 5, 'cov': 6}
reverse_map = {v: k for k, v in state_map.items()}

def correct_change(pos, union_state, true_switch, true_state):
    hit = 0
    for i in range(len(true_switch)):
        switch = true_switch[i][1]
        state = true_state[i]
        if pos in switch and union_state[pos] == state_map[state]:
            hit = 1
    return hit

def plot_matrix(ax, matrix, x, y, x_label, y_label):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax.matshow(matrix, cmap="Blues")

    xticks = np.arange(0, len(x), 1)
    ax.set_xticks(xticks)
    yticks = np.arange(0, len(y), 1)
    ax.set_yticks(yticks)
    ax.set_xticklabels(x)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(y)
    ax.set_yticklabels(ax.get_yticklabels())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.5, len(x)-0.5)
    ax.set_ylim(len(y)-0.5, -0.5)
    # plt.show()


def plot_analysis(analyze, fea, prob, label2id_dict):
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    # plot attention
    plot_matrix(ax1, analyze["hypothesis_attention"].squeeze(0).numpy(), fea["p_tokens"], fea["h_tokens"],
                "Premise", "Hypothesis")
    # plot basic rel
    plot_matrix(ax2, analyze["br_rel_1"].numpy(), [i[0] for i in sorted(REL_LABEL_DICT.items(), key=lambda d: d[1])],
                fea["h_tokens"], "Basic Relations", "Alignment (Aligned Premmise:Hypothesis)")
    # plot union
    plot_matrix(ax3, analyze["union_1"].squeeze(0).numpy(), [i[0] for i in sorted(REL_LABEL_DICT.items(), key=lambda d: d[1])],
                fea["h_tokens"], "Basic Relations", "Union Result Step by Step")
    # plot result prob
    plot_matrix(ax4, prob, [i[0] for i in sorted(label2id_dict.items(), key=lambda d: d[1])], ["Prob"], None, None)

    plt.tight_layout()
    # plt.show()
    return plt


if __name__ == "__main__":

    dataset = 'med2hop_bysnli_perturb'
    label2id_dict = {'entailment': 0,  'contradiction': 1, 'neutral': 2}
    id_label_dict = {id: label for label, id in label2id_dict.items()}
    #model_path = './results/nnl_saved_model_snli-aligned_modular'
    #model_path = './results/nnl_saved_model_snli-20200506-173831'
    #model_path = './results/nnl_saved_model_snli-20200506-225410'
    model_path = './results/nnl_saved_model_snli-memory_module'
    #model_path = './results/nnl_saved_model_snli-20200520-170221'
    #model_path = './results/nnl_saved_model_snli-20200509-130525'
    #model_path = './results/nnl_saved_model_snli-direct_84.25'


    plot_dir = os.path.join(model_path, 'plots_frag_quan')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    test_iterator = DataLoader('./data/{}/test_records.pkl'.format(dataset), align_file='./data/{}/test_align.pkl'.format(dataset))

    f = open('./data/{}/word_emb.pkl'.format(dataset), 'rb')
    embedding = torch.tensor(pickle.load(f)).cuda()

    model = NNL(embedding, hidden_size=300, dropout=0, num_classes=3)
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(model_path, 'nnl_model.pt')))

    model.eval()

    figure_list = []
    n_all = 0
    t_correct = 0
    shift_covered = []
    shift_correct = []
    category_hit = {}
    for idx, x1_batch, x2_batch, m1_batch, m2_batch, y_batch, p1_batch, p2_batch, align_1, align_2 \
            in tqdm(test_iterator.sampled_batch(batch_size=1, phase='test'),
                                                total=len(test_iterator), ascii=True):
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

        _, analyze, batch_pred, prob = model(x1_batch, m1_batch, p1_batch, align_1, x2_batch, m2_batch, p2_batch, align_2, y_batch, analyze=True)
        #_, analyze, batch_pred, prob = model(x1_batch, m1_batch, p1_batch, x2_batch, m2_batch, p2_batch, y_batch, analyze=True)
        prob = prob.detach().cpu().numpy()
        predict = np.argmax(prob, axis=1)


        n_sample = y_batch.shape[0]
        n_all += n_sample
        #pred_label = torch.argmax(batch_pred, dim=1)
        #for i in range(pred_label.size(0)):
        #    if pred_label[i] == 1:
        #        pred_label[i] = 2
        #t_correct += torch.sum(pred_label == y_batch).item()

        t_correct += torch.sum(torch.argmax(batch_pred, dim=1) == y_batch).item()
        pred_answer = torch.argmax(batch_pred, dim=1).item()
        true_answer = y_batch.item()

        sent_1, sent_2 = test_iterator.get_sentences(idx[0])
        sents = {'p_tokens': sent_1, 'h_tokens': sent_2}
        figure_list.append((idx[0], analyze, sents, prob, 'p{}-t{}'.format(pred_answer, true_answer)))
        
        #"""
        union = analyze['union_1']
        union_state = torch.argmax(union[0], dim=-1)
        true_switch = test_iterator.dataset[idx[0]]['switch_points']
        true_state = test_iterator.dataset[idx[0]]['switch_states']
        nop_states = test_iterator.dataset[idx[0]]['switch_states_nop']

        state_hit = []

        #if pred_answer != true_answer:
        #    continue
        #print(true_switch, true_state, 'nop:', nop_states)
        for i in range(len(true_switch)):
            switch = true_switch[i][1]
            state = true_state[i]
            #print(true_switch)
            #print(true_state)

            hit = 0
            pred_state = []
            for pos in switch:
                #pred_state.append(reverse_map[union_state[pos].item()])
                if pos < union_state.size(0) and union_state[pos].item() == state_map[state]:
                    hit = 1
            #print(pred_state, hit, switch, state)
            state_hit.append(hit)
        #print(state_hit)

        if len(true_switch) >= 2:
            category_key = true_state[0] + '+' + true_state[1]
            if category_key not in category_hit.keys():
                category_hit[category_key] = [0, 0, 0, 0]
            if state_hit[0] == 1 and state_hit[1] == 1:
                category_hit[category_key][3] += 1
            if state_hit[0] == 0 and state_hit[1] == 0:
                category_hit[category_key][0] += 1
            if state_hit[0] == 1 and state_hit[1] == 0:
                category_hit[category_key][1] += 1
            if state_hit[0] == 0 and state_hit[1] == 1:
                category_hit[category_key][2] += 1

        state_hit = sum(state_hit) / len(state_hit)
        #print('')
        shift_covered.append(state_hit)


        change = []
        for pos in range(union_state.size(0)):
            if pos == 0 and union_state[pos] != state_map['eq']:
                change.append(correct_change(pos, union_state, true_switch, true_state))
            elif pos > 0 and union_state[pos] != union_state[pos - 1]:
                change.append(correct_change(pos, union_state, true_switch, true_state))
        if len(change) > 0:
            change = sum(change) / len(change)
            shift_correct.append(change)
        #"""
        
    print("Accuarcy: {:.2f}% ".format(100.0 * t_correct / n_all))
    print("Shift_Covered {:.2f}%, State Correct {:.2f}%".format(100.0 * sum(shift_covered) / n_all, 100.0 * sum(shift_correct) / n_all))
    exit()
    for k, v in category_hit.items():
        print(k, v)
    for id, analyze, sents, prob, pred_true in figure_list:
        plt = plot_analysis(analyze, sents, prob, label2id_dict)
        premise = ' '.join(sents['p_tokens'])
        hypothesis = ' '.join(sents['h_tokens'])
        plt.savefig(os.path.join(plot_dir, str(id) + "_" + pred_true + ".png"))
        plt.close()







