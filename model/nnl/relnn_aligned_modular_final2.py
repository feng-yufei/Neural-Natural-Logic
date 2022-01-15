""" RelNet Structure """
import logging
import torch
from torch import nn
from utils.torch import batch_dot
from utils.torch.loss import symmetry_loss, reverse_loss
# from utils.embed.sentence import avg_embed

logger = logging.getLogger(__name__)

MODEL_VERSION = "v7"


NUM_REL = 7

class RelUnit(nn.Module):
    def __init__(self, d, dropout=0.0, num_heads=3):
        super(RelUnit, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        # self.rel = nn.Sequential(nn.Dropout(p=dropout),
        #                          nn.Linear(d, d, bias=False))
        self.rel = nn.ModuleList([nn.Sequential(nn.Dropout(p=dropout),
                                                nn.Linear(d, d, bias=False)) for _ in range(num_heads)])
        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, embed_a, embed_b):
        # embed_a = self.rel(embed_a)
        # embed_b = self.rel(embed_b)
        embed_a = torch.cat([unit(embed_a) for unit in self.rel], -1)
        # embed_b = torch.cat([unit(embed_b) for unit in self.rel], -1)
        embed_b = torch.cat([embed_b for i in range(self.num_heads)], -1)
        #sim_score = self.cos(embed_a, embed_b)
        sim_score = batch_dot(embed_a, embed_b) / self.num_heads
        return sim_score



class RelNet(nn.Module):
    def __init__(self, d, dropout=0.0, num_heads=3):
        super(RelNet, self).__init__()
        self.num_heads = num_heads

        rel_models = []
        entf_entr = RelUnit(d, dropout, num_heads)
        for i in range(NUM_REL):
            if i in [2, 3]:
                rel_models.append(entf_entr)
            else:
                rel_models.append(RelUnit(d, dropout, num_heads))

        self.units = nn.ModuleList(rel_models)
        # self.eq = RelUnit(d)
        # self.ent_f = RelUnit(d)
        # self.ent_r = RelUnit(d)
        # self.neg = RelUnit(d)
        # self.alt = RelUnit(d)
        # self.cov = RelUnit(d)
        # self.ind = RelUnit(d)

    def forward(self, embed_as, embed_bs, label_ids=None, hypothesis_aligned=None):
        """
        :param embed_a: one row is a sample
        :param embed_b: one row is a sample
        :param label_ids: if label_id is not None, then output[0] will be the loss
        :return: tuple((loss), logits, probs)
                logits: similarity score of two embeddings
                probs: after softmax
        """
        # embed_a = avg_embed(sentence_a, mask_a).squeeze(1)
        # embed_b = avg_embed(sentence_b, mask_b).squeeze(1)
        logits = []
        for i, unit in enumerate(self.units):
            if i == 3:
                logits.append(unit(embed_bs, embed_as))
            else:
                logits.append(unit(embed_as, embed_bs))
        logits = torch.stack(logits).t()
        #if prior_logits is not None:
        #    logits += prior_logits
        # aligned: bs*n_word, 1
        # logits: bs*n_word, 7
        ones = torch.ones(logits.shape[0], 1, dtype=torch.float32).cuda()
        rel_mask = torch.cat([hypothesis_aligned, 1 - hypothesis_aligned, hypothesis_aligned.repeat(1, 2), ones, hypothesis_aligned, ones], dim=-1)
        # ind ,eq, entf, entr, neg, alt, cov
        #rel_mask = torch.cat([rel_mask, hypothesis_aligned, rel_mask.repeat(1, 5)], dim=-1)
        probs = nn.Softmax(dim=-1)(logits - 1e06 * rel_mask)  #

        output = (logits, probs)
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            prob_max = torch.max(probs, dim=1)[0]
            # ce loss
            loss_ce = loss_fct(logits.view(-1, len(self.units)), label_ids.view(-1))
            # margin loss
            loss_m = torch.max((1 - prob_max) / prob_max) / 6.0

            params = dict()
            for name, param in self.named_parameters():
                params[name] = param
            # symmetry loss [ind, eq, neg, alt, cov]
            loss_sym_list = torch.Tensor().to(probs.device)
            for unit_idx in ["0", "1", "4", "5", "6"]:
                name_list = ["units.{}.rel.{}.1.weight".format(unit_idx, head_id) for head_id in range(self.num_heads)]
                loss_sym_list = torch.cat([loss_sym_list,
                                           max([symmetry_loss(params[name]) for name in name_list]).unsqueeze(-1)], -1)
            loss_sym = torch.sum(loss_sym_list)
            # reverse loss [ent_r, ent_f]
            loss_reverse = max([reverse_loss(
                params["units.2.rel.{}.1.weight".format(head_id)],
                params["units.3.rel.{}.1.weight".format(head_id)])
                    for head_id in range(self.num_heads)]) * 2

            loss_all = loss_ce + loss_m + loss_sym + loss_reverse
            output = ((loss_all, loss_ce, loss_m, loss_sym, loss_reverse),) + output
            del params
        return output



if __name__ == "__main__":
    import numpy as np
    embed1 = torch.from_numpy(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype="float32"))
    embed2 = torch.from_numpy(np.array([[2.0, 2.0, 1.0], [2.0, 2.0, 1.0]], dtype="float32"))
    label = torch.from_numpy(np.array([[1], [0]], dtype="long"))
    relnet = RelNet(3)
    output = relnet(embed1, embed2, label)
    print("output", output)
