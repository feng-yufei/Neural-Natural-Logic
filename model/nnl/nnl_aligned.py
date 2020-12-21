""" Neural Natural Logic Structure """

import logging

from utils.torch.esim import *
from utils.torch.model import init_weights
from utils.torch.compute import get_last_masked_tensor
from utils.torch.loss import symmetry_loss, reverse_loss

from .relnn_aligned import RelNet
#from .relnn import RelNet

from datautils.datareaders.basic_rel import LABEL_DICT as REL_LABEL_DICT
MODEL_VERSION = "v6"

logger = logging.getLogger(__name__)

# From datareaders.LABEL_DICT
# {'ind': 0, 'eq': 1, 'ent_f': 2, 'ent_r': 3, 'neg': 4, 'alt': 5, 'cov': 6}

UNION = {
    ('eq', 'eq'): 'eq', ('eq', 'ent_f'): 'ent_f', ('eq', 'ent_r'): 'ent_r', ('eq', 'neg'): 'neg',
    ('eq', 'alt'): 'alt', ('eq', 'cov'): 'cov', ('eq', 'ind'): 'ind',
    ('ent_f', 'eq'): 'ent_f', ('ent_f', 'ent_f'): 'ent_f', ('ent_f', 'ent_r'): 'ind', ('ent_f', 'neg'): 'alt',
    ('ent_f', 'alt'): 'alt', ('ent_f', 'cov'): 'ind', ('ent_f',  'ind'): 'ind',
    ('ent_r', 'eq'): 'ent_r', ('ent_r', 'ent_f'): 'ind', ('ent_r', 'ent_r'): 'ent_r', ('ent_r', 'neg'): 'cov',
    ('ent_r', 'alt'): 'ind', ('ent_r', 'cov'): 'cov', ('ent_r', 'ind'): 'ind',
    ('neg', 'eq'): 'neg', ('neg', 'ent_f'): 'cov', ('neg', 'ent_r'): 'alt', ('neg', 'neg'): 'eq',
    ('neg', 'alt'): 'ent_r', ('neg', 'cov'): 'ent_f', ('neg', 'ind'): 'ind',
    ('alt', 'eq'): 'alt', ('alt', 'ent_f'): 'ind', ('alt', 'ent_r'): 'alt', ('alt', 'neg'): 'ent_f',
    ('alt', 'alt'): 'ind', ('alt', 'cov'): 'ent_f', ('alt', 'ind'): 'ind',
    ('cov', 'eq'): 'cov', ('cov', 'ent_f'): 'cov', ('cov', 'ent_r'): 'ind', ('cov', 'neg'): 'ent_r',
    ('cov', 'alt'): 'ent_r', ('cov', 'cov'): 'ind', ('cov', 'ind'): 'ind',
    ('ind', 'eq'): 'ind', ('ind', 'ent_f'): 'ind', ('ind', 'ent_r'): 'ind', ('ind', 'neg'): 'ind',
    ('ind', 'alt'): 'ind', ('ind', 'cov'): 'ind', ('ind', 'ind'): 'ind',
}

UNION_MATRIX = torch.zeros((len(REL_LABEL_DICT), len(REL_LABEL_DICT)))
for k, v in UNION.items():
    UNION_MATRIX[REL_LABEL_DICT[k[0]]][REL_LABEL_DICT[k[1]]] = REL_LABEL_DICT[v]
UNION_MATRIX = UNION_MATRIX.int()

UNION_ADD_DICT = dict()
UNION_SORT, UNION_IDX = UNION_MATRIX.view(-1).sort()
for l in REL_LABEL_DICT.values():
    UNION_ADD_DICT[l] = UNION_IDX[UNION_SORT == l]


class NNL_Aligned(nn.Module):
    def __init__(self, embed_model, hidden_size, padding_idx=0, dropout=0.5, dropout_relnet=0.5, num_classes=3, num_heads=3):
        super(NNL_Aligned, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.dropout_relnet = dropout_relnet
        self.num_heads = num_heads
        self.word_embed = nn.Embedding(embed_model.shape[0],
                                       embed_model.shape[1],
                                       padding_idx=padding_idx,
                                       _weight=embed_model)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        embed_model.shape[1],
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = BasicAttention()
        self.relnet = RelNet(2*hidden_size, self.dropout_relnet, num_heads)
        # self.relnet_2 = RelNet(2 * hidden_size, self.dropout_relnet, num_heads)
        # self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
        #                                      nn.Linear(len(REL_LABEL_DICT.keys()),
        #                                                self.hidden_size),
        #                                      nn.Tanh(),
        #                                      nn.Dropout(p=self.dropout),
        #                                      nn.Linear(self.hidden_size, self.num_classes))
        self._classification = nn.Sequential(nn.Linear(len(REL_LABEL_DICT.keys()),
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size, self.num_classes))
        self.loss_fct = nn.CrossEntropyLoss()
        # self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
        #                                      nn.Linear(len(REL_LABEL_DICT.keys()),
        #                                                self.hidden_size),
        #                                      nn.Tanh(),
        #                                      nn.Dropout(p=self.dropout))
        # self._last = nn.Linear(self.hidden_size, self.num_classes)
        # self._last = ArcMarginProduct(self.hidden_size, self.num_classes)


        # Initialize all weights and biases in the model.
        self.apply(init_weights)


        # self._margin_last = ArcMarginProduct(self.hidden_size, self.num_classes)

    def forward(self,
                premise_ids,
                premise_masks,
                premise_project,
                premise_aligned,
                hypothesis_ids,
                hypothesis_masks,
                hypothesis_project,
                hypothesis_aligned,
                label_ids=None,
                method="sum",
                basic_rel=False,
                analyze=False):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """

        premise_lengths = torch.sum(premise_masks, dim=-1, dtype=torch.int64)
        max_premise = int(torch.max(premise_lengths).item())
        hypothesis_lengths = torch.sum(hypothesis_masks, dim=-1, dtype=torch.int64)
        max_hypothesis = int(torch.max(hypothesis_lengths).item())
        premise_ids = premise_ids[:, 0:max_premise]
        hypothesis_ids = hypothesis_ids[:, 0:max_hypothesis]

        premise_masks = premise_masks[:, 0:max_premise]
        premise_aligned = premise_aligned[:, 0:max_premise]
        #premise_masks = torch.clamp(premise_masks - premise_aligned, min=0, max=1)
        hypothesis_masks = hypothesis_masks[:, 0:max_hypothesis]
        hypothesis_aligned = hypothesis_aligned[:, 0:max_hypothesis]
        #hypothesis_masks = torch.clamp(hypothesis_masks - hypothesis_aligned, min=0, max=1)

        # embedded_premise = self.word_embed(premise_ids)
        # embedded_hypothesis = self.word_embed(hypothesis_ids)
        premise_project = premise_project[:, 0:max_premise, :]
        hypothesis_project = hypothesis_project[:, 0:max_hypothesis, :]

        assert method in ["sum", "max"]
        embedded_premise = self.word_embed(premise_ids)
        embedded_hypothesis = self.word_embed(hypothesis_ids)

        if self.dropout:
            embedded_premise = self._rnn_dropout(embedded_premise)
            embedded_hypothesis = self._rnn_dropout(embedded_hypothesis)

        encoded_premise, _ = self._encoding(embedded_premise,
                                          premise_lengths)
        encoded_hypothesis, _ = self._encoding(embedded_hypothesis,
                                            hypothesis_lengths)

        if not basic_rel:
            # Normal NLI dataset
            premise_attention, hypothesis_attention = \
                self._attention(encoded_premise, premise_masks,
                                encoded_hypothesis, hypothesis_masks)

            premise_aligned = weighted_sum(encoded_premise, hypothesis_attention, hypothesis_masks)
            # hypothesis_aligned = weighted_sum(encoded_hypothesis, premise_attention, premise_masks)

            br_res_1 = self.relnet(premise_aligned.view(-1, encoded_hypothesis.shape[-1]),
                                    encoded_hypothesis.view(-1, encoded_hypothesis.shape[-1]),
                                    hypothesis_aligned=hypothesis_aligned.reshape(-1, 1))
            #print(hypothesis_aligned[0])
            #print(br_res_1[-1].shape)
            #print(br_res_1[-1].view((premise_aligned.shape[0], premise_aligned.shape[1], -1))[0])
            #exit(0)

            # br_res_2 = self.relnet_2(encoded_premise.view(-1, encoded_premise.shape[-1]),
            #                       hypothesis_aligned.view(-1, encoded_premise.shape[-1]))

            hypothesis_project_1 = hypothesis_project[:, 0:max(hypothesis_lengths), :].contiguous().view(-1, hypothesis_project.shape[-1])
            projected_rel_1 = self.project_prob(br_res_1[-1], hypothesis_project_1)
            projected_rel_1 = projected_rel_1.contiguous().view((premise_aligned.shape[0], premise_aligned.shape[1], -1))

            # premise_project_2 = premise_project[:, 0:max(premise_lengths), :].contiguous().view(-1,
            #                                                                                    premise_project.shape[-1])
            # projected_rel_2 = self.project_prob(br_res_2[-1], premise_project_2)
            # projected_rel_2 = projected_rel_2.contiguous().view(
            #    (hypothesis_aligned.shape[0], hypothesis_aligned.shape[1], -1))


            if method == "sum":
                union_1 = self.compute_union_matrix_sum(projected_rel_1)
                # union_2 = self.compute_union_matrix_sum(projected_rel_2)
            elif method == "max":
                union_1 = self.compute_union_matrix_max(projected_rel_1)
                # union_2 = self.compute_union_matrix_max(projected_rel_2)

            last_union_1 = get_last_masked_tensor(union_1, hypothesis_masks)
            # last_union_2 = get_last_masked_tensor(union_2, premise_masks)

            if method == "sum":
                # last_union = torch.cat([last_union_1, last_union_2], -1)
                last_union = last_union_1
                # last_union = (last_union_1 + last_union_2) / 2.0
                # logits = self._classification(last_union)
                # tmp = self._classification(last_union)
            if method == "max":
                mask1 = (torch.zeros(last_union_1.shape).to(last_union_1.device)
                                              .scatter_(-1, torch.max(last_union_1, dim=-1, keepdim=True)[1], 1))
                # mask2 = (torch.zeros(last_union_2.shape).to(last_union_2.device)
                #         .scatter_(-1, torch.max(last_union_2, dim=-1, keepdim=True)[1], 1))
                # last_union = torch.cat([mask1 * last_union_1, mask2 * last_union_2], -1)
                last_union = mask1 * last_union_1
                #last_union = (mask1 * last_union_1 + mask2 * last_union_2) / 2.0

                # logits = self._classification(last_union)
                # tmp = self._classification(mask * last_union)
            # logits = self._last(tmp)

            # last_union : batch_size, 7
            #logits = self._classification(last_union)
            logit_entail = torch.max(last_union[:, [1, 2]], dim=-1, keepdim=True)[0]
            logit_contradiction = torch.max(last_union[:, [4, 5]], dim=-1, keepdim=True)[0]
            logit_neutral = torch.max(last_union[:, [0, 3, 6]], dim=-1, keepdim=True)[0]
            logits = torch.cat([logit_entail, logit_contradiction, logit_neutral], dim=-1)

            probabilities = logits

            if analyze:
                analyze_dict = dict()
                analyze_dict["hypothesis_attention"] = hypothesis_attention.detach().cpu()
                analyze_dict["premise_attention"] = premise_attention.detach().cpu()
                analyze_dict["br_rel_1"] = br_res_1[-1].detach().cpu()
                # analyze_dict["br_rel_2"] = br_res_2[-1].detach().cpu()
                analyze_dict["union_1"] = union_1.detach().cpu()
                # analyze_dict["union_2"] = union_2.detach().cpu()
                output = (analyze_dict, logits, probabilities)
            else:
                output = (logits, probabilities)
            if label_ids is not None:

                # ce loss
                loss_ce = self.loss_fct(logits, label_ids.view(-1))
                # margin loss
                rel_net_prob_max_1 = torch.max(br_res_1[-1], dim=1)[0]
                # rel_net_prob_max_2 = torch.max(br_res_2[-1], dim=1)[0]
                # loss_relnet_margin = torch.max((1 - rel_net_prob_max_1) / rel_net_prob_max_1)/6.0 + \
                #                     torch.max((1 - rel_net_prob_max_2) / rel_net_prob_max_2)/6.0
                loss_relnet_margin = torch.max((1 - rel_net_prob_max_1) / rel_net_prob_max_1)/6.0

                params = dict()
                for name, param in self.named_parameters():
                    if "relnet" in name:
                        params[name] = param
                # symmetry loss [ind, eq, neg, alt, cov]
                loss_sym_list1 = torch.Tensor().to(logits.device)
                for unit_idx in ["0", "1", "4", "5", "6"]:
                    name_list = ["relnet.units.{}.rel.{}.1.weight".format(unit_idx, head_id) for head_id in
                                 range(self.num_heads)]
                    loss_sym_list1 = torch.cat([loss_sym_list1,
                                               max([symmetry_loss(params[name]) for name in name_list]).unsqueeze(-1)],
                                              -1)
                loss_sym1 = torch.mean(loss_sym_list1)
                # loss_sym_list2 = torch.Tensor().to(logits.device)
                # for unit_idx in ["0", "1", "4", "5", "6"]:
                #     name_list = ["relnet_2.units.{}.rel.{}.1.weight".format(unit_idx, head_id) for head_id in
                #                  range(self.num_heads)]
                #     loss_sym_list2 = torch.cat([loss_sym_list2,
                #                                 max([symmetry_loss(params[name]) for name in name_list]).unsqueeze(-1)],
                #                                -1)
                # loss_sym2 = torch.mean(loss_sym_list2)
                # loss_sym = loss_sym1 + loss_sym2
                loss_sym = loss_sym1
                # reverse loss [ent_r, ent_f]
                loss_reverse1 = max([reverse_loss(
                    params["relnet.units.2.rel.{}.1.weight".format(head_id)],
                    params["relnet.units.3.rel.{}.1.weight".format(head_id)])
                    for head_id in range(self.num_heads)])
                # loss_reverse2 = max([reverse_loss(
                #     params["relnet_2.units.2.rel.{}.1.weight".format(head_id)],
                #     params["relnet_2.units.3.rel.{}.1.weight".format(head_id)])
                #     for head_id in range(self.num_heads)])
                # loss_reverse = loss_reverse1 + loss_reverse2
                loss_reverse = loss_reverse1
                # inconsistancy with relnet result
                # loss_relclass = self.loss_relclass(last_union, probabilities)
                loss_relclass = 0
                loss_all = 3 * loss_ce  # + loss_relnet_margin + loss_sym + loss_reverse + loss_relclass
                #output = ((loss_all, loss_ce, loss_relnet_margin, loss_sym, loss_reverse, loss_relclass),) + output
                output = (loss_all,) + output
                del params
            return output

        else:
            # Basic Rel dataset for auxiliary training
            br_res = self.relnet(torch.mean(encoded_premise, dim=1), torch.mean(encoded_hypothesis, dim=1),
                                 label_ids=label_ids)
            return br_res


    def project_prob(self, prob, project):
        """
        prob: batch_size * class_num
        """

        device = prob.device
        batch_size = prob.shape[0]
        # output = torch.zeros(prob.shape).to(device)
        #
        # for i, vec in enumerate(project):
        #     for j, idx in enumerate(vec):
        #         output[i][idx] = output[i][idx] + prob[i][j]
        output = torch.Tensor().to(device)
        for l in range(project.shape[-1]):
            output = torch.cat([output, torch.sum((project == l).int() * prob, dim=-1)], -1)
        output = output.view(-1, batch_size).transpose(0, 1)
        return output


    def compute_union_matrix_sum(self, state_batch):

        device = state_batch.device
        class_dim = state_batch.shape[-1]
        batch_size = state_batch.shape[0]
        # for i, batch in enumerate(state_batch):
        #     for j, state in enumerate(batch):
        #         if j == 0:
        #             output[i][j] = state
        #         else:
        #             for k, s in enumerate(state):
        #                 for kk, ss in enumerate(output[i][j-1]):
        #                     output[i][j][UNION_MATRIX[kk][k]] = output[i][j][UNION_MATRIX[kk][k]].clone() + ss * s
        output = state_batch[:, 0, :].unsqueeze(1)
        for j in range(1, state_batch.shape[1]):
            last_output = output[:, j-1, :].unsqueeze(-1).expand(batch_size, class_dim, class_dim)
            this_state = state_batch[:, j, :].unsqueeze(1).expand(batch_size, class_dim, class_dim)
            output_times = (last_output * this_state).view(batch_size, -1)
            this_output = torch.Tensor().to(device)
            for l in range(class_dim):
                this_output = torch.cat([this_output, torch.sum(output_times[:, UNION_ADD_DICT[l]], dim=-1)], -1)
            this_output = this_output.view(-1, batch_size).transpose(0, 1).unsqueeze(1)
            output = torch.cat([output, this_output], dim=1)
        return output


    # def compute_union_matrix_max(self, state_batch):
    #
    #     device = state_batch.device
    #     class_dim = state_batch.shape[-1]
    #     batch_size = state_batch.shape[0]
    #
    #     this_state = state_batch[:, 0, :]
    #     index = this_state.argmax(-1).unsqueeze(-1)
    #     mask = torch.zeros((batch_size, class_dim)).to(device).scatter_(1, index, 1)
    #     output = (this_state * mask).unsqueeze(1)
    #     for j in range(1, state_batch.shape[1]):
    #         last_output = output[:, j-1, :].unsqueeze(-1)
    #         max_last, last_index = last_output.max(dim=-1)
    #         mask_last = torch.zeros((batch_size, class_dim)).to(device).scatter_(1, last_index, 1).unsqueeze(-1)
    #         # last_output = last_output.expand(batch_size, class_dim, class_dim)
    #         this_state = state_batch[:, j, :].unsqueeze(1)
    #         max_state, this_index = this_state.max(dim=-1)
    #         mask_this = torch.zeros((batch_size, class_dim)).to(device).scatter_(1, this_index, 1).unsqueeze(1)
    #
    #         # this_state = this_state.expand(batch_size, class_dim, class_dim)
    #         output_times = ((last_output * mask_last).expand(batch_size, class_dim, class_dim) *
    #                         (this_state * mask_this).expand(batch_size, class_dim, class_dim)).view(batch_size, -1)
    #         this_output = torch.Tensor().to(device)
    #         for l in range(class_dim):
    #             this_output = torch.cat([this_output, torch.sum(output_times[:, UNION_ADD_DICT[l]], dim=-1)], -1)
    #         this_output = this_output.view(-1, batch_size).transpose(0, 1).unsqueeze(1)
    #         output = torch.cat([output, this_output], dim=1)
    #     return output

    def compute_union_matrix_max(self, state_batch):

        device = state_batch.device
        class_dim = state_batch.shape[-1]
        batch_size = state_batch.shape[0]
        # for i, batch in enumerate(state_batch):
        #     for j, state in enumerate(batch):
        #         if j == 0:
        #             output[i][j] = state
        #         else:
        #             for k, s in enumerate(state):
        #                 for kk, ss in enumerate(output[i][j-1]):
        #                     output[i][j][UNION_MATRIX[kk][k]] = output[i][j][UNION_MATRIX[kk][k]].clone() + ss * s
        output = state_batch[:, 0, :].unsqueeze(1)
        for j in range(1, state_batch.shape[1]):
            last_output = output[:, j - 1, :].unsqueeze(-1).expand(batch_size, class_dim, class_dim)
            this_state = state_batch[:, j, :].unsqueeze(1).expand(batch_size, class_dim, class_dim)
            output_times = (last_output * this_state).view(batch_size, -1)
            this_output = torch.Tensor().to(device)
            for l in range(class_dim):
                this_output = torch.cat([this_output, torch.max(output_times[:, UNION_ADD_DICT[l]], dim=-1)[0]], -1)
            this_output = this_output.view(-1, batch_size).transpose(0, 1).unsqueeze(1)
            output = torch.cat([output, this_output], dim=1)
        return output

    # def loss_relclass(self, last_union, prob):
    #     mask = (torch.zeros(last_union.shape).to(last_union.device)
    #              .scatter_(-1, torch.max(last_union, dim=-1, keepdim=True)[1], 1))
    #     rel_prob_max = mask * last_union
    #     # ent
    #     #loss_eq = - torch.sum(rel_prob_max[:, 1] * torch.log(prob[:, 0]))
    #     #loss_ent_f = - torch.sum(rel_prob_max[:, 2] * torch.log(prob[:, 0]))
    #     loss_eq = - torch.sum(rel_prob_max[:, 1] * torch.log(max(rel_prob_max[:, 1] - prob[:, 0], 0)))
    #     # neu
    #     #loss_ent_r = - torch.sum(rel_prob_max[:, 3] * torch.log(prob[:, 1]))
    #     # con
    #     #loss_neg = - torch.sum(rel_prob_max[:, 4] * torch.log(prob[:, 2]))
    #     return (loss_eq + loss_ent_f + loss_ent_r + loss_neg) #/ last_union.shape[0]




class BasicAttention(nn.Module):
    def forward(self, batch_a, mask_a, batch_b, mask_b):
        """
        batch_a, batch_b: batch_size * max_length * dim
        mask_a, mask_b: batch_size * max_length
        """
        similarity_matrix = torch.bmm(batch_a, batch_b.transpose(-1, -2).contiguous())

        # Softmax attention weights.
        a_attention = masked_softmax(similarity_matrix, mask_b)
        b_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), mask_a)

        return a_attention, b_attention

