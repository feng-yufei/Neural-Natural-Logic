""" Neural Natural Logic Structure """

import logging

from utils.torch.esim import *
from utils.torch.model import init_weights

MODEL_VERSION = "v1"

logger = logging.getLogger(__name__)


class ESIM_Aligner(nn.Module):
    def __init__(self, embed_model, hidden_size, padding_idx=0, dropout=0.5, num_classes=3):
        super(ESIM_Aligner, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        self._word_embedding = nn.Embedding(embed_model.shape[0],
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

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2 * 4 * self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(init_weights)

    def forward(self,
                premise_ids,
                premise_mask,
                hypothesis_ids,
                hypothesis_mask,
                label_ids=None):
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

        premise_lengths = torch.sum(premise_mask, dim=-1, dtype=torch.int64)
        max_premise = int(torch.max(premise_lengths).item())
        hypothesis_lengths = torch.sum(hypothesis_mask, dim=-1, dtype=torch.int64)
        max_hypothesis = int(torch.max(hypothesis_lengths).item())
        premise_ids = premise_ids[:, 0:max_premise]
        hypothesis_ids = hypothesis_ids[:, 0:max_hypothesis]
        premise_mask = premise_mask[:, 0:max_premise]
        hypothesis_mask = hypothesis_mask[:, 0:max_hypothesis]

        embedded_premise = self._word_embedding(premise_ids)
        embedded_hypothesis = self._word_embedding(hypothesis_ids)

        if self.dropout:
            embedded_premise = self._rnn_dropout(embedded_premise)
            embedded_hypothesis = self._rnn_dropout(embedded_hypothesis)

        encoded_premise, _ = self._encoding(embedded_premise,
                                          premise_lengths)
        encoded_hypothesis, _ = self._encoding(embedded_hypothesis,
                                            hypothesis_lengths)

        attended_premise, attended_hypothesis, attention_matrix = \
            self._attention(encoded_premise, premise_mask,
                            encoded_hypothesis, hypothesis_mask, attention_matrix=True)
        #print(premise_ids.shape)
        #print(hypothesis_ids.shape)
        #print(encoded_hypothesis.shape)
        #print(attention_matrix.shape)
        #exit()
        enhanced_premise = torch.cat([encoded_premise,
                                       attended_premise,
                                       encoded_premise - attended_premise,
                                       encoded_premise * attended_premise],
                                      dim=-1)
        enhanced_hypothesis = torch.cat([encoded_hypothesis,
                                         attended_hypothesis,
                                         encoded_hypothesis -
                                         attended_hypothesis,
                                         encoded_hypothesis *
                                         attended_hypothesis],
                                        dim=-1)

        projected_premise = self._projection(enhanced_premise)
        projected_hypothesis = self._projection(enhanced_hypothesis)

        if self.dropout:
            projected_premise = self._rnn_dropout(projected_premise)
            projected_hypothesis = self._rnn_dropout(projected_hypothesis)

        v_ai, _ = self._composition(projected_premise, premise_lengths)
        v_bj, _ = self._composition(projected_hypothesis, hypothesis_lengths)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(premise_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypothesis_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(hypothesis_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premise_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypothesis_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        output = (logits, probabilities)
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids.view(-1))
            output = (loss,) + output
        return output, attention_matrix


