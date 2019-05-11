from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy


MAX_LENGTH = 400

class EncoderRNN(nn.Module):
    def __init__(self, embedding, input_size, embed_size, hidden_size, device='cuda'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(input_size, embed_size)
        self.embedding = embedding
        self.LSTM = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        # self.output_cproj = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        # self.output_hproj = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self._initialize_bridge('LSTM',
                                hidden_size,
                                1)
        self.device = device

    @staticmethod
    def _fix_enc_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
        return hidden

    def forward(self, input, input_length, hidden=None):
        batch_size, length = input.size()
        embedded = self.embedding(input)
        output = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
        #output, hidden = self.gru(output, hidden)
        output, hidden = self.LSTM(output, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # final_hn = F.relu(self.output_hproj(hidden[0].transpose(0, 1).contiguous().view(1, batch_size, -1)))
        # final_cn = F.relu(self.output_cproj(hidden[1].transpose(0, 1).contiguous().view(1, batch_size, -1)))
        (final_hn, final_cn) = self._bridge(hidden)
        final_hn = self._fix_enc_hidden(final_hn)
        final_cn = self._fix_enc_hidden(final_cn)
        #outputs = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return output, (final_hn, final_cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])
    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs




class concatAttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, attn_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, device = 'cuda'):
        super(concatAttnDecoderRNN, self).__init__()
        self.device = device
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attn_size = attn_size

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        #self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size)

        self.attnm = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.attnq = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.alignment = nn.Linear(self.attn_size, 1, bias=False)
        #self.alignment = nn.Parameter(torch.FloatTensor(self.attn_size, 1))

        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size + self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        input = input.view(1, -1)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        logging.debug(embedded.size())
        logging.debug(hidden.size())

        output, hidden_new = self.gru(embedded, hidden)

        #query = self.attnq(output)
        query = self.attnq(hidden)
        keys = self.attnm(encoder_outputs)


        alignment = self.alignment(torch.tanh(query + keys))
        attn_weights = F.softmax(alignment, dim=0)
        attn_applied = torch.bmm(attn_weights.permute(1,2,0),
                                 encoder_outputs.permute(1,0,2))
        output = torch.cat((output, attn_applied.squeeze(1).unsqueeze(0)), 2)
        output = self.out(output)

        return output, hidden_new, attn_weights.squeeze(2)

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, embedding, embed_size, hidden_size, attn_size, output_size, dropout_p=0., max_length=MAX_LENGTH, device = 'cuda'):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.device = device
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attn_size = attn_size

        #self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedding = embedding
        #self.gru = nn.GRU(self.embedding_size + self.hidden_size, self.hidden_size)
        self.LSTM = nn.LSTM(self.embedding_size + self.hidden_size, self.hidden_size)

        self.attnm = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.attnq = nn.Linear(self.hidden_size, self.attn_size, bias=True)
        self.alignment = nn.Linear(self.attn_size, 1, bias=False)
        #self.alignment = nn.Parameter(torch.FloatTensor(self.attn_size, 1))
        self.linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, input, hidden, encoder_outputs, input_feed):
        input = input.view(1, -1)
        batch_size = input.size(1)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # print(embedded.size())
        # print(input_feed.size())
        embedded = torch.cat((embedded, input_feed), -1)
        output, hidden = self.LSTM(embedded, hidden)
        #_h = output
        #query = self.attnq(output)
        query = self.attnq(output)
        keys = self.attnm(encoder_outputs)


        alignment = self.alignment(torch.tanh(query + keys))
        attn_weights = F.softmax(alignment, dim=0)
        attn_applied = torch.bmm(attn_weights.permute(1,2,0),
                                 encoder_outputs.permute(1,0,2)).view(-1, batch_size, self.hidden_size)
        output = torch.cat((output, attn_applied), dim=2)
        output = self.linear_out(output)
        input_feed = output


        output = self.out(output)

        return output, hidden, attn_weights.squeeze(2), input_feed

class EncoderRNN_withttopic(nn.Module):
    def __init__(self, embedding, input_size, embed_size, hidden_size, device='cuda', topic_size=11, mode='stack'):
        super(EncoderRNN_withttopic, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(input_size, embed_size)
        self.embedding = embedding
        #self.gru = nn.GRU(embed_size, hidden_size)
        if mode == 'stack':
            self.LSTM = nn.LSTM(embed_size + topic_size, hidden_size, bidirectional=True)
        elif mode == 'mlp':
            self.Linear = nn.Linear(embed_size + topic_size, embed_size)
            # self.Linear_embed = nn.Linear(embed_size, embed_size)
            # self.Linear_topic = nn.Linear(topic_size, embed_size, bias=False)
            self.LSTM = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        # self.output_cproj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.output_hproj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self._initialize_bridge('LSTM',
                                hidden_size,
                                1)
        self.device = device
        self.topic_size = topic_size
        self.mode = mode

    @staticmethod
    def _fix_enc_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
        return hidden

    def forward(self, input, input_length, topic_seqs, hidden=None):
        batch_size, length = input.size()
        embedded = self.embedding(input)

        topic_expanded = topic_seqs.unsqueeze(1).expand(-1, length, -1)
        if self.mode == 'stack':
            embedded = torch.cat((embedded, topic_expanded), dim=2)
            output = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
            output, hidden = self.LSTM(output, hidden)
        elif self.mode == 'mlp':
            embedded = torch.cat((embedded, topic_expanded), dim=2)
            embedded = self.Linear(embedded)
            # embedded = torch.tanh(self.Linear_embed(embedded) + self.Linear_topic(embedded))
            output = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
            output, hidden = self.LSTM(output, hidden)
        #output, hidden = self.gru(output, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        (final_hn, final_cn) = self._bridge(hidden)
        final_hn = self._fix_enc_hidden(final_hn)
        final_cn = self._fix_enc_hidden(final_cn)

        # final_hn = self.output_hproj(hidden[0].transpose(0, 1).contiguous().view(1, batch_size, -1))
        # final_cn = self.output_cproj(hidden[1].transpose(0, 1).contiguous().view(1, batch_size, -1))
        #outputs = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return output, (final_hn, final_cn)

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])
    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

class BahdanauAttnDecoderRNN_withtopic(nn.Module):
    def __init__(self, embedding, embed_size, hidden_size, attn_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, device = 'cuda', topic_size=11, topicaware=False,
                 mode = 'mlp'):
        super(BahdanauAttnDecoderRNN_withtopic, self).__init__()
        self.device = device
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attn_size = attn_size
        self.topic_size = topic_size

        #self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedding = embedding
        #self.gru = nn.GRU(self.embedding_size + self.hidden_size, self.hidden_size)
        if topicaware:
            if mode == 'stack':
                self.LSTM = nn.LSTM(self.embedding_size + self.topic_size + self.hidden_size, self.hidden_size)
            elif mode == 'mlp':
                self.Linear_dec = nn.Linear(self.embedding_size + self.topic_size, self.embedding_size)
                self.LSTM = nn.LSTM(self.embedding_size + self.hidden_size, self.hidden_size)
        else:
            self.LSTM = nn.LSTM(self.embedding_size + self.hidden_size, self.hidden_size)

        self.attnm = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.attnq = nn.Linear(self.hidden_size, self.attn_size, bias=True)
        self.alignment = nn.Linear(self.attn_size, 1, bias=False)
        #self.alignment = nn.Parameter(torch.FloatTensor(self.attn_size, 1))
        self.linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)


        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.topicaware = topicaware
        self.mode = mode

    def forward(self, input, topic_seqs, hidden, encoder_outputs, input_feed):
        input = input.view(1, -1)
        batch_size = input.size(1)
        embedded = self.embedding(input)
        #embedded = self.dropout(embedded)
        topic_expanded = topic_seqs.unsqueeze(0)
        if self.topicaware:
            if self.mode == 'stack':
                embedded = torch.cat((embedded, topic_expanded, input_feed), dim=2)
            elif self.mode == 'mlp':
                embedded = self.Linear_dec(torch.cat((embedded, topic_expanded), dim=2))
                embedded = torch.cat((embedded, input_feed), dim=2)
        else:
            embedded = torch.cat((embedded, input_feed), dim=-1)
        output, hidden = self.LSTM(embedded, hidden)

        #query = self.attnq(output)
        query = self.attnq(output)
        keys = self.attnm(encoder_outputs)


        alignment = self.alignment(torch.tanh(query + keys))
        attn_weights = F.softmax(alignment, dim=0)
        attn_applied = torch.bmm(attn_weights.permute(1,2,0),
                                 encoder_outputs.permute(1,0,2)).view(-1, batch_size, self.hidden_size)
        output = torch.cat((output, attn_applied), dim=-1)
        output = self.linear_out(output)
        input_feed = output
        output = self.out(output)

        return output, hidden, attn_weights.squeeze(2), input_feed


class query_Encoder(nn.Module):
    def __init__(self, embedding, input_size, embed_size, hidden_size, device='cuda'):
        super(query_Encoder, self).__init__()
        self.query_embedding = nn.LSTM()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.LSTM = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self._initialize_bridge('LSTM',
                                hidden_size,
                                1)
        self.device = device

    @staticmethod
    def _fix_enc_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
        return hidden

    def forward(self, input, input_length, hidden=None):
        batch_size, length = input.size()
        embedded = self.embedding(input)
        output = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)
        #output, hidden = self.gru(output, hidden)
        output, hidden = self.LSTM(output, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # final_hn = F.relu(self.output_hproj(hidden[0].transpose(0, 1).contiguous().view(1, batch_size, -1)))
        # final_cn = F.relu(self.output_cproj(hidden[1].transpose(0, 1).contiguous().view(1, batch_size, -1)))
        (final_hn, final_cn) = self._bridge(hidden)
        final_hn = self._fix_enc_hidden(final_hn)
        final_cn = self._fix_enc_hidden(final_cn)
        #outputs = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return output, (final_hn, final_cn)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])
    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs