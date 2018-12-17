# PyTorch Imports
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DecoderRNN(nn.Module):
    def __init__(self, hidden_sz, vocab_sz, rnn_layers=1, device=torch.device('cpu')):
        """
        Decoder part of the seq2seq model.
        :param hidden_sz (int): The size of our RNN's hidden state
        :param vocab_sz (int): The size of the vocabulary we are decoding to (|V| where V is the COCO vocab)
        :param num_layers (int): The number of RNN cells we want in our net (c.f num_layers param in torch.nn.LSTMCell)
        """
        super(DecoderRNN, self).__init__()
        self.hidden_sz = hidden_sz
        self.vocab_sz = vocab_sz
        self.rnn_layers = rnn_layers

        self.embedding = nn.Embedding(self.vocab_sz, self.hidden_sz)
        self.rnn_decoder = nn.GRU(self.hidden_sz, self.hidden_sz, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_sz, self.vocab_sz)
        self.softmax = nn.LogSoftmax(dim=1)

        self._dev = device
        self.to(device)

    def forward_step(self, input_var, hidden):
        embedded = self.embedding(input_var)
        output, decoder_hidden = self.rnn(embedded, hidden)
        vocab_scores = self.fc(output)
        predicted_softmax = self.softmax(vocab_scores)

        return predicted_softmax, decoder_hidden

    def forward(self, tokens, seq_lens, features):
        """ The forward pass for our model.
            :param tokens (torch.tensor) batch_sz x vocab_sz - this is the target tensor
            :param seq_lens (int) Original sequence lengths of `tokens` before they were padded
            :param features (torch.tensor) final hidden state of the encoder RNN
        """
        curr_batch_size = seq_lens.shape[0] # hint: use this for reshaping RNN hidden state

        # 1. Embed the input target tensor to be of hidden_sz shape
        embeds = self.embedding(tokens)

        # Concatenate the visual/semantic features and the embeddings of the target tensor / output so far
        embeds = torch.cat((features, embeds), 1)

        # 2. Sort seq_lens and embeds in descending order of seq_lens. (check out torch.sort)
        #    This is expected by torch.nn.utils.pack_padded_sequence.
        sorted_seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        sorted_seq_tensor = embeds[perm_idx].squeeze(1)

        # 3. Obtain a PackedSequence object from pack_padded_sequence.
        #    Be sure to pass batch_first=True as the first dimension of our input is the batch dim.
        packed_input = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens.squeeze(1), batch_first=True)

        # 4. Apply the RNN over the sequence of packed embeddings to obtain a sentence encoding.
        output, hidden = self.rnn_decoder(packed_input, self.init_hidden(curr_batch_size))

        unpacked_output = pad_packed_sequence(output, batch_first=True)
        vocab_probs = self.softmax(self.fc(unpacked_output))

        # 6. Remember to unsort the output from step 5. If you sorted seq_lens and obtained a permutation
        #    over its indices (perm_ix), then the sorted indices over perm_ix will "unsort".
        #    For example:
        #       _, unperm_ix = perm_ix.sort(0)
        #       output = x[unperm_ix]
        #       return output
        _, unperm_idx = perm_idx.sort(0)
        unsorted_vocab_probs = vocab_probs.squeeze(0)[unperm_idx]
        return unsorted_vocab_probs

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_sz, device=self._dev)