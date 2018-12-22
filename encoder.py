from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models

def extract_feature_model(pretrained_vgg19):
    new_vgg = deepcopy(pretrained_vgg19)
    classifier_layers = list(list(pretrained_vgg19.children())[1].children())[:5]
    new_vgg.classifier = nn.Sequential(*classifier_layers)
    for p in new_vgg.parameters():
        p.requires_grad = False
    return new_vgg


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, device):
        """ Load the pre-trained VGG19 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.cnn = extract_feature_model(models.vgg19(pretrained=True))
        self.linear = nn.Linear(self.cnn.classifier[3].out_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self._dev = device
        self.to(device)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        return self.bn(self.linear(features))

class EncoderRNN(nn.Module):
    def __init__(self, hidden_sz, embedding_lookup, rnn_layers=2, device=torch.device('cpu')):
        """
        The constructor for our net. The architecture of this net is the following:

            Embedding lookup -> RNN Encoder -> Linear layer -> Linear Layer

        You should apply a non-linear activation after the first linear layer
        and finally a softmax for the logits (optional if using CrossEntropyLoss). If you observe overfitting,
        dropout can be inserted between the two linear layers. Note that looking up the embeddings are handled
        by the modules in embeddings.py which can then be called in this net's forward pass.

        :param hidden_sz (int): The size of our RNN's hidden state
        :param embedding_lookup (str): The type of word embedding used.
                                       Either 'glove', 'elmo', 'both', or 'random'.
        :param num_layers (int): The number of RNN cells we want in our net (c.f num_layers param in torch.nn.LSTMCell)
        """
        super(EncoderRNN, self).__init__()
        self.hidden_sz = hidden_sz
        self.rnn_layers = rnn_layers

        self.embedding_lookup = embedding_lookup.to(device) # instance of torch.nn.Module
        self.embed_size = embedding_lookup.embed_size # int value

        ## --- TODO: define the network architecture here ---
        ## Hint: you may wish to use nn.Sequential to stack linear layers, non-linear
        ##  activations, and dropout into one module
        ##
        ## Use GRU as your RNN architecture
        self.rnn_encoder = nn.GRU(self.embed_size, self.hidden_sz, num_layers=rnn_layers, batch_first=True, bidirectional=True)

        self._dev = device
        self.to(device)

    def forward(self, tokens, seq_lens):
        """ TODO The forward pass for our model.
                 :param tokens: vectorized sequence inputs as token ids.
                 :param seq_lens: original sequence lengths of each input (prior to padding)
            You should return a tensor of (batch_size x 2) logits for each class per batch element.
        """

        curr_batch_size = seq_lens.shape[0] # hint: use this for reshaping RNN hidden state

        # 1. Grab the embeddings:
        embeds = self.embedding_lookup(tokens) # the embeddings for the token sequence

        # 2. Sort seq_lens and embeds in descending order of seq_lens. (check out torch.sort)
        #    This is expected by torch.nn.utils.pack_padded_sequence.
        sorted_seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        sorted_seq_tensor = embeds[perm_idx].squeeze(1)

        # 3. Obtain a PackedSequence object from pack_padded_sequence.
        #    Be sure to pass batch_first=True as the first dimension of our input is the batch dim.
        packed_input = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens.squeeze(1), batch_first=True)

        # 4. Apply the RNN over the sequence of packed embeddings to obtain a sentence encoding.
        output, hidden = self.rnn_encoder(packed_input, self.init_hidden(curr_batch_size))

        # 6. Remember to unsort the output from step 5. If you sorted seq_lens and obtained a permutation
        #    over its indices (perm_ix), then the sorted indices over perm_ix will "unsort".
        #    For example:
        #       _, unperm_ix = perm_ix.sort(0)
        #       output = x[unperm_ix]
        #       return output
        _, unperm_idx = perm_idx.sort(0)
        unsorted_hidden_states = hidden[:, unperm_idx, :].squeeze(2)
        return unsorted_hidden_states

    def init_hidden(self, batch_size):
        return torch.zeros(2*self.rnn_layers, batch_size, self.hidden_sz, device=self._dev)
