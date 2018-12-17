import pdb
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

# Pytorch Imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
from tensorboardX import SummaryWriter

from tqdm import tqdm

# Stencil imports
import embeddings
from coco_dataset import COCODataset, get_vocab_path, get_data_set_file_paths
from vocab import Vocab
import Constants
import hyperparams

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
    def __init__(self, hidden_sz, embedding_lookup, rnn_layers=1, device=torch.device('cpu')):
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
        self.num_outputs = 2
        self.interm_size_1 = 256
        self.interm_size_2 = 64

        self.embedding_lookup = embedding_lookup.to(device) # instance of torch.nn.Module
        self.embed_size = embedding_lookup.embed_size # int value

        ## --- TODO: define the network architecture here ---
        ## Hint: you may wish to use nn.Sequential to stack linear layers, non-linear
        ##  activations, and dropout into one module
        ##
        ## Use GRU as your RNN architecture
        self.rnn_encoder = nn.GRU(self.embed_size, self.hidden_sz, num_layers=rnn_layers, batch_first=True)

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
        unsorted_hidden_states = hidden.squeeze(0)[unperm_idx]
        return unsorted_hidden_states

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_sz, device=self._dev)

def pre_process_image(image, device):
    image3 = image.to(device)

    # Some photos in the dataset are black and white - deal with those
    if image3.shape[1] == 1:
        pdb.set_trace()
        target_image = torch.zeros(3, 224, 224, device=device)
        target_image[0, :, :] = image3
        image3 = target_image
    return image3

def train(hp, embedding_lookup, device):
    """ This is the main training loop
            :param hp: HParams instance (see hyperparams.py)
            :param embedding_lookup: torch module for performing embedding lookups (see embeddings.py)
    """
    vocab = Vocab(filename=get_vocab_path(),
                  data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    data_set = COCODataset(*get_data_set_file_paths("val"), vocab)
    data_set_size = len(data_set)
    dataloader = DataLoader(data_set, batch_size=hp.batch_size, shuffle=True)

    caption_model = EncoderRNN(hp.rnn_hidden_size, embedding_lookup, device=device)
    image_model = EncoderCNN(hp.rnn_hidden_size, device)
    print(caption_model)
    print(image_model)

    criterion = nn.MSELoss()
    params = list(caption_model.parameters()) + list(image_model.linear.parameters()) + list(image_model.bn.parameters())
    optimizer = optim.Adam(params, lr=hp.learn_rate)

    training_loss = []
    writer = SummaryWriter()
    iteration = 0

    for epoch in range(1, hp.num_epochs + 1):
        running_loss = 0.
        for vectorized_seq, seq_len, image in tqdm(dataloader, desc='{}/{}'.format(epoch, hp.num_epochs)):
            vectorized_seq = vectorized_seq  # note: we don't pass this to GPU yet
            seq_len = seq_len.to(device)
            image = image.to(device)

            caption_model.train()
            caption_model.zero_grad()

            # Forward pass through our two encoder models
            caption_features = caption_model(vectorized_seq, seq_len).squeeze(1)
            image_features = image_model(image)

            loss = criterion(caption_features, image_features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("Loss", loss.item(), iteration)
            iteration = iteration + 1

        avg_loss = running_loss / (data_set_size / hp.batch_size)
        training_loss.append(avg_loss)
        print("Loss: {}".format(avg_loss))

        torch.save(image_model.state_dict(), "saved_runs/image_model_{i}_weights.pt".format(i=epoch))
        torch.save(caption_model.state_dict(), "saved_runs/caption_model_{i}_weights.pt".format(i=epoch))

    plt.figure()
    plt.plot(training_loss)
    plt.title("Training Loss")
    plt.show()

def main():
    device = torch.device("cuda")
    vocab_path = get_vocab_path()

    v = Vocab(filename=vocab_path,
              data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])

    lookup = embeddings.RandEmbed(v.size(), device=device)
    hp = hyperparams.RandEmbedHParams(embed_size=lookup.embed_size)

    train(hp, lookup, device)

if __name__ == "__main__":
    main()