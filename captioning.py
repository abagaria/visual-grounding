import pdb
import pickle

import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# PyTorch Imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Stencil imports
import embeddings
from coco_dataset import COCODataset, get_vocab_path, get_data_set_file_paths
from vocab import Vocab
import Constants
import hyperparams
from lookup_table import construct_semantic_lookup_table
from encoder import EncoderCNN, EncoderRNN

def train(hp, embedding_lookup, dataloader, device):
    """ This is the main training loop
            :param hp: HParams instance (see hyperparams.py)
            :param embedding_lookup: torch module for performing embedding lookups (see embeddings.py)
            :param dataloader: ms-coco dataloader
            :param device: CPU / Cuda
    """

    caption_model = EncoderRNN(hp.rnn_hidden_size, embedding_lookup, device=device)
    image_model = EncoderCNN(hp.rnn_hidden_size, device)
    print(caption_model)
    print(image_model)

    # criterion = n.CosineEmbeddingLoss(margin=0.1)
    criterion = nn.MSELoss()
    params = list(caption_model.parameters()) + list(image_model.linear.parameters()) + list(image_model.bn.parameters())
    optimizer = optim.Adam(params, lr=hp.learn_rate)

    training_loss = []
    writer = SummaryWriter("seg_deep_encoder_mse_loss")
    iteration = 0

    for epoch in range(1, hp.num_epochs + 1):
        running_loss = 0.
        for vectorized_seq, seq_len, image in tqdm(dataloader, desc='{}/{}'.format(epoch, hp.num_epochs)):
            vectorized_seq = vectorized_seq  # note: we don't pass this to GPU yet
            seq_len = seq_len.to(device)
            image = image.to(device)
            # related = related.to(device)

            caption_model.train()
            image_model.train()
            caption_model.zero_grad()
            image_model.zero_grad()

            # Forward pass through our two encoder models
            caption_features = caption_model(vectorized_seq, seq_len).squeeze(1)
            image_features = image_model(image)

            loss = criterion(caption_features, image_features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("Loss", loss.item(), iteration)
            iteration = iteration + 1

        avg_loss = running_loss / (len(dataloader) / hp.batch_size)
        training_loss.append(avg_loss)
        print("Loss: {}".format(avg_loss))

        torch.save(image_model.state_dict(), "saved_runs/seg_deep_image_model_{i}_weights.pt".format(i=epoch))
        torch.save(caption_model.state_dict(), "saved_runs/seg_deep_caption_model_{i}_weights.pt".format(i=epoch))

    plt.figure()
    plt.plot(training_loss)
    plt.title("Training Loss")
    plt.savefig("visual_deep_mse_training_loss.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_image", help="filepath to restore")
    parser.add_argument("--restore_caption", help="filepath to restore")
    args = parser.parse_args()

    device = torch.device("cuda")
    vocab_path = get_vocab_path()
    v = Vocab(filename=vocab_path, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    # lookup = embeddings.RandEmbed(v.size(), device=device)
    lookup = embeddings.VisualEmbed("segmentation_word_embeddings.npy", v, device)
    hp = hyperparams.RandEmbedHParams(embed_size=500)

    data_set = COCODataset(*get_data_set_file_paths("val"), v)
    data_set_size = len(data_set)
    print("Loading dataset of size {}, batch_size = {}".format(data_set_size, hp.batch_size))
    dataloader = DataLoader(data_set, batch_size=hp.batch_size, shuffle=True)

    if args.restore_image and args.restore_caption:
        rnn_encoder = EncoderRNN(hp.rnn_hidden_size, lookup, device=device)
        cnn_encoder = EncoderCNN(hp.rnn_hidden_size, device)
        rnn_encoder.load_state_dict(torch.load(args.restore_caption))
        cnn_encoder.load_state_dict(torch.load(args.restore_image))
        lut = construct_semantic_lookup_table(rnn_encoder, v, dataloader, device)

        with open("validation_data_lut.pkl", "wb") as _f:
            pickle.dump(lut, _f)

        # test_img = data_set.read_image("COCO_train2014_000000318556")
        # closest_captions = get_nn_captions_for_image(test_img, lut, cnn_encoder, device)
        # print("Closest Captions: ", closest_captions)

        return lut
    else:
        train(hp, lookup, dataloader, device)
        return None

if __name__ == "__main__":
    lookup_table = main()