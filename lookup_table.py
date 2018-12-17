import torch
from collections import defaultdict
from tqdm import tqdm

import pdb

def construct_lookup_table(vocab, image_model, dataloader):
    """ Construct a lookup table that maps image features to a sentence in the training data. """
    print("Constructing final lookup table")
    lut = defaultdict()
    for vectorized_seq, seq_len, image in tqdm(dataloader):
        image_model.eval()
        image_features = image_model(image)

        for batch_idx in range(image_features.shape[0]):
            image_feature = image_features[batch_idx, :]
            seq = vectorized_seq[batch_idx, :]
            sentence = vocab.convertToLabels(seq.detach().numpy().tolist(), -1)
            lut[image_feature] = sentence

    return lut

def construct_semantic_lookup_table(caption_model, vocab, dataloader):
    """ Construct a lookup table that maps caption embeddings to sentences in the training data. """
    print("Constructing lookup table that maps semantic features to sentences..")
    lut = defaultdict()
    for vectorized_seq, seq_len, image in tqdm(dataloader):
        caption_model.eval()
        caption_features = caption_model(vectorized_seq, seq_len).squeeze(1)
        for batch_idx in range(caption_features.shape[0]):
            caption_feature = caption_features[batch_idx, :]
            seq = vectorized_seq[batch_idx, :]
            sentence = vocab.convertToLabels(seq.detach().numpy().tolist(), -1)
            lut[caption_feature] = sentence
    return lut

def get_nn_captions_for_image(image, lut, image_model, device):
    image_model.eval()
    test_image =image.to(device)
    image_features = image_model(test_image.unsqueeze(0))
    products = []
    for feature in lut.keys():
        product = torch.dot(image_features.squeeze(0), feature)
        products.append((product.item(), lut[feature]))
    return sorted(products, reverse=True)[:3]
