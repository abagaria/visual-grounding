# Python imports.
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
import pdb
import random

# PyTorch imports.
import torch
import torch.utils.data as data
from torchvision import transforms

# Other imports.
import Constants
from vocab import Vocab
import os

class COCODataset(data.Dataset):
    def __init__(self, token_file_path, image_id_file_path, image_dir_path, vocab):
        super(COCODataset, self).__init__()
        self.vocab = vocab
        self.unpadded_sentences = self.read_sentences(token_file_path)
        self.padded_sentences = self.pad_sentences(self.unpadded_sentences)
        self.image_dir_path = image_dir_path
        self.image_filenames = self.read_image_filenames(image_id_file_path)
        print("image_dir_path = ", image_dir_path)

    def __len__(self):
        assert len(self.unpadded_sentences) == len(self.padded_sentences), "Padding should not change num_sentences"
        return len(self.unpadded_sentences)

    def __getitem__(self, i):
        # if random.random() > 0.2:
        sentence = deepcopy(self.padded_sentences[i])
        seq_length = torch.tensor([len(self.unpadded_sentences[i])], dtype=torch.long)
        image_filename = deepcopy(self.image_filenames[i])
        image = deepcopy(self.read_image(image_filename))
        # related = torch.tensor(1, dtype=torch.float)
        return sentence, seq_length, image
        # image_filename = deepcopy(self.image_filenames[i])
        # image = deepcopy(self.read_image(image_filename))
        # random_index = random.choice(range(len(self.unpadded_sentences)))
        # sentence = deepcopy(self.padded_sentences[random_index])
        # seq_length = torch.tensor([len(self.unpadded_sentences[random_index])], dtype=torch.long)
        # not_related = torch.tensor(-1, dtype=torch.float)
        # return sentence, seq_length, image, not_related

    @staticmethod
    def read_image_filenames(filename):
        print("Reading image filenames..")
        with open(filename, 'r') as f:
            image_ids = [line for line in tqdm(f.readlines())]
        return image_ids

    def read_sentences(self, filename):
        print("Reading sentences..")
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long)

    def read_image(self, image_name):
        image_name = image_name.strip() + ".jpg"
        img = Image.open(os.path.join(self.image_dir_path, image_name))
        coco_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        transformed_image = coco_transforms(img)
        if transformed_image.shape[0] == 1:
            target_image = torch.zeros(3, 224, 224)
            target_image[0, :, :] = transformed_image
            transformed_image = target_image
        return transformed_image

    @staticmethod
    def pad_sentences(sentences):
        pad_token = Constants.PAD
        X_lengths = [len(sentence) for sentence in sentences]
        longest_sentence = max(X_lengths)
        batch_size = len(sentences)
        padded_sentences = torch.ones(batch_size, longest_sentence, dtype=torch.long) * pad_token

        # copy over the actual sequences
        print("Padding input sentences..")
        for i, x_len in tqdm(enumerate(X_lengths)):
            sequence = sentences[i]
            padded_sentences[i, 0:x_len] = sequence[:x_len]
        return padded_sentences

def get_data_set_file_paths(mode="val"):
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    prjt_dir = os.path.join(base_dir, "visual-grounding")
    data_dir = os.path.join(prjt_dir, "data")
    anno_dir = os.path.join(data_dir, "annotations")
    image_dir = os.path.join(data_dir, "{}2014".format("val" if mode == "toyval" else mode))

    tokens_path = os.path.join(anno_dir, "captions_{}2014.toks".format(mode))
    iids_path = os.path.join(anno_dir, "imageIDs_{}2014.txt".format(mode))

    return tokens_path, iids_path, image_dir

def get_vocab_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    prjt_dir = os.path.join(base_dir, "visual-grounding")
    data_dir = os.path.join(prjt_dir, "data")
    return os.path.join(data_dir, 'coco.vocab')

if __name__ == '__main__':
    tokens_path, iids_path, image_dir = get_data_set_file_paths(mode="toyval")
    vocab_path = get_vocab_path()

    v = Vocab(filename=vocab_path, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    dset = COCODataset(tokens_path, iids_path, image_dir, v)
