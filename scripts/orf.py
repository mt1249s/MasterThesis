'''
from Bio import SeqIO
import re
import torch
import matplotlib.pyplot as plt

H_train = 'H_train.fasta'

# define a dic
letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb because the padded_input_tensor is zero


def orf_finder(file_name):
    ORF = []
    label_orf = []
    with open(file_name) as fn:
        for record in SeqIO.parse(fn, 'fasta'):
            stop_to_stop_codons = re.split(r'TAA|TAG|TGA', str(record.seq))
            # print(stop_to_stop_codons)
            codon_len = [len(seq) for seq in stop_to_stop_codons]
            codon_len = codon_len[1:-2]
            # print(codon_len)
            for loop in codon_len:
                index_max = codon_len.index(max(codon_len))
                if (max(codon_len) % 3 == 0) & (max(codon_len) != 0):
                    orf = stop_to_stop_codons[index_max + 1]
                    # print(orf)
                    break
                codon_len[index_max] = 0
                orf = '0'
                # print(codon_len[index_max])
            ORF.append(orf)
            orfs = torch.tensor(ORF, dtype=torch.long)

            label = 0 if 'CDS' in record.id else 1
            label_orf.append(label)
            labels_orf = torch.tensor(label_orf, dtype=torch.long)

        return orfs, labels_orf


[orfs, labels_orf] = orf_finder(H_train)

'''

import numpy as np
import torch
from Bio import SeqIO
# from torch import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import train
import re
from torch_position_embedding import PositionEmbedding

torch.cuda.empty_cache()
# We move our tensor to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preparing dataset (generate integer_seq / batching & padding / embedding)
H_train = 'H_train.fasta'
H_test = 'H_test.fasta'
# define a dic
letters = 'XACGT'
emb_dict = {letter: number for number, letter in
            enumerate(letters)}  # number+1 for emb because the padded_input_tensor is zero

# padding
def collate_seqs(integerized_samples):
    ''' input; tensor of shape [b, l] '''
    ''' input: or tuple (tensor of shape [b, l] contains sequence, tensor of shape [b, l] contains ORF indicator'''
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    maxlen = max([len(seq) for seq in integerized_seqs])
    padded_input_tensor = torch.zeros((maxlen, batch_size), dtype=torch.long)
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])

    return padded_input_tensor, label


# TODO: implement string to int as transform
# TODO: implement list of ints to tensor as a transform
# TODO: implement orf finding as transform
# TODO: adapt collate function to handle orfs optionally
class ClassificationDataset(torch.utils.data.Dataset):  # An abstract class representing a Dataset.
    def __init__(self, file_name, transform=None):  # loading targets and integerized data
        self.samples = []
        self.targets = []

        self.transform = transform

        with open(file_name)as fn:
            for record in SeqIO.parse(fn, 'fasta'):
                label_train = 0 if 'CDS' in record.id else 1
                y = torch.tensor(label_train, dtype=torch.long)
                self.targets.append(y)
                integerized_seq = []

                # ORFTransform()
                stop_to_stop_codons = re.split(r'TAA|TAG|TGA', str(record.seq))
                codon_len = [len(seq) for seq in stop_to_stop_codons]
                codon_len = codon_len[1:-1]
                for _ in codon_len:
                    index_max = codon_len.index(max(codon_len))
                    if (max(codon_len) % 3 == 0) and (max(codon_len) != 0):
                        orf = stop_to_stop_codons[index_max + 1]
                        # orf = len(orf)*[1]
                        break
                    codon_len[index_max] = 0
                    orf = 'X'
                    # print(codon_len[index_max])
                    a = record.seq + orf
                    # print(f'orf: {orf}')

                # for index, letter in enumerate(record.seq):
                for index, letter in enumerate(a):
                    integerized_seq.append(emb_dict[letter])
                    # print(integerized_seq)
                x = torch.tensor(integerized_seq, dtype=torch.long)
                # print(f'x: {x}')
                self.samples.append(x)

    def __getitem__(self, idx):  # indexing
        if self.transform is not None:
            return self.transform(self.samples[idx]), self.targets[idx]
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)



stop_cods = set('TAA', 'TAG', 'TGA')
def find_orf(sequence):
    frames = re.split('|'.join(stop_cods), sequence)
    if len(frames) < 3:
        return ''.join([0]*len(sequence))
    frames = frames[1: -1]
    frame_orf_inds = []
    for frame in frames:
        # TODO finish
        pass

if __name__ == '__main__':
    # no stop codons
    ''
    print(find_orf('TAGAAATGA'))
    print(find_orf('TAGAAATGAGGGGGTAG'))
    frames = ['AAA']
    print(find_orf('AAATAGAAA'))

