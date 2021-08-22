import re

import torch
from Bio import SeqIO


# NOTE '-' is padding character
letters = '-ACGT'
letter2int = {l: i for i, l in enumerate(letters)}
default_stop_codons = set(['TAA', 'TAG', 'TGA'])


class LNCRNANetData(torch.utils.data.Dataset):
    def __init__(self, file_name, transform=None, channel_last=True):
        self.samples = []
        self.targets = []

        self.transform = transform

        with open(file_name)as f:
            for record in SeqIO.parse(f, 'fasta'):
                target = 0 if 'CDS' in record.id else 1
                self.targets.append(torch.tensor(target, dtype=torch.long))
                self.samples.append(str(record.seq))

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.samples[idx]), self.targets[idx]
        return self.samples[idx], self.targets[idx]

    def max_seqlen(self):
        return max([len(seq) for seq in self.samples])

    def __len__(self):
        return len(self.samples)


def _pad_collate_seqs(batch, max_len=None, pad_val=0):
    '''
    samples is a list of length batch_size
    it may contain either tuples or torch.tensors
    a tuple in turn may contain either tuples or torch.tensors
    a tuple may only have to elements e.g. (input: tensor, target: tensor) or (input: tuple(tensor, tensor), target: tensor)
    '''
    elem = batch[0]
    # NOTE if it's a tuple unzip and recurse
    if isinstance(elem, tuple):
        assert len(elem) == 2
        first = [e[0] for e in batch]
        second = [e[1] for e in batch]
        return _pad_collate_seqs(first, max_len=max_len, pad_val=pad_val), _pad_collate_seqs(second, max_len=max_len, pad_val=pad_val)
    # NOTE if it's a tensor, check if padding required and stack
    elif isinstance(elem, torch.Tensor):
        shapes = set(tuple(t.size()) for t in batch)
        # TODO this might ignore max_len incorrectly if all sequences in the batch happen to have same length
        if len(shapes) == 1:
            # NOTE target
            return torch.stack(batch, 0)
        else:
            # NOTE
            if max_len is None:
                max_len = max(t.size(0) for t in batch)
            out = torch.full((len(batch), max_len, *(elem.size())[1:]), pad_val)
            for i in range(len(batch)):
                out[i, :batch[i].size()[0]] = batch[i][...]
            return out

    else:
        raise ValueError('unexpected batch format')


class Collator():
    def __init__(self, max_len=None, pad_val=0):
        self.max_len = max_len
        self.pad_val = pad_val

    def __call__(self, batch):
        return _pad_collate_seqs(batch, self.max_len, self.pad_val)


# TODO probably not optimal
def _find_orf(seq, stop_codons):
    frames = re.split('|'.join(stop_codons), seq)
    if len(frames) < 3:
        return ''.join([0] * len(seq))
    frame_orf_inds = '0' * len(frames[0])
    last_frame_orf_inds = '0' * (3 + len(frames[-1]))

    orfind = len(frames)
    orflen = 0
    for i, frame in enumerate(frames):
        framelen = len(frame)
        if framelen % 3 == 0:
            if orfind == len(frames) or orflen < framelen:
                orfind = i
                orflen = framelen

    for i, frame in enumerate(frames[1: -1]):
        if i + 1 == orfind:
            frame_orf_inds += '000' + '1' * len(frame)
        else:
            frame_orf_inds += '0' * (3 + len(frame))

    frame_orf_inds += last_frame_orf_inds
    frame_orf_inds = [int(x) for x in frame_orf_inds]

    return frame_orf_inds


class ORF_Finder():
    def __init__(self, stop_codons=default_stop_codons):
        self.stop_codons = stop_codons

    def __call__(self, seq):
        return seq, torch.tensor(_find_orf(seq, self.stop_codons))


# TODO naming
class Integerize():
    '''
    integerizes a sequence of letters given as string, using a given mapping
    if additional derived features are given, they are assumed to be contained in the second item of the input tuple
    '''
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, seq):
        if isinstance(seq, str):
            return torch.tensor([self.mapping[letter] for letter in seq])
        elif isinstance(seq, tuple) and len(seq) == 2:
            # NOTE sequence and orf indicator
            return torch.tensor([self.mapping[letter] for letter in seq[0]]), seq[1]
        else:
            raise ValueError


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    transform = Compose([ORF_Finder(), Integerize(letter2int)])
    ds = LNCRNANetData('./H_test.fasta', transform)
    dl = DataLoader(ds, batch_size=4, collate_fn=Collator(), shuffle=False)
    for batch in dl:
        print(batch)
        break
