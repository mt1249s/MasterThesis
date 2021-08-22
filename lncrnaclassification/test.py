import torch

from data import ORF_Finder, Integerize, letter2int


def test_orf():
    orf_finder = ORF_Finder()
    x = 'TAAAAATAG'
    y = torch.tensor([int(i) for i in '000111000'], dtype=torch.long)
    assert (orf_finder(x)[1] == y).all()

    x = 'TAAAATAG'
    y = torch.zeros((len(x),), dtype=torch.long)
    assert (orf_finder(x)[1] == y).all()

    # NOTE should take the first longest frame found as orf
    x = 'TAAAAATAGAAATAG'
    y = torch.tensor([int(i) for i in '000111000000000'], dtype=torch.long)
    assert (orf_finder(x)[1] == y).all()


def test_integerize():
    integerize = Integerize(letter2int)
    x = 'AAACCCGGGTTT'
    y = torch.tensor([int(i) for i in '111222333444'], dtype=torch.long)
    assert (integerize(x) == y).all()
