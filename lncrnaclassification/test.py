import torch

from data import ORF_Finder, Integerize, letter2int, OneHot


def test_orf():
    orf_finder = ORF_Finder()
    x = 'TAAAAATAG'
    y = torch.tensor([int(i) for i in '000111000'], dtype=torch.long).unsqueeze(-1)
    print(orf_finder(x))
    assert (orf_finder(x)[1] == y).all()

    x = 'TAAAATAG'
    y = torch.zeros((len(x),), dtype=torch.long)
    assert (orf_finder(x)[1] == y).all()

    # NOTE should take the first longest frame found as orf
    x = 'TAAAAATAGAAATAG'
    y = torch.tensor([int(i) for i in '000111000000000'], dtype=torch.long).unsqueeze(-1)
    assert (orf_finder(x)[1] == y).all()

    orf_finder = ORF_Finder('TAA|TAG|TGA')
    assert (orf_finder(x)[1] == y).all()


def test_integerize():
    integerize = Integerize(letter2int)
    x = 'AAACCCGGGTTT'
    y = torch.tensor([int(i) for i in '111222333444'], dtype=torch.long)
    assert (integerize(x) == y).all()


def test_one_hot():
    orf_finder = ORF_Finder()
    integerize = Integerize(letter2int)
    one_hot = OneHot(5)

    x = 'TAAAAATAG'

    x = orf_finder(x)
    x = integerize(x)
    x = one_hot((x))

    assert (x[0, :] == torch.tensor([0., 0., 0., 0., 1., 0])).all()
