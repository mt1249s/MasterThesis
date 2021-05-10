import os
import torch
from Bio import SeqIO

file = os.environ['DATA_PATH'] + '/H_test.fasta'

letters = 'ACGT'
emb_dict = {letter: number for number, letter in enumerate(letters)}

# for record in SeqIO.parse(file, 'fasta'):
#     print(len(record))
#     integerized_seq = []
#     for index, letter, in enumerate(record.seq):
#         integerized_seq.append(emb_dict[letter])
#     x = torch.tensor(integerized_seq, dtype=torch.int)

def collate_seqs(integerized_samples):
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    maxlen = max([len(seq) for seq in integerized_seqs])
    padded_input_tensor = torch.ones((maxlen, batch_size), dtype=torch.int) * -1
    print(padded_input_tensor.size())
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])
    return padded_input_tensor, label

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.samples = []
        self.targets = []
        for record in SeqIO.parse(file, 'fasta'):
            self.targets.append(0) # TODO
            integerized_seq = []

            for index, letter, in enumerate(record.seq):
                integerized_seq.append(emb_dict[letter])
            x = torch.tensor(integerized_seq, dtype=torch.int)
            self.samples.append(x)

    def __getitem__(self, idx):
        # TODO transforms
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)


ds = ClassificationDataset(file)
dl = torch.utils.data.DataLoader(ds, collate_fn=collate_seqs, batch_size=4)
crit = torch.nn.CrossEntropyLoss()

for sample in ds:
    print(sample)
    break

for batch in dl:
    # TODO embed batch -> [L, B, H]
    print(batch[0])
    # TODO training goes here

    prediction = model(batch)
    loss = crit(prediction, target)
