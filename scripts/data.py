import os
import torch
from Bio import SeqIO
#from torch.utils.data import Dataset

#print(os.getcwd())
#file = os.environ.get['DATA_PATH']#+'/H_train.fasta'
file = open('H_train.fasta')

letters = 'ACGUT'
emb_dict = {letter: number for number, letter in enumerate(letters)}
#with open('H_test.fasta', 'r') as file:
#for record in SeqIO.parse(file, 'fasta'):
    #print(len(record))
#     integerized_seq = []
#     for index, letter, in enumerate(record.seq):
#         integerized_seq.append(emb_dict[letter])
#     x = torch.tensor(integerized_seq, dtype=torch.int)


def collate_seqs(integerized_samples): #padding
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    #print(integerized_seqs)
    maxlen = max([len(seq) for seq in integerized_seqs])
    padded_input_tensor = torch.ones((maxlen, batch_size), dtype=torch.int) * -1
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])
    return padded_input_tensor, label



# --> DataLoader can do the batch computation for us
# Implement a custom Dataset:(integerized)
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class ClassificationDataset(torch.utils.data.Dataset): #An abstract class representing a Dataset.
    def __init__(self, file): #loading targets and integerized data
        self.samples = []
        self.targets = []
        for record in SeqIO.parse(file, 'fasta'):
            label_train = 0 if 'CDS' in record.id else 1
            y = torch.tensor(label_train, dtype=torch.int)
            self.targets.append(y)  # TODO
            integerized_seq = []

            for index, letter, in enumerate(record.seq):
                integerized_seq.append(emb_dict[letter])
            x = torch.tensor(integerized_seq, dtype=torch.int)
            self.samples.append(x)

    def __getitem__(self, idx): #indexing
        # TODO transforms
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)

# create dataset
ds = ClassificationDataset(file)

# get first sample and unpack
first_ds = ds[0]
features, labels = first_ds
print(features, labels)


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl = torch.utils.data.DataLoader(ds, collate_fn=collate_seqs, batch_size=4, shuffle=True)

# convert to an iterator and look at one random sample
dataiter = iter(dl)
data = dataiter.next()
features, labels = data
print(features.shape, labels.shape)
print(features, labels)


crit = torch.nn.CrossEntropyLoss()

for sample in ds:
    #print(sample)
    break

for batch in dl:
    # TODO embed batch -> [L, B, H]
    #print(batch[0])


    # TODO training goes here

    prediction = model(batch)
    loss = crit(prediction, target)
