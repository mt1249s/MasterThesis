import os
import torch
from Bio import SeqIO
#from torch.utils.data import Dataset

#print(os.getcwd())
#file = os.environ.get['DATA_PATH']#+'/H_train.fasta'
file = open('H_train.fasta')

letters = 'ACGT'
emb_dict = {letter: number+1 for number, letter in enumerate(letters)} #number+1 for emb because the padded_input_tensor is zero
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
    padded_input_tensor = torch.zeros((maxlen, batch_size), dtype=torch.long)
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
            self.targets.append(y)
            integerized_seq = []

            for index, letter, in enumerate(record.seq):
                integerized_seq.append(emb_dict[letter])
            x = torch.tensor(integerized_seq, dtype=torch.long)
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
features, label = first_ds
#print(features, label)


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl = torch.utils.data.DataLoader(ds, collate_fn=collate_seqs, batch_size=4, shuffle=True)

# convert to an iterator and look at one random sample
#dataiter = iter(dl)
#data = dataiter.next()
#features, labels = data
#print(features.shape, labels.shape)
#print(features, labels)


#crit = torch.nn.CrossEntropyLoss()

#for sample in ds:
    #print(sample)
    #break

em = torch.nn.Embedding(5, 4, 0)

for batch in dl:
    print(batch[0])
    print(batch[0].size())
    print(batch[0].max())
    print(em(batch[0]).size())
    print(em(batch[0]))
    print(batch)
    print(type(batch[0]))

    raise
                       # TODO embed batch -> [L, B, H]
    #print(batch[0])


    # TODO training goes here

    prediction = model(batch)
    loss = crit(prediction, target)
