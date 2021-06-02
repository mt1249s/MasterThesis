import torch
import torch.nn as nn
from Bio import SeqIO

# We move our tensor to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = open('H_train.fasta')
# define a dic
letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb because the padded_input_tensor is zero


# padding
def collate_seqs(integerized_samples):
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    # print(integerized_seqs)
    maxlen = max([len(seq) for seq in integerized_seqs])
    padded_input_tensor = torch.zeros((maxlen, batch_size), dtype=torch.long)
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])
    return padded_input_tensor, label


class ClassificationDataset(torch.utils.data.Dataset):  # An abstract class representing a Dataset.
    def __init__(self, file):  # loading targets and integerized data
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

    def __getitem__(self, idx):  # indexing
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)


# create dataset
ds = ClassificationDataset(file)

# get first sample and unpack
first_ds = ds[0]
features, label = first_ds
# print(features, label)


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl = torch.utils.data.DataLoader(ds, collate_fn=collate_seqs, batch_size=1, shuffle=False)

# convert to an iterator and look at one random sample
# dataiter = iter(dl)
# data = dataiter.next()
# features, labels = data
# print(features.shape, labels.shape)
# print(features, labels)

num_embeddings = 5
embedding_dim = 10
em = torch.nn.Embedding(num_embeddings, embedding_dim)


# for batch in dl:
# print(batch[0])
# print(batch[0].size())
# print(batch[0].max())
# print(em(batch[0]).size())
# print(em(batch[0]))
# print(batch)
# print(batch[0].dtype)
# print(f"Device tensor is stored on: {batch[0].device}")

# raise


class RNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


input_size = embedding_dim
hidden_size = 128
n_categories = 2
rnn = RNN(input_size, hidden_size, n_categories)
'''
# one letter
for batch in dl:
    input_tensor = em(batch[0][0])
    hidden_tensor = rnn.init_hidden()
    output, next_hidden = rnn(input_tensor, hidden_tensor)
    break


#print(output.size())
#print(next_hidden.size())
'''

# one RNA seq??
for batch in dl:
    input_tensor = em(batch[0])
    hidden_tensor = rnn.init_hidden()
    print(input_tensor[0].size())
    # print(hidden_tensor.size())
    output, next_hidden = rnn(input_tensor[0], hidden_tensor)
    print(output)


# print(output.size())
# print(next_hidden.size())
    break


def category_from_output(output):
    category_idx = torch.argmax(output)
    return category_idx.item()


print(category_from_output(output))


'''
##
1. Design a model
2. Construct loss & optimizer
3. Training loop:
- forward pass (call model to predict)
- backward pass (autograd)
- update weights 

# Hyper-parameters
input_size = 
num_classes = 2
num_epochs = 2
#batch_size = 4
learning_rate = 0.001

input_size =
#sequence_length = varied
hidden_size = 128
num_layers = 2

# Design model

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# prediction = model(batch)
# loss = crit(prediction, target)

# TODO training goes here
'''

