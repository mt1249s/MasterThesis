# purpose: distinguish lncRNA from the cRNA
# 1. Design a model
# 2. Construct loss & optimizer
# 3. Training loop:
# - forward pass (call model to predict)
# - backward pass (autograd)
# - update weights


import torch
import torch.nn as nn
from Bio import SeqIO
from torch import utils
import matplotlib.pyplot as plt

# We move our tensor to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 2
num_epochs = 2
batch_size = 4
# learning_rate = 0.001

# input_size = ?
# sequence_length = ?
hidden_size = 128
num_layers = 1

# preparing dataset (generate integer_seq / batching & padding / embedding)
H_train = 'H_train.fasta'
H_test = 'H_test.fasta'
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
    # print(padded_input_tensor)
    # print(padded_input_tensor.size())
    return padded_input_tensor, label


class ClassificationDataset(torch.utils.data.Dataset):  # An abstract class representing a Dataset.
    def __init__(self, file_name):  # loading targets and integerized data
        self.samples = []
        self.targets = []
        with open(file_name)as fn:
            for record in SeqIO.parse(fn, 'fasta'):
                label_train = 0 if 'CDS' in record.id else 1
                y = torch.tensor(label_train, dtype=torch.long)
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
ds_train = ClassificationDataset(H_train)
ds_test = ClassificationDataset(H_test)


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl_train = torch.utils.data.DataLoader(ds_train, collate_fn=collate_seqs, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, collate_fn=collate_seqs, batch_size=batch_size, shuffle=True)

# embedding
num_embeddings = 5
embedding_dim = 6
em = torch.nn.Embedding(num_embeddings, embedding_dim)


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        # output = self.sigmoid(output)
        return output, hidden


input_size = embedding_dim
rnn = RNN(input_size, hidden_size, num_layers, num_classes)

# one letter
# Training
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()  # The negative log likelihood size:(minibatch,C)
learning_rate = 0.01
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(sample, target):
    hidden_tensor = torch.zeros(batch_size, hidden_size)
    for i in range(sample.size(0)):
        input_tensor = em(sample[i])
        prediction, hidden_tensor = rnn(input_tensor, hidden_tensor)

    # print(prediction, target)
    guess = torch.argmax(prediction, dim=1)
    # print(guess)
    loss = criterion(prediction, target)
    print(format(loss.item(), '.4f'))
    optimizer.zero_grad()  # Zero the gradients while training the network
    loss.backward()  # compute gradients
    optimizer.step()  # updates the parameters

    return guess, loss.item()


current_loss = 0
all_losses = []
plot_steps, print_steps = 50, 250
n_iters = 10500

for i in range(n_iters):
    for batch in dl_train:
        sample, target = batch
        guess, loss = train(sample, target)
        current_loss += loss

        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        if (i + 1) % print_steps == 0:
            correct = "CORRECT" if guess == target else f"WRONG ({target})"
            print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {target} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

'''
# Test model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for sample, target in dl_test:

        n_samples += target.size(0)
        n_correct += (guess == target).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the RNN network: {acc} %')
'''