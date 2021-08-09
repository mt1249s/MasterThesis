# purpose: distinguish lncRNA from the cRNA
# 1. Design a model
# 2. Construct loss & optimizer
# 3. Training loop:
# - forward pass (call model to predict)
# - backward pass (calculate autograd)
# - update weights

import numpy as np
import torch
from Bio import SeqIO
from torch import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import train



# We move our tensor to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preparing dataset (generate integer_seq / batching & padding / embedding)
H_train = 'H_train.fasta'
H_test = 'H_test.fasta'
# define a dic
letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb bause the padded_input_tensor is zero


# padding
def collate_seqs(integerized_samples):
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    maxlen = max([len(seq) for seq in integerized_seqs])
    padded_input_tensor = torch.zeros((maxlen, batch_size), dtype=torch.long)
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])

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
# ds_test = ClassificationDataset(H_test)


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl_train = torch.utils.data.DataLoader(ds_train, collate_fn=collate_seqs, batch_size=train.batch_size, shuffle=True)
# dl_test = torch.utils.data.DataLoader(ds_test, collate_fn=collate_seqs, batch_size=train.batch_size, shuffle=False)


# embedding
num_embeddings = 5
embedding_dim = 6

em = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)


# Train the model
loss = 0
avg_loss = 0
all_losses = []
mean_losses = []
all_acc_train = []
dis_par = 0
all_dis_par = []

# Test the model
loss_test = 0
avg_loss_test = 0
all_losses_test = []
mean_losses_test = []
all_acc_test = []
epoch_diff_params = []

num_epochs = 20
# n_total_steps = len(dl_train)
for epoch in range(num_epochs):
    n_samples = 0
    n_correct = 0
    for batch in tqdm(dl_train):
        sample, target = batch
        sample = sample.to(device)
        target = target.to(device)
        guess, loss = train.train_model(sample, target)
        all_losses.append(loss)

        n_samples += target.size(0)
        n_correct += (guess == target).sum().item()

    diff_params = sum(train.batch_mean_dis) / len(train.batch_mean_dis)
    epoch_diff_params.append(diff_params)
'''
    with torch.no_grad():
        n_correct_test = 0
        n_samples_test = 0
        for batch in dl_test:
            sample, target = batch
            sample = sample.cuda()
            target = target.cuda()
            guess, loss_test = train.test_model(sample, target)
            all_losses_test.append(loss_test)

            n_samples_test += target.size(0)
            n_correct_test += (guess == target).sum().item()

        acc_test = 100.0 * n_correct_test / n_samples_test
        print(f'test accuracy [{epoch + 1}/{num_epochs}]: {acc_test:.4f} %')
        all_acc_test.append(acc_test)

    avg_loss_test = sum(all_losses_test) / len(all_losses_test)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss test: {avg_loss_test:.4f}')
    mean_losses_test.append(avg_loss_test)

    acc_train = 100.0 * n_correct / n_samples
    print(f'train accuracy [{epoch + 1}/{num_epochs}]: {acc_train:.4f} %')
    all_acc_train.append(acc_train)

    avg_loss = sum(all_losses) / len(all_losses)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss train: {avg_loss:.4f}')
    mean_losses.append(avg_loss)


# plot the loss and accuracy of the model
x_train = np.arange(1, num_epochs+1)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('performance of the model')

ax1.plot(x_train, mean_losses, label='train', color='tab:blue', marker='o')
ax1.plot(x_train, mean_losses_test, label='test', color='tab:red', marker='o')
ax1.set_ylabel('mean loss')
ax1.legend()

ax2.plot(x_train, all_acc_train, label='train', color='tab:blue', marker='o')
ax2.plot(x_train, all_acc_test, label='test', color='tab:red', marker='o')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.legend()
plt.show()

plt.plot(x_train, epoch_diff_params, marker='o')
plt.xlabel('epoch')
plt.ylabel('mean distance')
plt.title('distance between model parameters before and after update')
plt.text(5, 10, r'batch_size=1, lr=0.0004, Model:')
plt.show()
'''
# plot distance between weights before and after update in each epoch
plt.plot(x_train, all_dis_par)
plt.xlabel('epoch')
plt.ylabel('distance')
plt.title('distance between model parameters before and after update')
plt.text(1, .0004, r'batch_size=1, lr={learning_rate}, Model')
'''
'''