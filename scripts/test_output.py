import torch
from .data import ClassificationDataset
from .data import collate_seqs


H_train = 'H_train.fasta'
H_test = 'H_test.fasta'
batch_size = 1

ds_train = ClassificationDataset(H_train)
ds_test = ClassificationDataset(H_test)
dl_train = torch.utils.data.DataLoader(ds_train, collate_fn=collate_seqs, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, collate_fn=collate_seqs, batch_size=batch_size, shuffle=True)


model_file = 'model_params.h5'
model = torch.load(model_file)

outputs = []
for batch in dl_train:
    sample_batch, _ = batch

    output = model(sample_batch)

    output = torch.nn.functional.softmax(output)

outputs = torch.tensor(outputs).squeeze()

if (outputs == torch.unsqueeze(torch.tensor([1., 0.], dtype=torch.float), 0)).all():
    print("all training samples predicted as negative")

if (outputs == torch.unsqueeze(torch.tensor([0., 1.], dtype=torch.float), 0)).all():
    print("all training samples predicted as positvive")

# TODO same for test dataset
