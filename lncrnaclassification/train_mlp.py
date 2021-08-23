import torch
from torch.utils.data import DataLoader
from models import LNCMLP
from data import Compose, OneHot, Integerize, ORF_Finder, Collator, LNCRNANetData, letter2int

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


device = torch.device('cuda')
model = LNCMLP(6 * 3000, 1, num_classes=2).to(device)
crit = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

transform = Compose([ORF_Finder(), Integerize(letter2int), OneHot(5)])
ds_train = LNCRNANetData('./H_test.fasta', transform=transform)
ds_test = LNCRNANetData('./H_test.fasta', transform=transform)
dl_train = DataLoader(ds_train, batch_size=100, collate_fn=Collator(max_len=3000), shuffle=True)
dl_test = DataLoader(ds_test, batch_size=1, collate_fn=Collator(max_len=3000), shuffle=False)

log_interval = 10
trainer = create_supervised_trainer(model, optimizer, crit, device)
metrics = {"acc": Accuracy(), 'loss': Loss(crit)}
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(dl_train)
    metrics = train_evaluator.state.metrics
    print(f"Results: - Epoch[{trainer.state.epoch}] acc: {metrics['acc']:.4f}, loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_test_results(trainer):
    test_evaluator.run(dl_test)
    metrics = test_evaluator.state.metrics
    print(f"Test Results: - Epoch[{trainer.state.epoch}] acc: {metrics['acc']:.4f}, loss: {metrics['loss']:.2f}")


trainer.run(dl_train, max_epochs=10)
