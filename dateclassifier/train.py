from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import torch
from models import create_model
from datasets import TimeDataset, MiniDataset, MNISTData, split_dataset
from torch.utils.data import Dataset, DataLoader
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

dataset = MiniDataset(TimeDataset(), 200000000000)
train_dataset, test_dataset = split_dataset(dataset, 0.3)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, True)

device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda:0')
print(device)
model = create_model(dataset.classes, False).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = torch.nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

pbar = ProgressBar()
pbar.attach(trainer, ['loss'])

val_metrics = {
    "accuracy": Accuracy(),
    "crossentropy": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

@trainer.on(Events.ITERATION_COMPLETED(every=20000))
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    print('evaluating')
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics["accuracy"], metrics["crossentropy"]))
    name = f'checkpoints/model-{metrics["accuracy"]:.2f}.pt'
    torch.save(model, name)

trainer.run(train_loader, max_epochs=25)
