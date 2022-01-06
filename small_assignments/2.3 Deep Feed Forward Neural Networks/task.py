from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import tarfile
import re
from functools import partial
from itertools import islice
import random
import numpy as np

lines_required = 40

with tarfile.open("ecg.tar.gz", "r:gz") as t:
    is_lead_data_re = re.compile(r"^ecg/(ab)?normal/.*\.[01]$")
    is_normal_re = re.compile(r"^ecg/normal")
    last_num_re = re.compile(r"-?[0-9]+$")

    file_set = set()

    for member in t.getmembers():
        if is_lead_data_re.match(member.name) is None:
            continue

        is_normal = is_normal_re.match(member.path) is not None
        file_set.add((member.path[:-2], is_normal))

    global X
    global Y
    X, Y = [], []

    for path, is_normal in file_set:
        x = []
        y = [float(is_normal)]

        with t.extractfile(path+".0") as f:
            str_line_iter = islice(
                map(partial(str, encoding="utf-8"), f), lines_required)
            match_iter = filter(lambda match: match is not None, map(
                lambda line: last_num_re.search(line), str_line_iter))
            float_iter = map(float, map(
                lambda match: match.group(), match_iter))
            floats = list(float_iter)
            if len(floats) != lines_required:
                continue
            x = floats

        with t.extractfile(path+".1") as f:
            str_line_iter = islice(
                map(partial(str, encoding="utf-8"), f), lines_required)
            match_iter = filter(lambda match: match is not None, map(
                lambda line: last_num_re.search(line), str_line_iter))
            float_iter = map(float, map(
                lambda match: match.group(), match_iter))
            floats = list(float_iter)
            if len(floats) != lines_required:
                continue
            x.extend(floats)

        X.append(x)
        Y.append(y)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f"Using device: {device}")

# Make the program deterministic
# torch.use_deterministic_algorithms(True)
# random.seed(123456789)
generator = torch.Generator(device=device).manual_seed(123456789)
cpu_generator = torch.manual_seed(123456789)
# np.random.seed(123456789)

X = torch.tensor(X, device=device)
Y = torch.tensor(Y, device=device)

data_length = len(X)
assert(data_length > 100)
assert(len(Y) == data_length)

# shuffle data
rp = torch.randperm(len(X), generator=generator, device=device)
X = X[rp]
Y = Y[rp]

# split the data into training and test data
training_data_length = 20
X_train = X[:-training_data_length]
X_test = X[-training_data_length:]
Y_train = Y[:-training_data_length]
Y_test = Y[-training_data_length:]


train_dataset = TensorDataset(X_train, Y_train)
train_dataloader = DataLoader(
    train_dataset, batch_size=data_length-training_data_length, generator=cpu_generator)

test_dataset = TensorDataset(X_test, Y_test)
test_dataloader = DataLoader(
    test_dataset, batch_size=training_data_length, generator=cpu_generator)

hidden_layer_size = 2000


class Atan(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.atan(input)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(lines_required*2, hidden_layer_size),
            Atan(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            Atan(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            Atan(),
            nn.Linear(hidden_layer_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    num_batches = len(dataloader)
    train_loss = 0
    accuracy = torch.Tensor([0]).to(device)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        accuracy += torch.sum((pred > 0.5) == y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    accuracy /= len(dataloader) * dataloader.batch_size
    print(f"Train loss: {train_loss}")
    print(f"Train accuracy: {accuracy.item()}")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    accuracy = torch.Tensor([0]).to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            accuracy += torch.sum((pred > 0.5) == y)

            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    correct /= size
    accuracy /= len(dataloader) * dataloader.batch_size
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {accuracy.item()}")


model = NeuralNetwork().to(device)

#loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, momentum=.9, weight_decay=0.5)


def run():
    for t in range(1000):
        print(f"Epoch: {t+1}", end="\t")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        print()
    print("Done")


run()
#torch.save(model, 'model.bin')
