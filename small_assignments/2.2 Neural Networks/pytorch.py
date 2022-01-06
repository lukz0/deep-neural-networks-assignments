import re
from random import Random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# use seeded random to reproduce results
rng = Random(1)

data = []

with open('ecoli.data') as f:
    whitespace_re = re.compile(r' +')
    cp_or_im_re = re.compile(r'^(cp|im)$', re.IGNORECASE)
    for line in f:
        splitline = re.split(whitespace_re, line.strip())
        if re.match(cp_or_im_re, splitline[-1]) is not None:
            data.append(splitline)

X = [list(map(float, i[1:8])) for i in data]
Y = [1.0 if i[8] == 'im' else 0.0 for i in data]

indexes = list(range(0, 220))
rng.shuffle(indexes)
X = [X[i] for i in indexes]
Y = [Y[i] for i in indexes]

X_train = X[:200]
X_test = X[200:]
Y_train = Y[:200]
Y_test = Y[200:]

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
Y_train = torch.Tensor(Y_train).view(-1, 1)
Y_test = torch.Tensor(Y_test).view(-1, 1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert len(X) == len(Y), "tensors X and Y do not have equal length"
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 140)
        self.fc2 = nn.Linear(140, 140)
        self.fc3 = nn.Linear(140, 140)
        self.fc4 = nn.Linear(140, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


def train(net: Net, training_dataloader: DataLoader, evaluation_dataloader: DataLoader):
    optimizer = torch.optim.AdamW(
       net.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    criterion = nn.BCELoss().to(device)
    for epoch in range(1, 100001):
        train_accuracy = torch.Tensor([0]).to(device)
        loss_sum = 0.0
        for i, data in enumerate(training_dataloader, 1):
            global labels
            global inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            global outputs
            outputs = net(inputs)

            temp = (outputs > 0.5) == labels
            train_accuracy += torch.sum(temp)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_avg = loss_sum / len(training_dataloader)
        print(f"Epoch: {epoch}\tAvg loss: {loss_avg}")
        train_accuracy /= len(training_dataloader) * \
            training_dataloader.batch_size
        print(f"Training accuracy epoch {epoch}: {train_accuracy.item()}")
        evaluation_accuracy = torch.Tensor([0]).to(device)
        with torch.no_grad():
            for data in evaluation_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)

                evaluation_accuracy += torch.sum((outputs > 0.5) == labels)
        evaluation_accuracy /= len(evaluation_dataloader) * \
            evaluation_dataloader.batch_size
        print(
            f"Evaluation accuracy epoch {epoch}: {evaluation_accuracy.item()}")
        print()


train_dataset = TensorDataset(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_dataset = TensorDataset(X_test, Y_test)
test_dataloader = DataLoader(test_dataset, batch_size=20)

net = Net().to(device)
train(net, train_dataloader, test_dataloader)
