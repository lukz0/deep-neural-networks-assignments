from typing import OrderedDict, Tuple
import zipfile
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
from functools import reduce

device = "cuda"


def imshow(img: torch.Tensor, s: str = ""):
    npimg = (img.T*255.0).type(dtype=torch.uint8).numpy()
    plt.imshow(npimg)
    if s and not "\n" in s:
        s = " ".join(s.split())
        p = s.find(" ", int(len(s)/2))
        s = s[:p]+"\n"+s[p+1:]
    plt.text(0, -20, s)
    plt.show()


class F11Dataset(Dataset):
    label_re = re.compile(r"\d+")

    def __init__(self, zf: zipfile.ZipFile, zi_list: list[zipfile.ZipInfo], random_transform: bool = False):
        self.zf = zf
        self.zi_list = zi_list
        self.transformations = nn.Sequential(
            torchvision.transforms.Resize(size=(256, 256)),
            torchvision.transforms.AutoAugment()
        ) if random_transform else torchvision.transforms.Resize(size=(256, 256))

    def __len__(self):
        return len(self.zi_list)

    def __getitem__(self, idx):
        zi = self.zi_list[idx]
        with self.zf.open(zi) as f:
            image = torch.from_numpy(mpimg.imread(f))
            image: torch.Tensor = self.transformations(image.T)

            label = int(F11Dataset.label_re.search(zi.filename).group())
            return (image.type(dtype=torch.float)/255.0).contiguous(), label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        class View(nn.Module):
            def __init__(self, size: int):
                super(View, self).__init__()
                self.size = size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(-1, self.size)

        self.seq = nn.Sequential(OrderedDict([
            # 3@256x256 # 196608
            ("conv0", nn.Conv2d(3, 96, 5, 4)),  # 96@63x63
            ("gelu0", nn.GELU()),
            ("pool0", nn.MaxPool2d(2)),  # 96@31x31
            ("conv1", nn.Conv2d(96, 288, 3)),  # 288@29x29
            ("gelu1", nn.GELU()),
            ("pool1", nn.MaxPool2d(2)),  # 288@14x14
            ("conv2", nn.Conv2d(288, 576, 3)),  # 576@12x12
            ("gelu2", nn.GELU()),
            ("pool2", nn.MaxPool2d(2)),  # 576@6x6
            ("conv3", nn.Conv2d(576, 1152, 3)),  # 1152@4x4
            ("gelu3", nn.GELU()),
            ("pool3", nn.MaxPool2d(2)),  # 1152@2x2
            ("view", View(1152*2*2)),  # 4608
            ("fc0", nn.Linear(4608, 2304)),  # 2304
            ("gelu4", nn.GELU()),
            ("fc1", nn.Linear(2304, 576)),  # 576
            ("gelu5", nn.GELU()),
            ("fc2", nn.Linear(576, 11)),  # 11
            ("sigmoid", nn.Sigmoid())
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


def train(net: Net, training_dataloader: DataLoader, evaluation_dataloader: DataLoader) -> Tuple[list[int], list[int]]:
    train_losses = []
    evaluation_lossess = []

    # optimizer = torch.optim.SGD(
    #    net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.1)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    optimizer.load_state_dict(torch.load("optimizer_wd200.bin"))
    criterion = nn.CrossEntropyLoss().to(device)
    #for epoch in range(1, 101):
    for epoch in range (201, 301):
        train_accuracy = torch.Tensor([0]).to(device)
        loss_sum = 0.0
        for i, data in enumerate(training_dataloader, 1):

            inputs, labels = data
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)
            train_accuracy += torch.sum(predicted == labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
            #if i % 50 == 0:
            #    loss_avg = loss_sum / 50.0
            #    print(f"Epoch: {epoch}\tBatch: {i}\tAvg loss: {loss_avg}")
            #    train_losses.append(loss_avg)
            #    loss_sum = 0.0
        loss_avg = loss_sum / len(training_dataloader)
        print(f"Epoch: {epoch}\tAvg loss: {loss_avg}")
        train_accuracy /= len(training_dataloader) * \
            training_dataloader.batch_size
        print(f"Training accuracy epoch {epoch}: {train_accuracy.item()}")
        torch.save(net.state_dict(), f"net_wd{epoch}.bin")
        torch.save(optimizer.state_dict(), f"optimizer_wd{epoch}.bin")
        #evaluation_loss = 0.0
        evaluation_accuracy = torch.Tensor([0]).to(device)
        with torch.no_grad():
            for data in evaluation_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)

                _, predicted = torch.max(net(inputs), 1)
                evaluation_accuracy += torch.sum(predicted == labels)
                #loss = criterion(outputs, labels)
                #evaluation_loss += loss.item()
        evaluation_accuracy /= len(evaluation_dataloader) * \
            evaluation_dataloader.batch_size
        print(
            f"Evaluation accuracy epoch {epoch}: {evaluation_accuracy.item()}")
        print()
        # evaluation_lossess.append(evaluation_loss)
    return train_losses, evaluation_lossess


def validate(net: Net, validation_dataloader: DataLoader):
    with torch.no_grad():
        size = len(validation_dataloader)
        correct = 0
        for data in validation_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(net(images), 1)
            correct += reduce(
                lambda acc, elem:
                    acc + 1 if elem[0] == elem[1] else acc,
                zip(labels, predicted),
                0
            )
        print(f"Validation result: {correct}/{size}")


def run(zf: zipfile.ZipFile):
    training_zi: list[zipfile.ZipInfo] = []
    evaluation_zi: list[zipfile.ZipInfo] = []
    validation_zi: list[zipfile.ZipInfo] = []
    for zipinfo in zf.filelist:
        if zipinfo.filename.startswith("training/"):
            training_zi.append(zipinfo)
        elif zipinfo.filename.startswith("evaluation/"):
            evaluation_zi.append(zipinfo)
        elif zipinfo.filename.startswith("validation/"):
            validation_zi.append(zipinfo)

    training_dataset = F11Dataset(zf, training_zi, random_transform=True)
    evaluation_dataset = F11Dataset(zf, evaluation_zi)
    validation_dataset = F11Dataset(zf, validation_zi)
    training_dataloader = DataLoader(
        training_dataset, shuffle=True, batch_size=100)
    evaluation_dataloader = DataLoader(evaluation_dataset)
    validation_dataloader = DataLoader(validation_dataset)

    net = Net().to(device)
    net.load_state_dict(torch.load("net_wd200.bin"))
    train_lossess, evaluation_lossess = train(
        net, training_dataloader, evaluation_dataloader)
    torch.save(train_lossess, 'train_lossess.bin')
    torch.save(evaluation_lossess, 'evaluation_lossess.bin')
    # net.load_state_dict(torch.load("net67.bin"))
    #validate(net, validation_dataloader)


if __name__ == "__main__":
    with zipfile.ZipFile("food-11.zip") as zf:
        run(zf)
