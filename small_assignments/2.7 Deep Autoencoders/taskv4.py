import functools
from typing import OrderedDict, Tuple
import zipfile
from torch.autograd.grad_mode import F
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
from itertools import repeat, islice
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torchvision
import datetime
import random

default_font = ImageFont.load_default()

device = torch.device("cuda")
#device = torch.device('cpu')


def imshow(img: torch.Tensor, s: str = ""):
    npimg = (img.clamp(0.0, 1.0).T*255.0).type(dtype=torch.uint8).cpu().numpy()
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
            torchvision.transforms.Resize(size=(128, 128)),
            torchvision.transforms.AutoAugment()
        ) if random_transform else torchvision.transforms.Resize(size=(128, 128))
        self.pil_to_tensor = torchvision.transforms.PILToTensor()
        self.tensor_to_pil = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.zi_list)

    def __getitem__(self, idx):
        zi = self.zi_list[idx]
        with self.zf.open(zi) as f:
            start_date = datetime.date(1900, 1, 1)
            end_date = datetime.date(2200, 1, 1)
            time_between_dates = end_date - start_date
            days_between_dates = time_between_dates.days
            random_number_of_days = random.randrange(days_between_dates)

            date = str(
                start_date + datetime.timedelta(days=random_number_of_days))
            text_size = default_font.getsize(date)

            image = torch.from_numpy(mpimg.imread(f))
            image: torch.Tensor = self.transformations(image.T)

            pil_img = self.tensor_to_pil(image)

            ImageDraw.Draw(pil_img).text(
                (
                    random.randrange(0, 128-text_size[0]),
                    random.randrange(0, 128-text_size[1])
                ),
                text=date,
                font=default_font,
                fill=(
                    random.randrange(0, 256),
                    random.randrange(0, 256),
                    random.randrange(0, 256)
                )
            )

            return self.pil_to_tensor(pil_img).float()/255.0, image.float()/255.0


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            # 3@128x128
            ('enc_conv1', nn.Conv2d(3, 256, 3, 2, 1)),  # 256@64x64
            ('enc_elu1', nn.ELU()),
            ('enc_bnorm1', nn.BatchNorm2d(256)),
            ('enc_conv2', nn.Conv2d(256, 256, 3, 2, 1)),  # 256@32x32
            ('enc_elu2', nn.ELU()),
            ('enc_bnorm2', nn.BatchNorm2d(256)),
            ('enc_conv3', nn.Conv2d(256, 512, 3, 2, 1)),  # 512@16x16
            ('enc_elu3', nn.ELU()),
            ('enc_bnorm3', nn.BatchNorm2d(512)),
            ('enc_conv4', nn.Conv2d(512, 1024, 3, 2, 1)),  # 1024@8x8
            ('enc_bnorm4', nn.BatchNorm2d(1024)),
            ('enc_tanh', nn.Tanh()),

        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec_convt1', nn.ConvTranspose2d(1024, 512, 2, stride=2)),  # 512@16x16
            ('dec_elu1', nn.ELU()),
            ('dec_bnorm1', nn.BatchNorm2d(512)),
            ('dec_convt2', nn.ConvTranspose2d(512, 256, 2, stride=2)),  # 256@32x32
            ('dec_elu2', nn.ELU()),
            ('dec_bnorm2', nn.BatchNorm2d(256)),
            ('dec_convt3', nn.ConvTranspose2d(256, 256, 2, stride=2)),  # 256@64x64
            ('dec_elu3', nn.ELU()),
            ('dec_bnorm3', nn.BatchNorm2d(256)),
            ('dec_convt4', nn.ConvTranspose2d(256, 3, 2, stride=2)),  # 3@128x128
            ('dec_sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class Logger:
    def __init__(self, f: io.TextIOWrapper):
        self.f = f

    def write(self, text: str):
        self.f.write(text)
        self.f.write('\n')
        print(text)
        self.f.flush()


def train(aenc: Autoencoder, training_dataloader: DataLoader, device: torch.device, l: Logger):
    aenc = aenc.train(True)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        aenc.parameters(), lr=0.001, amsgrad=True, eps=0.001)
    for epoch in range(1, 101):
        train_loss = 0.0

        for i, (modified_imgs, imgs) in enumerate(training_dataloader, 1):
            modified_imgs = modified_imgs.to(device)
            imgs = imgs.to(device)

            optimizer.zero_grad()
            predicted_imgs = aenc(modified_imgs)
            loss = criterion(predicted_imgs, imgs)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            l.write(f'Batch {i} loss: {batch_loss}')
            train_loss += batch_loss

        train_loss /= len(training_dataloader)
        l.write(f'Epoch: {epoch} Training loss: {train_loss}')
        if epoch % 10 == 0:
            torch.save(
                {
                    'autoencoder': aenc.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                f'4checkpoint{epoch}.bin'
            )


def validate(aenc: Autoencoder, validation_dataloader: DataLoader, device: torch.device):
    aenc = aenc.train(False)
    for i, (modified_imgs, imgs) in enumerate(validation_dataloader, 1):
        modified_imgs = modified_imgs.to(device)
        imgs = imgs.to(device)
        predicted_imgs = aenc(modified_imgs)
        imshow(predicted_imgs[0])
        imshow(modified_imgs[0])
        imshow(imgs[0])


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

    training_dataset = F11Dataset(zf, training_zi)  # , random_transform=True)
    evaluation_dataset = F11Dataset(zf, evaluation_zi)
    validation_dataset = F11Dataset(zf, validation_zi)
    training_dataloader = DataLoader(
        training_dataset, shuffle=True, batch_size=64)
    evaluation_dataloader = DataLoader(evaluation_dataset)
    validation_dataloader = DataLoader(
        validation_dataset, shuffle=True)

    aenc = Autoencoder().to(device)
    aenc.load_state_dict(torch.load("4checkpoint90.bin")["autoencoder"])

    with open('4.log', 'at') as f:
        l = Logger(f)
        #train(aenc, training_dataloader, device, l)
        validate(aenc, validation_dataloader, device)


if __name__ == "__main__":
    with zipfile.ZipFile("food-11.zip") as zf:
        run(zf)
