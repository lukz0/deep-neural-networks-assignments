import zipfile
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import skimage.draw
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import random
from math import prod

class BalloonsDataset(Dataset):
    filename_re = re.compile(r"[^/]+$")

    def __init__(self, zf: zipfile.ZipFile, zis: list[zipfile.ZipInfo], via_region_data: dict):
        super().__init__()
        self.zf = zf
        self.zis = zis
        self.via_region_data = via_region_data

    def __len__(self):
        return len(self.via_region_data)

    def __getitem__(self, idx: int):
        zi = self.zis[idx]
        filename = next(
            BalloonsDataset.filename_re.finditer(zi.filename)
        ).group()
        via_region_data = self.via_region_data[
            next(
                filter(
                    lambda k: k.startswith(filename),
                    self.via_region_data.keys()
                )
            )
        ]
        regions = via_region_data['regions']
        with self.zf.open(self.zis[idx]) as f:
            img = (torch.from_numpy(mpimg.imread(
                f).T.swapaxes(1, 2)).to(dtype=torch.float)/255.0).float()

            _, height, width = img.shape

            length = len(regions)
            mask = np.zeros((length, height, width), dtype=np.uint8)

            boxes = []
            for i, region in enumerate(regions.values()):
                pointsX = region['shape_attributes']['all_points_x']
                pointsY = region['shape_attributes']['all_points_y']
                minX, maxX = min(pointsX), max(pointsX)
                minY, maxY = min(pointsY), max(pointsY)
                boxes.append([minX, minY, maxX, maxY])

                rr, cc = skimage.draw.polygon(
                    [y - 1 for y in region['shape_attributes']['all_points_y']],
                    [x - 1 for x in region['shape_attributes']['all_points_x']]
                )
                mask[i, rr, cc] = 1

            boxes = torch.as_tensor(boxes).float()

            labels = torch.ones((length,), dtype=torch.int64)

            mask = torch.as_tensor(mask.astype(bool), dtype=torch.uint8)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {
                'boxes': boxes,
                'image_id': torch.Tensor([idx]),
                'labels': labels,
                'masks': mask,
                'area': area,
                'iscrowd': torch.zeros((length,), dtype=torch.int64)
            }

            return img, target


def train(model: FasterRCNN, training_dataloader: DataLoader, test_dataloader: DataLoader, device: str):

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    # optimizer = torch.optim.AdamW(
    #    [p for p in model.parameters() if p.requires_grad],
    #    lr=0.001,
    #    eps=1e-3,
    #    amsgrad=True,
    #    weight_decay=0.0005
    # )
    for epoch in range(1, 11):
        print(f'Epoch {epoch}')
        loss_classifier = 0.0
        loss_box_reg = 0.0
        loss_objectness = 0.0
        loss_rpn_box_reg = 0.0
        model = model.train()
        for i, data in enumerate(training_dataloader, 1):
            # if i % 10 == 0:
            #    print(f'Batch: {i}/{len(training_dataloader)}')
            images, targets = data
            # FasterRCNN wants a list, not a tensor
            images = [image for image in images.to(device)]
            # DataLoader seems to add a dimensions to values in dictionary
            # We want a list of dictionaries instead
            targets = [{k: v[0].to(device) for k, v in targets.items()}]
            output = model(images, targets)
            loss_classifier += float(output['loss_classifier'])
            loss_box_reg += float(output['loss_box_reg'])
            loss_objectness += float(output['loss_objectness'])
            loss_rpn_box_reg += float(output['loss_rpn_box_reg'])

            losses = sum(loss for loss in output.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        loss_classifier /= len(training_dataloader) * \
            training_dataloader.batch_size
        loss_box_reg /= len(training_dataloader) * \
            training_dataloader.batch_size
        loss_objectness /= len(training_dataloader) * \
            training_dataloader.batch_size
        loss_rpn_box_reg /= len(training_dataloader) * \
            training_dataloader.batch_size
        print(f'Avg loss classifier: {loss_classifier}')
        print(f'Avg loss box reg: {loss_box_reg}')
        print(f'Avg loss objectness: {loss_objectness}')
        print(f'Avg loss rpn box reg: {loss_rpn_box_reg}')
        print()


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def test(model: FasterRCNN, test_dataloader: DataLoader, device: str):
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    #plt.imshow(result[0]['masks'][0].cpu().permute(1, 2, 0).detach())
    #plt.imshow(images[0].cpu().permute(1, 2, 0).detach())
    for images, targets in iter(test_dataloader):
        #images, targets = next(iter(test_dataloader))
        targets
        images = [image for image in images.to(device)]
        targets = [{k: v[0].to(device) for k, v in targets.items()}]

        result = model(images)[0]
        indexes = list(i.item() for i in torch.where(result['scores'] > 0.5)[0])
        image = images[0].detach().permute(1, 2, 0).cpu()
        #plt.imshow(image)
        grayimg = rgb2gray(image)
        shape = grayimg.shape
        shape = [*shape, 1]
        grayimg = grayimg.reshape(shape)
        grayimg = np.concatenate([grayimg, grayimg, grayimg], axis=2)
        #plt.imshow(grayimg)
        #print(rgb2gray(image).shape)
        masks = result['masks'][indexes].detach().permute(0, 2, 3, 1).cpu()
        for i, mask in enumerate(masks):
            grayimg = np.where(mask > 0.5, image, grayimg)
            #red = np.repeat(random.random(), prod(mask.shape)).reshape(mask.shape)
            #green = np.repeat(random.random(), prod(mask.shape)).reshape(mask.shape)
            #blue = np.repeat(random.random(), prod(mask.shape)).reshape(mask.shape)
            #alpha = mask
            #mask = np.concatenate([red, green, blue, alpha], axis=2)
            #plt.imshow(mask)
        plt.imshow(grayimg)
        plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zf = zipfile.ZipFile("balloon_dataset.zip")
    train_zis = []
    test_zis = []

    train_re = re.compile(r"^balloon/train/.*.jpg$")
    test_re = re.compile("^balloon/val/.*.jpg$")

    for zi in zf.filelist:
        if train_re.match(zi.filename):
            train_zis.append(zi)
        elif test_re.match(zi.filename):
            test_zis.append(zi)

    train_via_region_data, test_via_region_data = None, None

    with zf.open("balloon/train/via_region_data.json") as f:
        train_via_region_data = json.load(f)

    with zf.open("balloon/val/via_region_data.json") as f:
        test_via_region_data = json.load(f)

    # load a model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    #    pretrained=True
    # )
#
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # num_classes = 2  # 1 class (baloon) + background
    # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   hidden_layer,
    #                                                   num_classes)
    model = get_model_instance_segmentation(2)

    model = model.to(device)

    training_dataset = BalloonsDataset(
        zf, train_zis, train_via_region_data)
    test_dataset = BalloonsDataset(
        zf, test_zis, test_via_region_data)
    training_dataloader = DataLoader(
        training_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset)
    #train(model, training_dataloader, test_dataloader, device)

    #torch.save(model.state_dict(), 'model.bin')
    model.load_state_dict(torch.load('model.bin'))

    model = model.eval()

    #global result
    #global images
    #global targets
    #images, targets = next(iter(test_dataloader))
    # targets
    #images = [image for image in images.to(device)]
    #targets = [{k: v[0].to(device) for k, v in targets.items()}]
    #
    #result = model(images)
    test(model, test_dataloader, device)


if __name__ == '__main__':
    main()
