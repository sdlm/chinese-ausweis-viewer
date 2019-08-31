import argparse
import copy
import os
import time

import torch
from torch import nn, optim, cuda
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from utils import classes, datasets


MAX_SAMPLES_COUNT = 57668
TRAIN_COUNT = 2 ** 15
TEST_COUNT = 0
EPOCHS_COUNT = 20
NUM_CLASSES = 2
MODEL_PATH = './data/weights/{arch}_classification_v1_temp.pt'
MODEL_PATH_CHECKPOINT = './data/weights/{arch}_classification_v1.2.{checkpoint}.pt'


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, arch: str = None):
    since = time.time()
    writer = SummaryWriter()
    # mse_criterion = nn.MSELoss()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train"]:  # , 'val'
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_mse_loss = 0.0
            # running_corrects = 0

            max_x_coord = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # move to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    output = model(inputs)
                    loss = criterion(output, labels)

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_mse_loss += mse_loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_mse_loss = running_mse_loss / len(dataloaders[phase].dataset)
            epoch_mse_loss = 0
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(
                "Epoch {: >3}, {: >7} Loss: {: >7.4f}, MSE loss = {:.1f}".format(
                    epoch, phase, epoch_loss, epoch_mse_loss
                )
            )
            # writer.add_scalar(f'L1Loss/{phase}', epoch_loss, epoch)
            # writer.add_scalar(f'MSELoss/{phase}', epoch_mse_loss, epoch)
            writer.add_scalar(f'MSELoss/{phase}', epoch_loss, epoch)

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

            torch.save(model.state_dict(), MODEL_PATH.format(arch=arch))

        # print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    writer.close()

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history


def parse_arguments():
    messages = {
        "cnn": "Use Convolution network",
        "cnn4": "Use Convolution-5 network",
        "resnet": "Use pretrained Resnet-18 network",
        "resnet18": "Use pretrained Resnet-18 network",
        "resnet34": "Use pretrained Resnet-34 network",
        "resnet50": "Use pretrained Resnet-50 network",
        "resnet101": "Use pretrained Resnet-101 network",
        "resnet152": "Use pretrained Resnet-152 network",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", help=messages["cnn"], action="store_true")
    parser.add_argument("--cnn4", help=messages["cnn4"], action="store_true")
    parser.add_argument("--resnet", help=messages["resnet"], action="store_true")
    parser.add_argument("--resnet18", help=messages["resnet18"], action="store_true")
    parser.add_argument("--resnet34", help=messages["resnet34"], action="store_true")
    parser.add_argument("--resnet50", help=messages["resnet50"], action="store_true")
    parser.add_argument("--resnet101", help=messages["resnet101"], action="store_true")
    parser.add_argument("--resnet152", help=messages["resnet152"], action="store_true")
    args = parser.parse_args()
    if args.cnn:
        print(messages["cnn"])
        return "CNN-2"
    if args.cnn4:
        print(messages["cnn4"])
        return "CNN-4"
    if args.resnet:
        print(messages["resnet"])
        return "resnet18"
    if args.resnet18:
        print(messages["resnet18"])
        return "resnet18"
    if args.resnet34:
        print(messages["resnet34"])
        return "resnet34"
    if args.resnet50:
        print(messages["resnet50"])
        return "resnet50"
    if args.resnet101:
        print(messages["resnet101"])
        return "resnet101"
    if args.resnet152:
        print(messages["resnet152"])
        return "resnet152"
    return "CNN-2"


# noinspection PyShadowingNames
def get_model(arch: str):
    model = None
    if arch == "CNN-2":
        model = classes.ConvNet2(first_conv=32, first_fc=2048, fc=8)
        # path = MODEL_PATH.format(arch=arch)
        # net.load_state_dict(torch.load(path))
        # print(f'Succesfully load weights from {path}')
        # net.eval()
    if arch == "CNN-4":
        model = classes.ConvNet4(first_conv=32, first_fc=2048, fc=8)
        # path = MODEL_PATH.format(arch=arch)
        # net.load_state_dict(torch.load(path))
        # print(f'Succesfully load weights from {path}')
        # net.eval()
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
    if arch == "resnet34":
        model = models.resnet34(pretrained=True)
    if arch == "resnet50":
        model = models.resnet50(pretrained=True)
    if arch == "resnet101":
        model = models.resnet101(pretrained=True)
    if arch == "resnet152":
        model = models.resnet152(pretrained=True)

    if model is None:
        raise Exception('No valid network architecture selected')

    if arch.startswith("resnet"):
        # freeze weights
        freeze_layers(model, 7)

        # setup last layer with current number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

        path = MODEL_PATH.format(arch=arch)
        if os.path.exists(path):
            # noinspection PyBroadException
            try:
                model.load_state_dict(torch.load(path))
            except:  # noqa
                print(f'Got exception on loading weights from {path}')
            else:
                print(f'Successfully load weights from {path}')
                model.eval()
        else:
            print(f'Can\'t find weights by path: {path}')

    model = model.to(device)
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_layers(model, count):
    for k, child in enumerate(model.children()):
        if k == count:
            break
        for param in child.parameters():
            param.requires_grad = False


if __name__ == "__main__":

    arch = parse_arguments()

    if cuda.is_available():
        print("USE GPU")
        torch.cuda.empty_cache()

    dataset_params = {"train": TRAIN_COUNT, "val": TEST_COUNT}
    image_datasets = {x: datasets.ChineseCardClassificationDataset(dataset_params[x]) for x in ["train"]}  # , "val"
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=1024-64, shuffle=True, num_workers=4) for x in ["train"]
    }
    # 512+128+64+16

    model = get_model(arch)

    k = 1
    while True:
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=EPOCHS_COUNT, arch=arch)

        torch.save(model.state_dict(), MODEL_PATH_CHECKPOINT.format(arch=arch, checkpoint=k))
        k += 1
