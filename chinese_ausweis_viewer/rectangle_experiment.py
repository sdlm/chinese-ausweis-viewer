import copy
import time

import torch
from torch import nn, optim, cuda
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from utils import classes, datasets

TRAIN_COUNT = 15000
TEST_COUNT = 1500
EPOCHS_COUNT = 10


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    writer = SummaryWriter()
    mse_criterion = nn.MSELoss()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", 'val']:  # , 'val'
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
                labels = [l.float().to(device) for l in labels]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    coords = model(inputs)
                    predict_coords = torch.split(coords, 1, dim=1)
                    joint_loss = 0
                    mse_loss = 0
                    for k, predict_coord in enumerate(predict_coords):
                        true_coord = labels[k].unsqueeze(1)
                        joint_loss += 1/8.0 * criterion(predict_coord, true_coord)
                        mse_loss += 1/8.0 * mse_criterion(predict_coord, true_coord)

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        joint_loss.backward()
                        optimizer.step()
                        scheduler.step(epoch)

                # statistics
                running_loss += joint_loss.item() * inputs.size(0)
                running_mse_loss += mse_loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_mse_loss = running_mse_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(
                "Epoch {: >3}, {: >7} Loss: {: >7.4f}, MSE loss = {:.1f}".format(
                    epoch, phase, epoch_loss, epoch_mse_loss
                )
            )
            writer.add_scalar(f'L1Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'MSELoss/{phase}', epoch_mse_loss, epoch)

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        # print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    writer.close()

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":

    if cuda.is_available():
        print("USE GPU")
        torch.cuda.empty_cache()

    dataset_params = {"train": TRAIN_COUNT, "val": TEST_COUNT}
    image_datasets = {x: datasets.ChineseCardDataset(dataset_params[x]) for x in ["train", "val"]}
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ["train", "val"]
    }

    model = classes.ConvNet(first_conv=32, first_fc=2048, fc=8).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=60)
