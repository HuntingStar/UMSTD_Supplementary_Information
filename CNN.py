from typing import List, Optional
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.resnet import ResNetBase
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from get_data import SignalDataset
from ResNet import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def draw_loss_curve(train_loss, val_loss, save_path, name):
    # save the train and validation loss in excel file
    with open(os.path.join(save_path, name + '.txt'), 'w') as f:
        f.write("train_loss\n")
        for loss in train_loss:
            f.write(str(loss) + '\n')
        f.write("val_loss\n")
        for loss in val_loss:
            f.write(str(loss) + '\n')

    plt.figure()
    plt.title(name, fontsize=15, fontweight='bold')
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.plot(train_loss, color='blue', label="Train_loss")
    plt.plot(val_loss, color='red', label="test_loss")
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.close()



def ResNetModel(n_blocks: List[int],
                n_channels: List[int],
                bottlenecks_channels: Optional[List[int]] = None,
                in_channels: int = 3,
                first_kernel_size: int = 3):
    """
    ### Create model
    """
    # [ResNet](index.html)
    base = ResNetBase(n_blocks,
                      n_channels,
                      bottlenecks_channels,
                      in_channels=in_channels,
                      first_conv_kernel_size=first_kernel_size)
    # Linear layer for classification
    classification = nn.Linear(n_channels[-1], 12)

    # Stack them
    model = nn.Sequential(base, classification)
    # Move the model to the device
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = '7'  # &&&
    if not os.path.exists('train_9axis_12regions/' + task):
        os.mkdir('train_9axis_12regions/' + task)
    if not os.path.exists('train_9axis_12regions/' + task + '/init_train'):
        os.mkdir('train_9axis_12regions/' + task + '/init_train')
    if not os.path.exists('train_9axis_12regions/' + task + '/init_train/CNN'):
        os.mkdir('train_9axis_12regions/' + task + '/init_train/CNN')

    root_dir = 'round data/dataset_regions'
    data_CNN = np.load('round data/dataset_regions/train_dataset_CNN.npy')
    signals = data_CNN[:, :9]
    labels = data_CNN[:, 9]


    signals_train, signals_valid, labels_train, labels_valid = train_test_split(signals, labels, test_size=0.01, random_state=42, shuffle=True)

    train_dataset = SignalDataset(signals_train, labels_train)
    valid_dataset = SignalDataset(signals_valid, labels_valid)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    model = ResNetModel(n_blocks=[3, 3, 3],
                        n_channels=[32, 64, 128],   # [4, 8, 16],
                        in_channels=9,
                        first_kernel_size=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    epochs = 200
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, gt in train_loader:
            if torch.isnan(inputs).any():
                print("NaN values found in input signals")
            inputs = inputs.unsqueeze(1)
            inputs = inputs.permute(0, 2, 1)
            inputs = inputs.to(device)
            gt = gt.to(device)
            gt = gt.long() ### CrossEntropyLoss() does not need one-hot encoding
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss/len(train_loader))
        print(f'epoch: {epoch}, loss: {total_loss/len(train_loader)}')
        model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, gt in valid_loader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.permute(0, 2, 1)
                inputs = inputs.to(device)  # &&&
                gt = gt.to(device)
                outputs = model(inputs)
                ### only for validation epoch to get the max predicted index
                _, predicted = torch.max(outputs.data, 1)
                total += gt.size(0)
                correct += (predicted == gt).sum().item()
                ## visualize the confusion matrix
                if epoch == epochs-1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(gt.cpu().numpy())
            val_loss.append(criterion(outputs, gt).item())
        print(f'epoch: {epoch}, Valid Accuracy: {100*correct/total}%')

        ## Draw the confusion matrix
        if epoch == epochs-1:
            conf_mat = confusion_matrix(all_labels, all_preds)
            # Convert into accuracy percentage
            row_sums = conf_mat.sum(axis=1)
            normalized_conf_mat = (conf_mat / row_sums[:, np.newaxis])*100
            label_names = [x for x in range(12)]

            plt.figure(figsize=(10, 8))
            sns.heatmap(normalized_conf_mat, annot=True, cmap='Blues', xticklabels=range(12), yticklabels=range(12), fmt=".2f")
            plt.xlabel('Predicted Label')
            plt.xticks(ticks=range(len(label_names)), labels=label_names, rotation=45)
            plt.ylabel('True Label')
            plt.yticks(ticks=range(len(label_names)), labels=label_names, rotation=45)
            plt.title('Confusion Matrix 9axis')
            plt.savefig('train_9axis_12regions/'+ task + '/init_train/CNN/Confusion_matrix_deg.png')
            plt.show()
            plt.close()

            draw_loss_curve(train_loss, val_loss, "train_9axis_12regions/" + task + '/init_train/CNN', 'train_val_loss_deg')

    torch.save(model.state_dict(), 'train_9axis_12regions/' + task + '/init_train/CNN/model_deg.pth')
