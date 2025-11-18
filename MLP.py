import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import os
import seaborn as sns
import random
from matplotlib.gridspec import GridSpec
from get_data import *

class MLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, output_dim=5):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            all_preds.append(output)
            all_targets.append(target)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    accuracy = np.sum(all_preds.cpu().numpy() == all_targets.cpu().numpy())
    accuracy = accuracy / len(all_preds.cpu().numpy()) * 100
    mse = criterion(all_preds, all_targets).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    print(f'Model Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}')
    return mse, mae, accuracy

def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            all_preds.append(output)
            all_targets.append(target)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets

def calculate_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, mae, r2

def train_model(model, train_loader, num_epochs=20, learning_rate=1e-3):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    mse, mae, r2 = [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_loss = []
        batch_mse, batch_mae, batch_r2 = [], [], []

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_loss.append(loss.item())

            target = target.detach().numpy()
            output = output.detach().numpy()
            batch_mse.append(mean_squared_error(target, output))
            batch_mae.append(mean_absolute_error(target, output))
            batch_r2.append(r2_score(target, output))

        epoch_mse = sum(batch_mse) / len(batch_mse)
        epoch_mae = sum(batch_mae) / len(batch_mae)
        epoch_r2 = sum(batch_r2) / len(batch_r2)
        mse.append(epoch_mse)
        mae.append(epoch_mae)
        r2.append(epoch_r2)
        batch_loss = np.array(batch_loss).flatten().tolist()
        epoch_loss = sum(batch_loss) / len(batch_loss)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')
    return train_losses, mse, mae, r2


if __name__ == "__main__":
    mse, mae, r2 = [], [], []
    inputs, targets = [], []
    task = '1'
    if not os.path.exists('train_9axis_12regions/' + task):
        os.mkdir('train_9axis_12regions/' + task)
    if not os.path.exists('train_9axis_12regions/' + task + '/init_train'):
        os.mkdir('train_9axis_12regions/' + task + '/init_train')
    if not os.path.exists('train_9axis_12regions/' + task + '/init_train/only_MLP'):
        os.mkdir('train_9axis_12regions/' + task + '/init_train/only_MLP')

    path = os.path.join('train_9axis_12regions', task, 'init_train/only_MLP')
    root_dir = 'round data/dataset_regions'

    scaler_inputs = StandardScaler()
    scaler_targets = StandardScaler()
    inputs = scaler_inputs.fit_transform(inputs)
    targets = scaler_targets.fit_transform(targets)
    joblib.dump(scaler_inputs, f"{path}/scaler_inputs.pkl")
    joblib.dump(scaler_targets, f"{path}/scaler_targets.pkl")
    inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(inputs, targets, test_size=0.15,random_state=42, shuffle=True)
    train_dataset = SignalDataset_region(inputs_train, targets_train)
    valid_dataset = SignalDataset_region(inputs_valid, targets_valid)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=15, shuffle=False)
    epochs = 500
    mlp_model = MLP(input_dim=9, hidden_dim=128, output_dim=5)
    train_losses, train_mse, train_mae, train_r2 = train_model(mlp_model, train_loader, num_epochs=epochs)
    torch.save(mlp_model.state_dict(), f"{path}/mlp_model.pth")
    print("模型已保存")
    print("Evaluating MLP Model:")
    mlp_mse, mlp_mae, test_accuracy = evaluate_model(mlp_model, valid_loader)
    train_preds, train_targets = get_predictions(mlp_model, train_loader)
    print(f'Training Set - MSE: {sum(train_mse) / len(train_mse):.4f}, MAE: {sum(train_mae) / len(train_mae):.4f}, Aver R²: {sum(train_r2) / len(train_r2):.4f}')
    test_preds, test_targets = get_predictions(mlp_model, valid_loader)
    test_mse, test_mae, test_r2 = calculate_metrics(test_preds.numpy(), test_targets.numpy())
    print(f'Test Set - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}\n\n')

    train_preds = scaler_targets.inverse_transform(train_preds.numpy())
    train_targets = scaler_targets.inverse_transform(train_targets.numpy())

    test_preds = scaler_targets.inverse_transform(test_preds.numpy())
    test_targets = scaler_targets.inverse_transform(test_targets.numpy())

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(10, 10)
    ax0 = fig.add_subplot(gs[0:5, 0:6])
    ax0.plot(range(len(train_losses)), train_losses, label="Training Loss", color='b')
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.set_title("Training Loss Curve")
    ax0.grid()
    ax0.legend()
    ax1 = fig.add_subplot(gs[0:3, 6:])
    ax1.plot(range(len(train_mse)), train_mse, label="MSE", color='r')
    ax1.plot(range(len(train_mae)), train_mae, label="MAE", color='g')
    ax1.plot(range(len(train_r2)), train_r2, label="R²", color='b')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("")
    ax1.set_title("Training MSE, MAE and R² curve")
    ax1.grid()
    ax1.legend()
    test_data = {'Test MSE': [test_mse],
                 'Test MAE': [test_mae],
                 'Test R²': [test_r2]}
    test_data = pd.DataFrame(test_data)
    ax2 = fig.add_subplot(gs[3:5, 6:])
    ax2.axis('tight')
    ax2.axis('off')
    ax2.table(cellText=test_data.values, colLabels=test_data.columns, cellLoc='center', loc='center')
    ax2.set_title("Testing data")