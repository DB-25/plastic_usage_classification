# Yalala Mohit
# Dhruv Kamalesh Kumar

# Import libraries
import errno
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import models
from tqdm import tqdm

import cached_dataloader
from basemodels import MLP
from metrics import *


# Make directories for saving models and results.
def makedirectory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


makedirectory('best_models')
makedirectory('best_results')

# List available models
all_models = models.list_models()
classification_models = models.list_models(module=models)
print("All available models in Torch Vision : \n", classification_models)

# global variables
NUM_CLASSES = 4
REGULARIZATION = False
REG_LAMBDA = 0.005
LABELS = ["Heavy Plastic", "No Image", "No Plastic", "Some Plastic"]

# using gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train method - trains the model
# Input: train_loader, val_loader, device, model, criterion, optimizer, lr_scheduler, epochs
# Output: best model and train_metrics (dictionary)
def train(train_loader, val_loader, device, model, criterion, optimizer, lr_scheduler, epochs=10):
    best_val_acc = 0
    train_metrics = {"epoch": [], "num_steps": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                     "train_precision": [], "val_precision": [], "train_recall": [], "val_recall": [],
                     "train_f1score": [], "val_f1score": []}

    for epoch in range(1, epochs + 1):
        #  Storage variables
        num_steps = 0
        val_num_steps = 0
        running_loss = 0.0
        val_running_loss = 0.0
        true = []
        pred = []
        val_true = []
        val_pred = []

        # Set training mode
        model.train()

        # Train each epoch loop
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Epoch {epoch}]",
                                        ascii=' >='):
            # Zero Gradients
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            labels_one_hot = F.one_hot(labels.to(torch.int64).squeeze(), NUM_CLASSES)

            outputs = model(images).to(device)
            if (REGULARIZATION == True):
                # add L2 regularization to the loss function
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.square(param))

                loss = criterion(outputs, labels_one_hot.to(torch.float32)) + REG_LAMBDA * regularization_loss
            else:
                loss = criterion(outputs, labels_one_hot.to(torch.float32))

            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            true.extend(labels_one_hot.cpu().detach().numpy())
            pred.extend(outputs.cpu().detach().numpy())

            num_steps += 1

        # validation
        model.eval()
        with torch.no_grad():
            for i, (val_images, val_labels) in tqdm(enumerate(val_loader), total=len(val_loader),
                                                    desc=f"[Epoch {epoch}]", ascii=' >='):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_labels_one_hot = F.one_hot(val_labels.to(torch.int64).squeeze(), NUM_CLASSES)
                val_outputs = model(val_images).to(device)
                val_loss = criterion(val_outputs, val_labels_one_hot.to(torch.float32))
                val_true.extend(val_labels_one_hot.cpu().detach().numpy())
                val_running_loss += val_loss.item()

                val_pred.extend(val_outputs.cpu().detach().numpy())

                val_num_steps += 1

        # Update the LR
        lr_scheduler.step()
        print("Train True Unique ******************* ", np.unique(np.argmax(true, axis=1)))
        print("Val True Unique ******************* ", np.unique(np.argmax(val_true, axis=1)))

        print("Train Pred Unique ******************* ", np.unique(np.argmax(pred, axis=1)))
        print("Val Pred Unique ******************* ", np.unique(np.argmax(val_pred, axis=1)))

        train_acc = accuracy(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_acc = accuracy(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_precision = precision(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_precision = precision(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_recall = recall(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_recall = recall(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_f1 = f1score(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_f1 = f1score(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        print(
            f'Num_steps : {num_steps}, Learning rate : {lr_scheduler.get_last_lr()[0]}, train_loss : {running_loss / num_steps:.3f}, val_loss : {val_running_loss / val_num_steps:.3f}, train_acc : {train_acc}, val_acc : {val_acc}, train_f1score : {train_f1}, val_f1score : {val_f1}')

        if (val_acc > best_val_acc):
            best_val_acc = val_acc
            best_model = model

        train_metrics["epoch"].append(epoch)
        train_metrics["num_steps"].append(num_steps)
        train_metrics["train_loss"].append(running_loss / num_steps)
        train_metrics["val_loss"].append(val_running_loss / val_num_steps)
        train_metrics["train_acc"].append(train_acc)
        train_metrics["val_acc"].append(val_acc)
        train_metrics["train_precision"].append(train_precision)
        train_metrics["val_precision"].append(val_precision)
        train_metrics["train_recall"].append(train_recall)
        train_metrics["val_recall"].append(val_recall)
        train_metrics["train_f1score"].append(train_f1)
        train_metrics["val_f1score"].append(val_f1)

    return best_model, train_metrics


# Test method - tests the model on the test set and plots the confusion matrix
# Input - test_loader, device, model
def test(test_loader, device, model):
    true = []
    pred = []

    # Set evaluation mode
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader), desc="[Test]", ascii=' >='):
            images, labels = images.to(device), labels.to(device)
            labels_one_hot = F.one_hot(labels.to(torch.int64).squeeze(), NUM_CLASSES)
            outputs = model(images).to(device)
            true.extend(labels_one_hot.cpu().detach().numpy())
            pred.extend(outputs.cpu().detach().numpy())

    # Plot confusion matrix
    cm = confusion_matrix(np.argmax(true, axis=1), np.argmax(pred, axis=1))
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=LABELS, yticklabels=LABELS)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig('best_results/ConfusionMatrix.png')

    # Calculate and print test metrics
    test_acc = accuracy(np.argmax(true, axis=1), np.argmax(pred, axis=1))
    test_precision = precision(np.argmax(true, axis=1), np.argmax(pred, axis=1))
    test_recall = recall(np.argmax(true, axis=1), np.argmax(pred, axis=1))
    test_f1 = f1score(np.argmax(true, axis=1), np.argmax(pred, axis=1))

    print(f'Test accuracy: {test_acc:.3f}')
    print(f'Test precision: {test_precision:.3f}')
    print(f'Test recall: {test_recall:.3f}')
    print(f'Test F1 score: {test_f1:.3f}')


# Hyperparameter tuning using Optuna
def objective(trial):
    # Set Hyperparameters to tune
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])
    EPOCHS = 50
    LR = trial.suggest_categorical("LR", [0.001, 0.01, 0.1])
    MOMENTUM = trial.suggest_categorical("MOMENTUM", [0.9, 0.95, 0.99])
    WEIGHT_DECAY = trial.suggest_categorical("WEIGHT_DECAY", [0.0001, 0.001, 0.01])
    STEP_SIZE = trial.suggest_categorical("STEP_SIZE", [5, 10, 15])
    GAMMA = trial.suggest_categorical("GAMMA", [0.1, 0.5, 0.9])
    TRAIN_SPLIT = trial.suggest_categorical("TRAIN_SPLIT", [0.7, 0.8, 0.9])
    MLP_HIDDEN_SIZES = trial.suggest_categorical("MLP_HIDDEN_SIZES", [[512, 256, 128], [256, 128, 64], [128, 64, 32]])
    DROPOUT = trial.suggest_categorical("DROPOUT", [[0.05, 0.1, 0.2], [0.2, 0.1, 0.05], [0.1, 0.1, 0.1]])

    # get data loaders
    data_loader, train_loader, val_loader, test_loader = cached_dataloader.getData(BATCH_SIZE, TRAIN_SPLIT)

    # make model
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features

    model.fc = MLP(in_features, NUM_CLASSES, MLP_HIDDEN_SIZES, DROPOUT)

    for params in model.parameters():
        params.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_model, train_metrics = train(train_loader, val_loader, device, model, criterion, optimizer, lr_scheduler,
                                      EPOCHS)
    trial.set_user_attr("best_model", best_model)
    trial.set_user_attr("train_metrics", train_metrics)
    trial.set_user_attr("BATCH_SIZE", BATCH_SIZE)
    trial.set_user_attr("EPOCHS", EPOCHS)
    trial.set_user_attr("LR", LR)
    trial.set_user_attr("MOMENTUM", MOMENTUM)
    trial.set_user_attr("WEIGHT_DECAY", WEIGHT_DECAY)
    trial.set_user_attr("STEP_SIZE", STEP_SIZE)
    trial.set_user_attr("GAMMA", GAMMA)
    trial.set_user_attr("TRAIN_SPLIT", TRAIN_SPLIT)
    trial.set_user_attr("MLP_HIDDEN_SIZES", MLP_HIDDEN_SIZES)
    trial.set_user_attr("DROPOUT", DROPOUT)
    return train_metrics["val_acc"][-1]


# Run hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Get best model and hyperparameters
best_model = study.best_trial.user_attrs["best_model"]
train_metrics = study.best_trial.user_attrs["train_metrics"]
BATCH_SIZE = study.best_trial.user_attrs["BATCH_SIZE"]
EPOCHS = study.best_trial.user_attrs["EPOCHS"]
LR = study.best_trial.user_attrs["LR"]
MOMENTUM = study.best_trial.user_attrs["MOMENTUM"]
WEIGHT_DECAY = study.best_trial.user_attrs["WEIGHT_DECAY"]
TRAIN_SPLIT = study.best_trial.user_attrs["TRAIN_SPLIT"]
MLP_HIDDEN_SIZES = study.best_trial.user_attrs["MLP_HIDDEN_SIZES"]

# print best hyperparameters
print(f'Best hyperparameters: \n'
      f'BATCH_SIZE: {BATCH_SIZE} \n'
      f'EPOCHS: {EPOCHS} \n'
      f'LR: {LR} \n'
      f'MOMENTUM: {MOMENTUM} \n'
      f'WEIGHT_DECAY: {WEIGHT_DECAY} \n'
      f'TRAIN_SPLIT: {TRAIN_SPLIT} \n'
      f'MLP_HIDDEN_SIZES: {MLP_HIDDEN_SIZES} \n')

# get data loaders
data_loader, train_loader, val_loader, test_loader = cached_dataloader.getData(BATCH_SIZE, TRAIN_SPLIT)

# test best model
test(test_loader, device, best_model)

# Create some sample input data
input_data = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
# Save your PyTorch model in TorchScript format
traced_model = torch.jit.trace(best_model, input_data)
traced_model.save("best_models/BestResnet50.pt")
