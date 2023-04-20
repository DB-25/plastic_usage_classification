import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
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
EPOCHS = 50
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8
MLP_HIDDEN_SIZES = [1024, 512, 256]
DROPOUT_PROB = [0.2, 0.1, 0.05]
LR = 0.1
MOMENTUM = 0.9
NUM_CLASSES = 4
REGULARIZATION = False
REG_LAMBDA = 0.005
LABELS = ["Heavy Plastic", "No Image", "No Plastic", "Some Plastic"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader, train_loader, val_loader, test_loader = cached_dataloader.getData(BATCH_SIZE, TRAIN_SPLIT)

# count the number of instances for each label in the entire data
data_labels = []
for _, labels in data_loader:
    data_labels += labels.tolist()
data_label_counts = dict(Counter(data_labels))

# count the number of instances for each label in the train split
train_labels = []
for _, labels in train_loader:
    train_labels += labels.tolist()
train_label_counts = dict(Counter(train_labels))

# count the number of instances for each label in the test split
test_labels = []
for _, labels in test_loader:
    test_labels += labels.tolist()
test_label_counts = dict(Counter(test_labels))

# count the number of instances for each label in the val split
val_labels = []
for _, labels in val_loader:
    val_labels += labels.tolist()
val_label_counts = dict(Counter(val_labels))

print("Data label counts:", data_label_counts)
print("Train label counts:", train_label_counts)
print("Test label counts:", test_label_counts)
print("Val label counts:", val_label_counts)

# load the pre-trained ResNet50 model
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', weights='ResNet101_Weights.DEFAULT')
model = models.resnet50(pretrained=True)

# replace the last fully connected layer with a new one for our specific classification task
in_features = model.fc.in_features
model.fc = MLP(in_channels=in_features,
               num_classes=NUM_CLASSES,
               hidden_sizes=MLP_HIDDEN_SIZES,
               dropout_probability=DROPOUT_PROB)

# Enable Fine-tuning
for params in model.parameters():
    params.requires_grad = True

# move the model to the device
model = model.to(device)

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.0005)
# optimizer = optim.Adam(model.parameters(), lr=LR)

# define the learning rate scheduler
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# define the loss function
criterion = nn.CrossEntropyLoss()


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


best_model, train_metrics = train(train_loader, val_loader, device, model, criterion, optimizer, lr_scheduler, EPOCHS)
test(test_loader, device, best_model)

# Create some sample input data
input_data = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
# Save your PyTorch model in TorchScript format
traced_model = torch.jit.trace(best_model, input_data)
traced_model.save("best_models/BestResnet50.pt")
