# Dhruv Kamalesh Kumar
# 08-04-2023

# Importing the necessary packages
import torch
import torchvision
import torch.nn.functional as F

def train_one_epoch(model, optimizer, trainDataset, device, epoch, print_freq=10):
    # set the model to train mode
    model.train()
    # iterate over the dataset
    for batch_idx, (data, target) in enumerate(trainDataset):
        # get the data and target
        data, target = data.to(device), target.to(device)
        # zero the gradients
        optimizer.zero_grad()
        # forward pass
        output = model(data)
        # calculate the loss
        loss = F.nll_loss(output, target)
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        # print the loss
        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainDataset.dataset),
                       100. * batch_idx / len(trainDataset), loss.item()))


# define the evaluation function
def evaluate(model, testDataset, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testDataset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testDataset.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testDataset.dataset),
        100. * correct / len(testDataset.dataset)))

