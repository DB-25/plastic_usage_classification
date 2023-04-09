# Dhruv Kamalesh Kumar
# 08-04-2023

# Importing the necessary packages
import torch
import torchvision
import torchvision.transforms as transforms


# Define the transforms
dataTransform = transforms.Compose([
    transforms.Resize((800, 600)),   # Resize to a fixed size
    transforms.Pad((100, 100), fill=0),   # Pad with zeros to match the desired size
    transforms.ToTensor(),   # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # Normalize the pixel values
])


def getTrainTestDataset(batch_size, train_split):
    # load the dataset
    completeDataset = torchvision.datasets.ImageFolder("./dataset/plasticClassification/", transform=dataTransform, )
    # split the dataset
    train_size = int(train_split * len(completeDataset))
    test_size = len(completeDataset) - train_size
    trainDataset, testDataset = torch.utils.data.random_split(completeDataset, [train_size, test_size])
    # return as data loader
    return torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(
        testDataset,  shuffle=True)
