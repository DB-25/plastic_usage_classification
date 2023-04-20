import torch
from torchvision import datasets, transforms

# Define the transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to a fixed size
    transforms.ToTensor(),   # Convert the image to a tensor
])

# Load the dataset
dataset = datasets.ImageFolder(root='./dataset/plasticClassification', transform=data_transform)

# Create a data loader for the dataset
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Calculate the mean and standard deviation of the dataset
mean = 0.
std = 0.
for images, _ in data_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(data_loader.dataset)
std /= len(data_loader.dataset)

print('Mean: {}'.format(mean))
print('Std: {}'.format(std))
