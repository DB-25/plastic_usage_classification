# Yalala Mohit
# Dhruv Kamalesh Kumar

# Import libraries
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

# Set the random seed for reproducibility
random_seed = 123
torch.manual_seed(random_seed)

# Define the transforms
dataTransform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)), # Randomly crop the image to a size of 224x224
    # transforms.RandomHorizontalFlip(p=0.5), # Randomly flip the image horizontally with a probability of 0.5
    # transforms.RandomRotation(degrees=10), # Randomly rotate the image by up to 10 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust the brightness, contrast, saturation, and hue of the image
    transforms.ToTensor(), # Convert the image to a tensor
    transforms.Normalize([0.7561, 0.7166, 0.6853], [0.2465, 0.2584, 0.2781]) # Normalize the pixel values
])


# Create a custom dataset
class CachedDataset(Dataset):
    def __init__(self, data, save_dir=None):
        self.data = data
        self.cache = {}
        self.save_dir = save_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            x, y = self.data[idx]
            if isinstance(x, torch.Tensor):
                x = transforms.ToPILImage()(x)  # Convert tensor to PIL image
            transformed_x = dataTransform(x)
            if self.save_dir is not None:
                filename = 'image_{}.jpg'.format(idx)
                save_path = os.path.join(self.save_dir, filename)
                transformed_image = transforms.ToPILImage()(transformed_x)
                transformed_image.save(save_path)
            self.cache[idx] = transformed_x, y
            return transformed_x, y


# Create a custom data loader
class CachedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, save_dir=None):
        super().__init__(
            dataset=CachedDataset(dataset, save_dir=save_dir),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )
        self.save_dir = save_dir

    def collate_fn(self, batch):
        if isinstance(batch[0], tuple):
            images, labels = zip(*batch)
        else:
            images = batch
            labels = None
        images = torch.stack(images)
        if labels is not None:
            labels = torch.tensor(labels)
        return images, labels


# Create a function to get the data loaders
def getData(batch_size, train_split):
    # Set the paths for the train and validation sets
    data_dir = './dataset/plasticClassification'
    save_dir = './augmented_images'

    # Load the train and validation datasets using ImageFolder
    data = datasets.ImageFolder(data_dir, transform=dataTransform)
    # Split the train dataset to get a smaller train set and a validation set
    train_size = int(train_split * len(data))
    val_size = int((len(data) - train_size) / 2)
    test_size = len(data) - train_size - val_size
    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])
    # Create a directory to save augmented images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Create the data loaders for train and val using CachedDataLoader
    data_loader = CachedDataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                   save_dir=None)
    train_loader = CachedDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                    save_dir=None)
    val_loader = CachedDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                  save_dir=None)
    test_loader = CachedDataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                   save_dir=None)
    return data_loader, train_loader, val_loader, test_loader
