# Yalala Mohit
# Dhruv Kamalesh Kumar

# Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

import cached_dataloader
import helper

# Transforms
transform = T.ToPILImage()
inv_normalize = T.Normalize(
    mean=[-0.7561 / 0.2465, -0.7166 / 0.2584, -0.6853 / 0.2781],
    std=[1 / 0.2465, 1 / 0.2584, 1 / 0.2781]
)

# load the best model from the file
model = torch.jit.load("./best_models/BestResnet50.pt")

# get the test data loader
_, _, _, test_loader = cached_dataloader.getData(1, 0.8)

# find the first image for each class from the test set
images = [None, None, None, None]
# holds the images after applying inverse transform
invImages = [None, None, None, None]
for image, label in test_loader:
    label = label.item()
    image = image.squeeze(0)
    if label == 0 and images[0] is None:
        invImages[0] = transform(inv_normalize(image))
        image = transform(image)
        images[0] = image
    elif label == 1 and images[1] is None:
        invImages[1] = transform(inv_normalize(image))
        image = transform(image)
        images[1] = image
    elif label == 2 and images[2] is None:
        invImages[2] = transform(inv_normalize(image))
        image = transform(image)
        images[2] = image
    elif label == 3 and images[3] is None:
        invImages[3] = transform(inv_normalize(image))
        image = transform(image)
        images[3] = image
    elif images[0] is not None and images[1] is not None and images[2] is not None and images[3] is not None:
        break

# plotting original images from data loader, before passing through the model
# basic image transformations are applied during loading the image(i.e. in the data loader)
plt.figure()
plt.suptitle("Original images (i.e. before passed through model) \n Inverse Transform v/s After Transform")
for i in range(len(images) * 2):
    plt.subplot(3, 4, i + 1)
    plt.tight_layout()
    if i % 2 == 0:
        plt.imshow(invImages[i // 2], interpolation='none')
    else:
        plt.imshow(images[i // 2], interpolation='none')
    plt.title(helper.getLabel(i // 2))
    plt.xticks([])
    plt.yticks([])

plt.savefig('./augmented_images/original_images.png')
plt.show()

# visualizing each filter and their effect on each class of images when passed through the model
# get the first convolutional layer of the model
conv1 = model.conv1

# get the weights of the first convolutional layer
weights = conv1.weight.data
weights = weights.cpu()

# get the number of filters
n_filters = weights.shape[0]

# get the number of channels
n_channels = weights.shape[1]

# get the size of the filters
filter_size = weights.shape[2]

# create a figure to plot the filters
plt.figure(figsize=(8, 8))
plt.suptitle("Filters of the first convolutional layer")
for i in range(n_filters):
    plt.subplot(8, 8, i + 1)
    plt.imshow(weights[i][0], interpolation='none', cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.autoscale(tight=True)
plt.savefig('./augmented_images/filters/conv1.png')
plt.show()

# create a figure to plot the effect of each filter on each class of images
plt.figure(figsize=(8, 64))
plt.suptitle("Effect of each filter in conv1 on each class of images")
for i in range(n_filters * 5):
    plt.subplot(n_filters, 5, i + 1)

    if i % 5 == 0:
        plt.imshow(weights[i // 5][0], interpolation='none', cmap='gray')
        plt.ylabel(str(i // 5))
    elif i % 5 == 1:
        image = cv2.filter2D(np.array(images[0]), -1, weights[i // 5][0].numpy())
        image = np.maximum(image, 0, image)
        plt.imshow(image, interpolation='none')
    elif i % 5 == 2:
        image = cv2.filter2D(np.array(images[1]), -1, weights[i // 5][0].numpy())
        image = np.maximum(image, 0, image)
        plt.imshow(image, interpolation='none')
    elif i % 5 == 3:
        image = cv2.filter2D(np.array(images[2]), -1, weights[i // 5][0].numpy())
        image = np.maximum(image, 0, image)
        plt.imshow(image, interpolation='none')
    elif i % 5 == 4:
        image = cv2.filter2D(np.array(images[3]), -1, weights[i // 5][0].numpy())
        image = np.maximum(image, 0, image)
        plt.imshow(image, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    if i // 5 == 0:
        if i == 0:
            plt.title("Filter")
        elif i == 1:
            plt.title(helper.getLabel(0))
        elif i == 2:
            plt.title(helper.getLabel(1))
        elif i == 3:
            plt.title(helper.getLabel(2))
        elif i == 4:
            plt.title(helper.getLabel(3))
plt.autoscale()
plt.savefig('./augmented_images/filters_effect/conv1.png')
plt.show()
