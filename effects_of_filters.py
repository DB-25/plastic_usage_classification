# import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

import cached_dataloader
import helper

transform = T.ToPILImage()
inv_normalize = T.Normalize(
    mean=[-0.7561 / 0.2465, -0.7166 / 0.2584, -0.6853 / 0.2781],
    std=[1 / 0.2465, 1 / 0.2584, 1 / 0.2781]
)
# inv_normalize = T.Normalize(
#    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#    std=[1/0.229, 1/0.224, 1/0.225]
# )

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

# had a doubt with trainable params
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Total trainable params in model : ", params)
