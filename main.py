# Dhruv Kamalesh Kumar
# 08-04-2023

# import the necessary packages
import torch
import torchvision

import dataset_loader
import matplotlib.pyplot as plt
import helper
import train_and_eval

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# global variables
batch_size = 32
train_split = 0.9

# cuda check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)


# main function
def main():
    print("Starting the main function")
    # load the dataset
    trainDataset, testDataset = dataset_loader.getTrainTestDataset(batch_size, train_split)

    examples = enumerate(trainDataset)
    batch_idx, (example_data, example_targets) = next(examples)

    # visualize the data
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], interpolation='none')
        plt.title("{}".format(helper.getLabel(example_targets[i])))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # load the model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

    # change the number of classes
    num_classes = 4

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # for transfer learning
    # change the last layer
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    # move the model to the device
    model.to(device)

    # define the optimizer
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # define the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # define the number of epochs
    num_epochs = 10

    train_and_eval.evaluate(model, testDataset, device=device)
    # train the model
    for epoch in range(num_epochs):
        train_and_eval.train_one_epoch(model, optimizer, trainDataset, device, epoch, print_freq=10)
        lr_scheduler.step()
        train_and_eval.evaluate(model, testDataset, device=device)

    # return
    return


# run the main function
if __name__ == "__main__":
    main()
