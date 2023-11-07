import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import CustomCIFAR10Dataset
from src.model import SimpleConvNet
from src.train import train_and_evaluate_model
from src.utils import plot_losses, plot_accuracies


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CustomCIFAR10Dataset(root="data/cifar10", train=True, transform=train_transform)
    img, label = train_set[49999]
    test_set = CustomCIFAR10Dataset(root="data/cifar10", train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = SimpleConvNet(classes=10)
    model.to(device)

    num_epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loss, test_loss, test_accuracy = train_and_evaluate_model(model, train_loader, test_loader, criterion,
                                                                    optimizer, num_epochs, device)

    plot_losses(train_loss, test_loss)
    plot_accuracies(test_accuracy)
