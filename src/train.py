import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(model, data_loader, criterion, optimizer, device, curr_epoch, epochs):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        criterion (nn.Module): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
    """
    total_loss = 0.0
    num_batches = 0
    print(f"--------------------------Training epoch {curr_epoch}/{epochs}--------------------------")

    for data in tqdm(data_loader, desc=f"Train epoch {curr_epoch}/{epochs}"):
        img, label = data
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    tqdm.write(f"Train loss: {avg_loss:.4f}")

    return avg_loss


def evaluate_one_epoch(model, data_loader, criterion, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during testing.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
        float: The accuracy of the model on the test data.
    """
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            img, label = data
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            total_loss += loss.item()
            num_batches += 1
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    return total_loss / num_batches, correct / total


def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    """
    Trains a given model for a specified number of epochs using the provided data loader, criterion,
    and optimizer, and tracks the loss for each epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader providing the training data.
        test_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during training and testing.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler (default is None).

    Returns:
        list: A list of the average loss per batch for each epoch.
        list: A list of the average loss per batch for each testing epoch.
        list: A list of the accuracy for each testing epoch.
    """
    avg_train_losses = []
    avg_test_losses = []
    avg_test_accuracies = []
    for i in range(num_epochs):
        model.train()
        avg_train_losses.append(train_one_epoch(model, train_loader, criterion, optimizer, device, i + 1, num_epochs))
        avg_test_loss, avg_test_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)
        avg_test_losses.append(avg_test_loss)
        avg_test_accuracies.append(avg_test_accuracy)
        if scheduler is not None:
            scheduler.step()

    return avg_train_losses, avg_test_losses, avg_test_accuracies
