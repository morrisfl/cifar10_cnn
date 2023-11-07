import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation losses over the course of training.
    Args:
        train_losses (list): A list of training losses.
        val_losses (list): A list of validation losses.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training Loss", marker="o", linestyle="-")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", linestyle="-")

    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracies(val_accuracies):
    """
    Plots the validation accuracies over the course of training.
    Args:
        val_accuracies (list): A list of validation accuracies.
    """
    epochs = range(1, len(val_accuracies) + 1)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o", linestyle="-")

    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
